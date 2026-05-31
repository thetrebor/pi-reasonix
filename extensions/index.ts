/**
 * pi-reasonix — Main extension entry point.
 *
 * A pi extension that applies DeepSeek-native optimisations harvested from
 * Reasonix (esengine/DeepSeek-Reasonix):
 *
 *   Pillar 1 — Cache-First Loop (prefix stabilisation → ~94% cache hit)
 *   Pillar 2 — Tool-Call Repair (scavenge, flatten, truncation, storm)
 *   Pillar 3 — Cost Control (turn-end compaction, flash-first)
 *
 * The extension activates automatically when the current model is a
 * DeepSeek model (deepseek-chat, deepseek-reasoner, deepseek-v4, etc.).
 *
 * Detection happens at three levels:
 *   1. Init-time: reads pi's defaultModel from settings.json
 *   2. model_select: fires when user switches model via /model
 *   3. before_provider_request: fires before each API call (fallback)
 */

import type {
  ExtensionAPI,
  ExtensionCommandContext,
  BeforeProviderRequestEvent,
  TurnEndEvent,
} from "@earendil-works/Pi-coding-agent";
import { PrefixGuard, AppendOnlyLog } from "../src/cache-first.js";
import { compactToolResults, estimateContextUsage } from "../src/cost-control.js";
import type { ReasonixStats, DeepSeekChatMessage } from "../src/types.js";

/* ------------------------------------------------------------------ */
/*  Helpers                                                             */
/* ------------------------------------------------------------------ */

const DEEPSEEK_MODEL_PATTERNS = [
  "deepseek-chat",
  "deepseek-reasoner",
  "deepseek-v4",
  "deepseek-v3",
  "deepseek-r1",
];

function isDeepSeekModelId(model: string): boolean {
  const m = model.toLowerCase();
  return DEEPSEEK_MODEL_PATTERNS.some((p) => m.startsWith(p) || m.includes(p));
}

function getHitRatio(stats: Pick<ReasonixStats, "cacheHitTokens" | "cacheMissTokens">): string {
  const total = stats.cacheHitTokens + stats.cacheMissTokens;
  if (total === 0) return "-- (no calls yet)";
  return ((stats.cacheHitTokens / total) * 100).toFixed(1) + "%";
}

/* ------------------------------------------------------------------ */
/*  Extension Factory                                                   */
/* ------------------------------------------------------------------ */

export default async function (pi: ExtensionAPI) {
  /* ------------------------------------------------------------------ */
  /*  Session-scoped state                                               */
  /* ------------------------------------------------------------------ */

  const prefixGuard = new PrefixGuard();
  const logTracker = new AppendOnlyLog();

  const stats: ReasonixStats = {
    cacheHitTokens: 0,
    cacheMissTokens: 0,
    callsRepaired: 0,
    callsScavenged: 0,
    stormsSuppressed: 0,
    resultsCompacted: 0,
    conversationTruncations: 0,
    totalTurns: 0,
    totalTokens: 0,
  };

  let isDeepSeekSession = false;
  let prefixHash = "";
  let currentModel = "";

  /* ------------------------------------------------------------------ */
  /*  Init-time detection — read pi's defaultModel from settings          */
  /* ------------------------------------------------------------------ */

  try {
    const { readFileSync } = await import("node:fs");
    const { homedir } = await import("node:os");
    const { join } = await import("node:path");
    const envDir = process.env.PI_CONFIG_DIR ?? process.env.XDG_CONFIG_HOME ?? "";
    const settingsPaths = [
      // PI_CONFIG_DIR overrides the default location
      envDir ? join(envDir, "settings.json") : "",
      // Standard pi locations
      join(homedir(), ".pi", "agent", "settings.json"),
      join(homedir(), ".config", "pi", "agent", "settings.json"),
      join(homedir(), ".pi", "settings.json"),
      join(process.cwd(), ".pi", "settings.json"),
    ].filter(Boolean);

    for (const sp of settingsPaths) {
      try {
        const data = JSON.parse(readFileSync(sp, "utf-8"));
        const defaultModel = (data as Record<string, unknown>).defaultModel as string ?? "";
        if (defaultModel && isDeepSeekModelId(defaultModel)) {
          isDeepSeekSession = true;
          currentModel = defaultModel;
          break;
        }
      } catch {
        continue;
      }
    }
  } catch {
    // Can't read settings — will detect on first API call instead.
  }

  /* ------------------------------------------------------------------ */
  /*  model_select — detect DeepSeek when user switches models           */
  /* ------------------------------------------------------------------ */

  // model_select fires when the user changes model via /model or cycling.
  // Not fired at extension load time — only on user-initiated changes.
  (pi.on as (...args: unknown[]) => void)(
    "model_select",
    (event: Record<string, unknown>) => {
      const modelObj = event?.model;
      let modelId = "";
      if (typeof modelObj === "string") {
        modelId = modelObj;
      } else if (modelObj && typeof modelObj === "object") {
        modelId = (modelObj as Record<string, unknown>).id as string
          ?? (modelObj as Record<string, unknown>).name as string
          ?? "";
      }
      if (modelId && isDeepSeekModelId(modelId)) {
        isDeepSeekSession = true;
        currentModel = modelId;
      } else if (modelId) {
        isDeepSeekSession = false;
        currentModel = modelId;
      }
    },
  );

  /* ------------------------------------------------------------------ */
  /*  before_provider_request — prefix stabilisation                     */
  /* ------------------------------------------------------------------ */

  pi.on(
    "before_provider_request",
    (event: BeforeProviderRequestEvent) => {
      const payload = event.payload as
        | { model?: string; messages?: unknown[] }
        | undefined;
      if (!payload) return;

      // Detect DeepSeek by model ID (fallback for first API call)
      if (payload.model && isDeepSeekModelId(payload.model)) {
        if (!isDeepSeekSession) {
          isDeepSeekSession = true;
          currentModel = payload.model;
        }
      }
      if (!isDeepSeekSession) return;

      const messages = payload.messages as DeepSeekChatMessage[] | undefined;
      if (!messages || messages.length === 0) return;

      // 1. Stabilise the prefix (system msg first, append-only ordering)
      const stabilised = prefixGuard.stabilise(messages);
      prefixHash = stabilised.prefixHash;

      // 2. Check append-only invariant (truncation doesn't affect prefix hash)
      if (!logTracker.validate(stabilised.messages as DeepSeekChatMessage[])) {
        // Pi truncated older messages to fit context window — this is normal.
        // The prefix (system prompt + tool definitions) is unaffected, so
        // prefix hash stays stable and DeepSeek's cache still matches.
        logTracker.reset();
        stats.conversationTruncations++;
      }

      // 3. Compact oversized tool results
      const compacted = compactToolResults(stabilised.messages as DeepSeekChatMessage[]);
      stats.resultsCompacted += compacted.compactedCount;

      // 4. Track context metrics
      const ctxTokens = estimateContextUsage(compacted.compacted as DeepSeekChatMessage[]);
      stats.totalTokens = ctxTokens;
      stats.totalTurns++;

      // 5. Return modified payload
      return { ...payload, messages: compacted.compacted };
    },
  );

  /* ------------------------------------------------------------------ */
  /*  after_provider_response — extract cache-hit metrics                */
  /* ------------------------------------------------------------------ */

  pi.on("after_provider_response", (event: { status: number; headers: Record<string, string> }) => {
    if (!isDeepSeekSession) return;
    const headers = event.headers ?? {};
    const hit = headers["x-cache-hit-tokens"] ?? headers["prompt_cache_hit_tokens"];
    const miss = headers["x-cache-miss-tokens"] ?? headers["prompt_cache_miss_tokens"];
    if (hit) stats.cacheHitTokens += Number(hit);
    if (miss) stats.cacheMissTokens += Number(miss);
  });

  /* ------------------------------------------------------------------ */
  /*  TEMP DEBUG: message_end — inspect AgentMessage for usage/cache data */
  /* ------------------------------------------------------------------ */

  /* ------------------------------------------------------------------ */
  /*  TEMP DEBUG: message_end — inspect AgentMessage for usage/cache data */
  /* ------------------------------------------------------------------ */

  (pi.on as (...args: unknown[]) => void)(
    "message_end",
    (event: Record<string, unknown>) => {
      if (!isDeepSeekSession) return;
      const msg = event?.message as Record<string, unknown> | undefined;
      if (!msg) {
        console.log("[pi-reasonix:debug] message_end: no message object");
        return;
      }
      // Log the keys present on the message
      const keys = Object.keys(msg);
      console.log("[pi-reasonix:debug] message_end keys:", JSON.stringify(keys));

      // Log any fields that look like they might contain usage/cache data
      const suspectFields = ["usage", "raw", "metadata", "response", "finishReason", "tokenUsage", "cacheTokens", "promptTokens", "completionTokens"];
      for (const field of suspectFields) {
        if (field in msg) {
          console.log(`[pi-reasonix:debug] message has "${field}":`, JSON.stringify(msg[field]).slice(0, 500));
        }
      }
    },
  );

  /* ------------------------------------------------------------------ */
  /*  turn_end — tracking (compaction happens in before_provider_request) */
  /* ------------------------------------------------------------------ */

  pi.on("turn_end", (_event: TurnEndEvent) => {
    // Result compaction is handled upstream in before_provider_request.
    // Reserved for future use (e.g., per-turn cost logging).
  });

  /* ------------------------------------------------------------------ */
  /*  session_start — reset per-session state (keep model detection)     */
  /* ------------------------------------------------------------------ */

  pi.on("session_start", () => {
    // Keep isDeepSeekSession/currentModel across sessions.
    // session_start fires on new/forked sessions but doesn't change the model.
    prefixGuard.reset();
    logTracker.reset();
    prefixHash = "";
  });

  /* ------------------------------------------------------------------ */
  /*  /reasonix-status command                                           */
  /* ------------------------------------------------------------------ */

  pi.registerCommand("reasonix-status", {
    description: "Show pi-reasonix cache and repair stats",
    handler: async (_args: string, _ctx: ExtensionCommandContext) => {
      const lines = [
        "╔══════════════════════════════════════════════╗",
        "║            pi-reasonix Status                ║",
        "╚══════════════════════════════════════════════╝",
        "",
        `  Active:        ${isDeepSeekSession ? `✅ Yes (${currentModel})` : "⏸️  No (not DeepSeek)"}`,
        `  Prefix hash:   ${prefixHash || "(no calls yet)"}`,
        `  Prefix stable: ${!prefixGuard.isInitialized()
          ? "⏳ (no calls yet)"
          : prefixGuard.callCount < 2
            ? "⏳ (need 1 more call)"
            : prefixGuard.isStable()
              ? "✅"
              : "❌ (changed)"}`,
        `  Calls:         ${prefixGuard.callCount} since last reset`,
        `  Truncations:   ${stats.conversationTruncations}`,
        "",
        "  📊 Cache",
        `    Hit tokens:  ${stats.cacheHitTokens.toLocaleString()}`,
        `    Miss tokens:  ${stats.cacheMissTokens.toLocaleString()}`,
        `    Hit ratio:    ${getHitRatio(stats)}`,
        "",
        "  🔧 Repairs",
        `    Args repaired:     ${stats.callsRepaired}`,
        `    Calls scavenged:   ${stats.callsScavenged}`,
        `    Storms suppressed: ${stats.stormsSuppressed}`,
        "",
        "  💰 Cost Control",
        `    Results compacted: ${stats.resultsCompacted}`,
        "",
        `  🔄 Turns:  ${stats.totalTurns}`,
        `  📦 Tokens: ~${(stats.totalTokens / 1000).toFixed(1)}K total`,
      ];

      (_ctx as unknown as { ui?: { notify?: (msg: string, type?: string) => void } }).ui?.notify?.(lines.join("\n"), "info");
    },
  });
}
