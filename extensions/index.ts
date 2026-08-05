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
import {
  compactToolResults,
  estimateContextUsage,
} from "../src/cost-control.js";
import {
  repairTruncatedJSON,
  scavengeToolCalls,
  detectCallStorm,
} from "../src/repair.js";
import type { ReasonixStats, DeepSeekChatMessage } from "../src/types.js";

/* ------------------------------------------------------------------ */
/*  Config                                                              */
/* ------------------------------------------------------------------ */

/**
 * Auto-append tool calls scavenged from reasoning content.
 * Off by default: scavenging mutates the tool-call list, so it is opt-in
 * until it has been validated on a real DeepSeek session.
 */
const SCAVENGE_ENABLED =
  (process.env.REASONIX_SCAVENGE ?? "0") === "1";

/* ------------------------------------------------------------------ */
/*  Configuration (env vars, mirroring PI_HARNESS_* convention)         */
/* ------------------------------------------------------------------ */

function envBool(name: string, def: boolean): boolean {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return def;
  return !["0", "false", "no", "off", ""].includes(raw.toLowerCase());
}

const REASONIX_CONFIG = {
  /** Master switch. When false, no hooks fire. */
  enabled: envBool("PI_REASONIX_ENABLED", true),
  /** Pillar 1 — cache prefix stabilization (system-first reorder + hash tracking). */
  cache: envBool("PI_REASONIX_CACHE", true),
  /** Pillar 3 — tool-result compaction + context tracking. */
  cost: envBool("PI_REASONIX_COST", true),
  /** Cache metric extraction (after_provider_response + message_end). */
  metrics: envBool("PI_REASONIX_METRICS", true),
};

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

function getHitRatio(
  stats: Pick<ReasonixStats, "cacheHitTokens" | "cacheMissTokens">,
): string {
  const total = stats.cacheHitTokens + stats.cacheMissTokens;
  if (total === 0) return "-- (no calls yet)";
  return ((stats.cacheHitTokens / total) * 100).toFixed(1) + "%";
}

/** Extract a deterministic key for a toolCall block (name + serialized args). */
function toolCallKey(
  name: string | undefined,
  args: unknown,
): string {
  const argsKey =
    typeof args === "string"
      ? args
      : JSON.stringify(args ?? {});
  return `${name ?? ""}|${argsKey}`;
}

/* ------------------------------------------------------------------ */
/*  Extension Factory                                                   */
/* ------------------------------------------------------------------ */

export default async function (pi: ExtensionAPI) {
  if (!REASONIX_CONFIG.enabled) {
    console.log("[pi-reasonix] Disabled (PI_REASONIX_ENABLED=0).");
    return;
  }
  console.log(
    `[pi-reasonix] Loaded. cache=${REASONIX_CONFIG.cache ? "on" : "off"} cost=${REASONIX_CONFIG.cost ? "on" : "off"} metrics=${REASONIX_CONFIG.metrics ? "on" : "off"}`,
  );

  /* ------------------------------------------------------------------ */
  /*  Session-scoped state                                               */
  /* ------------------------------------------------------------------ */

  const prefixGuard = new PrefixGuard();
  const logTracker = new AppendOnlyLog();

  const stats: ReasonixStats = {
    cacheHitTokens: 0,
    cacheMissTokens: 0,
    cacheWriteTokens: 0,
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

  /**
   * Header-derived cache tokens for the current provider response.
   * Held back until message_end: usage fields (when present) are preferred,
   * and headers are only applied if no usage arrived — prevents the same
   * response from being counted twice.
   */
  let pendingHeaderTokens: { hit: number; miss: number } | null = null;

  /* ------------------------------------------------------------------ */
  /*  Init-time detection — read pi's defaultModel from settings          */
  /* ------------------------------------------------------------------ */

  try {
    const { readFileSync } = await import("node:fs");
    const { homedir } = await import("node:os");
    const { join } = await import("node:path");
    const envDir =
      process.env.PI_CONFIG_DIR ?? process.env.XDG_CONFIG_HOME ?? "";
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
        const defaultModel =
          (data as Record<string, unknown>).defaultModel as string ?? "";
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
        modelId =
          (modelObj as Record<string, unknown>).id as string ??
          (modelObj as Record<string, unknown>).name as string ??
          "";
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
        | { model?: string; messages?: unknown[]; tools?: unknown[] }
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

      let working = messages;

      // 1. Stabilise the prefix: system first, and hash the cache head
      //    (system message + tool *definitions*), not the growing history.
      if (REASONIX_CONFIG.cache) {
        const stabilised = prefixGuard.stabilise(
          working,
          payload.tools as unknown[] | undefined,
        );
        prefixHash = stabilised.prefixHash;
        working = stabilised.messages;

        // 2. Check append-only invariant (truncation doesn't affect prefix hash)
        if (!logTracker.validate(working as DeepSeekChatMessage[])) {
          // Pi truncated older messages to fit the context window. The cache
          // head (system + tools) survives; the truncated conversation bulk is
          // a cache miss on the next request — counted for the status display.
          logTracker.reset();
          stats.conversationTruncations++;
        }
      }

      // 3. Compact oversized tool results (head+tail preserving)
      if (REASONIX_CONFIG.cost) {
        const compacted = compactToolResults(working as DeepSeekChatMessage[]);
        stats.resultsCompacted += compacted.compactedCount;
        working = compacted.compacted;
      }

      // 4. Track context metrics
      const ctxTokens = estimateContextUsage(working as DeepSeekChatMessage[]);
      stats.totalTokens = ctxTokens;
      stats.totalTurns++;

      // 5. Return modified payload
      return { ...payload, messages: working };
    },
  );

  /* ------------------------------------------------------------------ */
  /*  after_provider_response — stash header cache tokens                */
  /* ------------------------------------------------------------------ */

  pi.on(
    "after_provider_response",
    (event: { status: number; headers: Record<string, string> }) => {
      if (!isDeepSeekSession || !REASONIX_CONFIG.metrics) return;
      const headers = event.headers ?? {};
      const hit =
        headers["x-cache-hit-tokens"] ?? headers["prompt_cache_hit_tokens"];
      const miss =
        headers["x-cache-miss-tokens"] ?? headers["prompt_cache_miss_tokens"];
      if (hit || miss) {
        // Stash, don't add — message_end usage fields are preferred and
        // applying both would double-count the same response.
        pendingHeaderTokens = {
          hit: Number(hit) || 0,
          miss: Number(miss) || 0,
        };
      }
    },
  );

  /* ------------------------------------------------------------------ */
  /*  message_end — extract cache metrics + repair model tool calls      */
  /* ------------------------------------------------------------------ */

  (pi.on as (...args: unknown[]) => void)(
    "message_end",
    (event: Record<string, unknown>) => {
      const msg = event?.message as Record<string, unknown> | undefined;
      if (!msg) return;

      /* ---- cache metrics (assistant messages carry usage) ---- */
      if (REASONIX_CONFIG.metrics && isDeepSeekSession && msg.role === "assistant") {
        const usage = msg.usage as Record<string, unknown> | undefined;
        if (usage) {
          // OpenCode format: usage.cacheRead, usage.cacheWrite
          // DeepSeek format: usage.prompt_cache_hit_tokens, ...
          const cacheRead =
            (usage.cacheRead as number) ??
            (usage.prompt_cache_hit_tokens as number) ??
            0;
          const cacheWrite =
            (usage.cacheWrite as number) ??
            (usage.prompt_cache_write_tokens as number) ??
            0;
          const totalInput =
            (usage.input as number) ?? (usage.prompt_tokens as number) ?? 0;

          if (cacheRead > 0) stats.cacheHitTokens += cacheRead;
          if (cacheWrite > 0) stats.cacheWriteTokens += cacheWrite;
          if (totalInput > 0) {
            const missTokens = Math.max(0, totalInput - cacheRead);
            stats.cacheMissTokens += missTokens;
          }
          // Usage is authoritative for this response; discard header stash.
          pendingHeaderTokens = null;
        } else if (pendingHeaderTokens) {
          // No usage fields — fall back to the response headers.
          stats.cacheHitTokens += pendingHeaderTokens.hit;
          stats.cacheMissTokens += pendingHeaderTokens.miss;
          pendingHeaderTokens = null;
        }
      }

      /* ---- tool-call repair (only assistant tool-call messages) ---- */
      if (!isDeepSeekSession || msg.role !== "assistant") return;
      const content = msg.content;
      if (!Array.isArray(content)) return;
      const toolCalls = content.filter(
        (c: Record<string, unknown>) => c?.type === "toolCall",
      );
      if (toolCalls.length === 0) return;

      try {
        // Pass 1 — repair truncated JSON arguments (string args only).
        for (const tc of toolCalls as Array<Record<string, unknown>>) {
          if (typeof tc.arguments === "string") {
            const { repaired, fixed } = repairTruncatedJSON(tc.arguments);
            if (fixed) {
              tc.arguments = repaired;
              stats.callsRepaired++;
            }
          }
        }

        // Pass 2 — scavenge tool calls leaked into reasoning content.
        // Opt-in (REASONIX_SCAVENGE=1): appending calls mutates the batch.
        if (SCAVENGE_ENABLED) {
          const reasoning =
            (msg.reasoning as string | null | undefined) ??
            (msg.reasoning_content as string | null | undefined) ??
            null;
          if (reasoning) {
            const scavenged = scavengeToolCalls(String(reasoning));
            const existing = new Set(
              (toolCalls as Array<Record<string, unknown>>).map(
                (tc) => tc.id as string,
              ),
            );
            for (const call of scavenged) {
              if (existing.has(call.id)) continue;
              content.push({
                type: "toolCall",
                id: call.id,
                name: call.function.name,
                arguments: call.function.arguments,
              });
              existing.add(call.id);
              stats.callsScavenged++;
            }
          }
        }

        // Pass 3 — storm suppression: drop exact (name, args) repeats
        // within a sliding window. Rebuilt with detectCallStorm semantics.
        const inputs = (toolCalls as Array<Record<string, unknown>>).map(
          (tc) => ({
            id: tc.id as string,
            type: "function",
            function: {
              name: tc.name as string,
              arguments:
                typeof tc.arguments === "string"
                  ? (tc.arguments as string)
                  : JSON.stringify(tc.arguments ?? {}),
            },
          }),
        );
        const { clean, stormCount } = detectCallStorm(inputs, 5);
        if (stormCount > 0) {
          const cleanIds = new Set(clean.map((c) => c.id));
          // Rebuild content keeping non-toolCall blocks untouched.
          msg.content = (content as Array<Record<string, unknown>>).filter(
            (c) => c.type !== "toolCall" || cleanIds.has(c.id as string),
          );
          stats.stormsSuppressed += stormCount;
        }
      } catch {
        // Repair must never break the agent loop.
      }
    },
  );

  /* ------------------------------------------------------------------ */
  /*  turn_end — apply stashed header tokens if no usage arrived         */
  /* ------------------------------------------------------------------ */

  pi.on("turn_end", (_event: TurnEndEvent) => {
    if (pendingHeaderTokens) {
      stats.cacheHitTokens += pendingHeaderTokens.hit;
      stats.cacheMissTokens += pendingHeaderTokens.miss;
      pendingHeaderTokens = null;
    }
  });

  /* ------------------------------------------------------------------ */
  /*  session_start — reset per-session state (keep model detection)     */
  /* ------------------------------------------------------------------ */

  pi.on("session_start", () => {
    // Keep isDeepSeekSession/currentModel across sessions.
    // session_start fires on new/forked sessions but doesn't change the model.
    if (REASONIX_CONFIG.cache) {
      prefixGuard.reset();
      logTracker.reset();
      prefixHash = "";
    }
    pendingHeaderTokens = null;
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
        `    Miss tokens: ${stats.cacheMissTokens.toLocaleString()}`,
        `    Write tokens: ${stats.cacheWriteTokens.toLocaleString()}`,
        `    Hit ratio:    ${getHitRatio(stats)}`,
        "",
        "  🔧 Repairs",
        `    Args repaired:     ${stats.callsRepaired}`,
        `    Calls scavenged:   ${stats.callsScavenged}`,
        `    Storms suppressed: ${stats.stormsSuppressed}`,
        "",
        "  💰 Cost Control",
        `    Results compacted: ${stats.resultsCompacted}`,
        `    Cap (tokens):      ${process.env.REASONIX_RESULT_CAP_TOKENS ?? "3000 (default)"}`,
        `    Scavenge:          ${SCAVENGE_ENABLED ? "on" : "off (REASONIX_SCAVENGE=1 to enable)"}`,
        "",
        `  🔄 Turns:  ${stats.totalTurns}`,
        `  📦 Tokens: ~${(stats.totalTokens / 1000).toFixed(1)}K total`,
      ];
      _ctx.ui?.notify?.(lines.join("\n"), "info");
    },
  });
}
