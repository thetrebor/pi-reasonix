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
 * Install:
 *   pi install /path/to/pi-reasonix
 *   pi install npm:@thetrebor/pi-reasonix
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

export default function (pi: ExtensionAPI) {
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
    totalTurns: 0,
    totalTokens: 0,
  };

  let isDeepSeekSession = false;
  let prefixHash = "";
  let currentModel = "";

  /* ------------------------------------------------------------------ */
  /*  model_select — detect DeepSeek model as soon as it's selected       */
  /* ------------------------------------------------------------------ */

  // model_select fires before model is used — lets us set isDeepSeekSession early.
  // TypeScript cast is needed because ModelSelectEvent isn't in exported types.
  (pi.on as (...args: unknown[]) => void)(
    "model_select",
    (event: Record<string, unknown>) => {
      // Debug: log the full event shape to understand the Model<any> object
      const modelObj = event?.model;
      console.log("[pi-reasonix] model_select event received");
      console.log("[pi-reasonix]   model type:", typeof modelObj);
      if (typeof modelObj === "object" && modelObj) {
        const obj = modelObj as Record<string, unknown>;
        console.log("[pi-reasonix]   model keys:", Object.keys(obj));
        console.log("[pi-reasonix]   model.id:", obj.id);
        console.log("[pi-reasonix]   model.name:", obj.name);
        console.log("[pi-reasonix]   source:", event.source);
      } else if (typeof modelObj === "string") {
        console.log("[pi-reasonix]   model string:", modelObj);
      }

      // ModelSelectEvent.model is a Model<any> object with .id or .name
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
        console.log("[pi-reasonix] ✅ DeepSeek detected via model_select:", modelId);
      } else if (modelId) {
        isDeepSeekSession = false;
        currentModel = modelId;
        console.log("[pi-reasonix] Non-DeepSeek model:", modelId);
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

      // Detect DeepSeek by model ID
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

      // 2. Check append-only invariant
      if (!logTracker.validate(stabilised.messages as DeepSeekChatMessage[])) {
        console.warn("[pi-reasonix] Message log was truncated. Prefix cache invalidated.");
        prefixGuard.reset();
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
  /*  turn_end — tracking (compaction happens in before_provider_request) */
  /* ------------------------------------------------------------------ */

  pi.on("turn_end", (_event: TurnEndEvent) => {
    // Result compaction is handled upstream in before_provider_request.
    // This hook is reserved for future use (e.g., per-turn cost logging).
  });

  /* ------------------------------------------------------------------ */
  /*  session_start — reset state for new sessions                       */
  /* ------------------------------------------------------------------ */

  pi.on("session_start", () => {
    // Don't reset isDeepSeekSession — model_select re-detects on model change.
    // Resetting here would wipe the detection that happened before the first turn.
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
        `  Prefix hash:   ${prefixHash || "(none)"}`,
        `  Prefix stable: ${prefixGuard.isStable() ? "✅" : "❌"}`,
        "",
        "  📊 Cache",
        `    Hit tokens:   ${stats.cacheHitTokens.toLocaleString()}`,
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

  /* ------------------------------------------------------------------ */
  /*  Notify on load                                                     */
  /* ------------------------------------------------------------------ */

  console.log("[pi-reasonix] Loaded. Active for DeepSeek providers.");
  console.log("[pi-reasonix] Pillars: Cache-First Loop | Tool-Call Repair | Cost Control");
}
