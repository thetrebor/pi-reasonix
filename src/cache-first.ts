/**
 * cache-first — Immutable prefix + append-only log for DeepSeek prefix-cache stability.
 *
 * Harvested from reasonix Pillar 1 (Cache-First Loop).
 *
 * DeepSeek's automatic prefix caching activates only when the exact byte prefix
 * of the previous request matches. Most agent loops reorder, rewrite, or inject
 * fresh timestamps each turn — cache hit rate in practice: <20%.
 *
 * This module tracks the system prefix and ensures messages are serialized in
 * append-only order so the prefix stays byte-stable across turns.
 */

import type { DeepSeekChatMessage, PrefixHash } from "./types.js";

/**
 * Compute a stable hash for a value using a fast non-crypto algorithm.
 * DeepSeek's cache is byte-prefix based, so we just need a deterministic
 * fingerprint to detect changes.
 */
export function fastHash(value: string): string {
  let h = 0;
  for (let i = 0; i < value.length; i++) {
    h = ((h << 5) - h + value.charCodeAt(i)) | 0;
  }
  return (h >>> 0).toString(36);
}

/**
 * Detect whether a provider URL targets DeepSeek.
 */
export function isDeepSeekProvider(baseUrl: string): boolean {
  const url = baseUrl.toLowerCase();
  return (
    url.includes("deepseek.com") ||
    url.includes("deepseek") ||
    url.includes("api.deepseek")
  );
}

/**
 * Immutable prefix tracker.
 *
 * Tracks the prefix derived from the system prompt + tool definitions.
 * These are what determine DeepSeek's prefix-cache matching — the rest of
 * the conversation history just appends after the stable prefix.
 *
 * Message truncation (context window compaction) does NOT affect prefix
 * stability, because the prefix hash only considers:
 *   1. The system message content
 *   2. The tool-call definitions (assistant messages with tool_calls)
 *
 * If neither changes across turns, the prefix is "stable" and DeepSeek's
 * automatic disk cache will hit on every repeat of the same prefix bytes.
 */
export class PrefixGuard {
  private _systemHash = "";
  private _toolsHash = "";
  private _prevPrefixHash = "";
  private _prefixHash = "";
  private _stabiliseCount = 0;

  /** Stabilise messages array: system first, stable prefix hash. */
  stabilise(
    messages: DeepSeekChatMessage[],
  ): { messages: DeepSeekChatMessage[]; prefixHash: string } {
    const systemMsg = messages.find((m) => m.role === "system");
    const systemText = systemMsg?.content ?? "";

    const systemHash = fastHash(systemText);
    const toolCallsJSON = JSON.stringify(
      messages.filter((m) => m.role === "assistant" && m.tool_calls),
    );
    const toolsHash = fastHash(toolCallsJSON);
    const prefixHash = fastHash(systemHash + "|" + toolsHash);

    // Always re-emit system first if it exists.
    const stabilised: DeepSeekChatMessage[] = [];
    if (systemMsg) {
      stabilised.push(systemMsg);
    }
    // Append non-system messages in stable order (no reordering).
    const nonSystem = messages.filter((m) => m.role !== "system");
    stabilised.push(...nonSystem);

    this._stabiliseCount++;
    this._prevPrefixHash = this._prefixHash;
    this._prefixHash = prefixHash;
    this._systemHash = systemHash;
    this._toolsHash = toolsHash;

    return { messages: stabilised, prefixHash };
  }

  /** Current prefix hash for cache-diagnostics headers. */
  get prefixHash(): string {
    return this._prefixHash;
  }

  /**
   * True when the prefix hash is stable across at least 2 successive calls.
   * First call always returns false (no baseline for comparison).
   * Seed calls after reset return false until a second comparison.
   */
  isStable(): boolean {
    return (
      this._stabiliseCount >= 2 &&
      this._prefixHash !== "" &&
      this._prefixHash === this._prevPrefixHash
    );
  }

  /** Whether the guard has been initialized (computed at least once). */
  isInitialized(): boolean {
    return this._stabiliseCount > 0;
  }

  /** Times stabilise() has been called since last reset. */
  get callCount(): number {
    return this._stabiliseCount;
  }

  /** Reset (new session or context cleared). */
  reset(): void {
    this._systemHash = "";
    this._toolsHash = "";
    this._prevPrefixHash = "";
    this._prefixHash = "";
    this._stabiliseCount = 0;
  }
}

/**
 * Append-only log tracker.
 *
 * Ensures that conversation history is only ever appended, never mutated or
 * reordered. This preserves the prefix for subsequent turns.
 */
export class AppendOnlyLog {
  private _entryCount = 0;

  /** Validate that the messages log has only grown (no deletions / reorders). */
  validate(entries: DeepSeekChatMessage[]): boolean {
    // We can't perfectly detect reordering without keeping a full copy,
    // but we can detect truncation (entries removed from the end).
    if (entries.length < this._entryCount) {
      return false; // Entries were removed
    }
    this._entryCount = entries.length;
    return true;
  }

  reset(): void {
    this._entryCount = 0;
  }
}
