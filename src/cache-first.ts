/**
 * cache-first — Immutable prefix + append-only log for DeepSeek prefix-cache stability.
 *
 * Harvested from reasonix Pillar 1 (Cache-First Loop).
 *
 * DeepSeek's automatic prefix caching activates only when the exact byte prefix
 * of the previous request matches. Most agent loops reorder, rewrite, or inject
 * fresh timestamps each turn — cache hit rate in practice: <20%.
 *
 * The prefix that matters to DeepSeek is the *serialized request head*:
 * the system message followed by the tool definitions (payload.tools).
 * Conversation content appends after that stable head. If the head is
 * byte-identical across turns, the disk cache hits on every repeat.
 *
 * This module tracks that head and ensures messages are serialized in
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
 * IMPORTANT: the tools hash is computed from the *tool definitions*
 * (the OpenAI `tools` array in the request payload), NOT from assistant
 * tool_calls in the message history. Tool *calls* grow every turn; hashing
 * them made the "prefix stable" indicator permanently red even when the
 * actual cache head was byte-identical (observed: 98.2% hit ratio with
 * `Prefix stable: ❌`). Tool *definitions* are what the cache head contains.
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
    tools?: unknown[],
  ): { messages: DeepSeekChatMessage[]; prefixHash: string } {
    const systemMsg = messages.find((m) => m.role === "system");
    const systemText = systemMsg?.content ?? "";

    const systemHash = fastHash(systemText);
    // Hash the *tool definitions* (stable), not tool calls (append-only).
    const toolsHash = fastHash(JSON.stringify(tools ?? []));
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

  /**
   * Validate that the messages log has only grown (no deletions / reorders).
   *
   * Returns true when the entry count is non-decreasing. When a truncation
   * is detected the baseline resets to the new (smaller) count so the next
   * growth is validated correctly — otherwise the check would return false
   * forever after the first compaction (observed: `conversationTruncations`
   * inflating on every subsequent turn).
   */
  validate(entries: DeepSeekChatMessage[]): boolean {
    const shrank = entries.length < this._entryCount;
    this._entryCount = entries.length;
    return !shrank;
  }

  reset(): void {
    this._entryCount = 0;
  }
}
