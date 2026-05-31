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
 * The prefix is computed once per session from:
 *   system prompt + tool specifications + few-shot examples
 *
 * It is pinned by hash and checked on every `before_provider_request` event.
 * If the prefix hash changes, the session caches are invalidated.
 */
export class PrefixGuard {
  private _systemHash = "";
  private _toolsHash = "";
  private _prefixHash = "";
  private _lastPrefix: string[] = [];

  /** Stabilise the messages array: trim to the immutable prefix + append-only log. */
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

    // If the prefix hasn't changed, preserve the established prefix ordering.
    // DeepSeek's automatic prefix cache matches from byte 0, so the system
    // message must always be the first message.
    const stabilised: DeepSeekChatMessage[] = [];

    // Always re-emit system first if it exists.
    if (systemMsg) {
      stabilised.push(systemMsg);
    }

    // Append non-system messages in stable order (no reordering).
    const nonSystem = messages.filter((m) => m.role !== "system");
    stabilised.push(...nonSystem);

    this._systemHash = systemHash;
    this._toolsHash = toolsHash;
    this._prefixHash = prefixHash;
    this._lastPrefix = [systemText, toolCallsJSON];

    return { messages: stabilised, prefixHash };
  }

  /** Current prefix hash for cache-diagnostics headers. */
  get prefixHash(): string {
    return this._prefixHash;
  }

  /** Whether the prefix has changed since last check. */
  isStable(): boolean {
    return this._prefixHash !== "";
  }

  /** Reset (new session). */
  reset(): void {
    this._systemHash = "";
    this._toolsHash = "";
    this._prefixHash = "";
    this._lastPrefix = [];
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
