/**
 * cost-control — Turn-end result compaction and flash-first defaults.
 *
 * Harvested from reasonix Pillar 3 (Cost Control).
 *
 * DeepSeek v4-flash is ~12× cheaper than v4-pro, and tool results that
 * accumulate across turns inflate context (and thus cost) unnecessarily.
 *
 * This module provides:
 * 1. Turn-end auto-compaction of large tool results
 * 2. Context-window pressure detection
 */

import { fastHash } from "./cache-first.js";
import type { DeepSeekChatMessage } from "./types.js";

/** Default token cap per tool result after compaction. */
export const RESULT_CAP_TOKENS = 3000;

/** Emergency context pressure threshold (% of window). */
export const EMERGENCY_THRESHOLD = 0.8;

/** Proactive shrink threshold. */
export const PROACTIVE_THRESHOLD = 0.4;

/**
 * Rough estimate of token count from string length.
 * DeepSeek models use ~4 chars/token for English text.
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Compact a tool result message to a summary.
 * Truncates to the cap and appends a summary suffix.
 */
export function compactToolResult(
  message: DeepSeekChatMessage,
  capTokens = RESULT_CAP_TOKENS,
): DeepSeekChatMessage {
  if (message.role !== "tool") return message;
  if (!message.content) return message;

  const estimatedTokens = estimateTokens(message.content);
  if (estimatedTokens <= capTokens) return message;

  // Truncate to cap tokens (rough 4-char estimate)
  const capChars = capTokens * 4;
  const truncated = message.content.slice(0, capChars);

  return {
    ...message,
    content: `${truncated}\n\n[content truncated: ~${estimatedTokens} tokens → ~${capTokens} tokens]`,
  };
}

export interface CompactionResult {
  compacted: DeepSeekChatMessage[];
  compactedCount: number;
}

/**
 * Compact all tool messages in the list that exceed the token cap.
 */
export function compactToolResults(
  messages: DeepSeekChatMessage[],
  capTokens = RESULT_CAP_TOKENS,
): CompactionResult {
  let compactedCount = 0;
  const compacted = messages.map((msg) => {
    if (msg.role === "tool") {
      const compactedMsg = compactToolResult(msg, capTokens);
      if (compactedMsg !== msg) compactedCount++;
      return compactedMsg;
    }
    return msg;
  });
  return { compacted, compactedCount };
}

/**
 * Estimate total context usage from the messages array.
 */
export function estimateContextUsage(
  messages: DeepSeekChatMessage[],
): number {
  return messages.reduce((acc, msg) => {
    let tokens = 0;
    if (msg.content) tokens += estimateTokens(msg.content);
    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        tokens += estimateTokens(tc.function.name);
        tokens += estimateTokens(tc.function.arguments);
      }
    }
    return acc + tokens;
  }, 0);
}

/**
 * Check whether context is under pressure.
 * Returns a string indicating the pressure level or null if fine.
 */
export function checkContextPressure(
  currentTokens: number,
  contextWindow: number,
): "none" | "proactive" | "emergency" {
  const ratio = currentTokens / contextWindow;
  if (ratio >= EMERGENCY_THRESHOLD) return "emergency";
  if (ratio >= PROACTIVE_THRESHOLD) return "proactive";
  return "none";
}

/**
 * Summarize a tool result for roll-up during compaction.
 * Used when a tool result is too large to keep inline.
 */
export function summarizeToolResult(
  message: DeepSeekChatMessage,
): string {
  if (message.role !== "tool" || !message.content) return "";
  const estimatedTokens = estimateTokens(message.content);

  // Extract first meaningful line for a summary prefix
  const firstLine = message.content.split("\n").find(
    (l) => l.trim().length > 0,
  );
  const summary = firstLine
    ? firstLine.trim().slice(0, 120)
    : "(empty result)";

  return `[tool result: ~${estimatedTokens}tok | ${summary}]`;
}
