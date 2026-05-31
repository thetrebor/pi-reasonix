/**
 * pi-reasonix — shared types
 *
 * Harvested from reasonix: esengine/DeepSeek-Reasonix
 */

/** A serialisable tool call the model emitted. */
export interface ToolCallRepairInput {
  id: string;
  type?: string;
  function: {
    name: string;
    arguments: string;
  };
}

/** Result of the tool-call repair pipeline. */
export interface RepairReport {
  original: ToolCallRepairInput[];
  repaired: ToolCallRepairInput[];
  scavenged: ToolCallRepairInput[];
  flattened: boolean;
  truncated: boolean;
  storms: number;
}

/** DeepSeek-specific chat message with optional reasoning_content. */
export interface DeepSeekChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
  tool_calls?: ToolCallRepairInput[];
  tool_call_id?: string;
  reasoning_content?: string | null;
}

/** Hash of the immutable prefix for cache tracking. */
export interface PrefixHash {
  systemHash: string;
  toolsHash: string;
  prefixHash: string;
}

/** Stats exposed in the TUI / logs. */
export interface ReasonixStats {
  /** Cumulative cache-hit tokens from DeepSeek response headers. */
  cacheHitTokens: number;
  cacheMissTokens: number;
  /** Repair pipeline counters. */
  callsRepaired: number;
  callsScavenged: number;
  stormsSuppressed: number;
  /** Cost control. */
  resultsCompacted: number;
  /** Conversation truncation events (pi compacted history). */
  conversationTruncations: number;
  /** Session-level roll-ups. */
  totalTurns: number;
  totalTokens: number;
}
