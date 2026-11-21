/**
 * pi-reasonix — shared types
 *
 * Harvested from reasonix: esengine/DeepSeek-Reasonix
 */

/** One block of an array-shaped message `content`. */
export interface MessageContentBlock {
  type?: string;
  text?: string;
  [key: string]: unknown;
}

/**
 * A message `content` in a provider payload is not always a plain string.
 *
 * OpenAI-shaped requests start with `content: "…"`, but the value becomes an
 * array of content blocks as soon as anything adds `cache_control` markers
 * (pi's own `applyAnthropicCacheControl` when a model sets
 * `compat.cacheControlFormat: "anthropic"`, or a caching extension such as
 * pi-opencode-go-cache), or when the message carries images.
 */
export type MessageContent =
  | string
  | null
  | undefined
  | ReadonlyArray<MessageContentBlock | null | undefined>;

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
  /**
   * Plain string in OpenAI-shaped payloads, an array of content blocks once
   * `cache_control` markers or images are involved. Never assume string.
   */
  content: MessageContent;
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
  /** Tokens written to disk cache for future requests. */
  cacheWriteTokens: number;
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
  /** Hook invocations that threw and were swallowed (payload shape surprises). */
  hookErrors: number;
}
