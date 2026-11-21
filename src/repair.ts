/**
 * repair — Tool-call repair pipeline for DeepSeek.
 *
 * Harvested from reasonix Pillar 2 (Tool-Call Repair).
 *
 * DeepSeek has known failure modes:
 * 1. Tool-call JSON emitted inside `<think>` blocks, missing from tool_calls
 * 2. Arguments dropped when schema has >10 params or deeply nested objects
 * 3. Same tool called repeatedly with identical args (call-storm)
 * 4. Truncated JSON due to max_tokens hit mid-structure
 *
 * This module provides four passes to address each one.
 */

import type { ToolCallRepairInput } from "./types.js";

/* ------------------------------------------------------------------ */
/*  Pass 1: flatten — simplify deep / wide parameter schemas           */
/* ------------------------------------------------------------------ */

interface ParamNode {
  name: string;
  type: string;
  depth: number;
  isRequired?: boolean;
  description?: string;
}

/** Detect whether a schema would benefit from flattening. */
export function needsFlattening(
  paramNames: string[],
  depth: number,
): boolean {
  return paramNames.length > 10 || depth > 2;
}

/**
 * Flatten a deep parameter name by joining with dots.
 * E.g. `style.color` instead of `style: { color: "red" }`.
 */
export function flattenParamName(path: string[]): string {
  return path.join(".");
}

/** Closing token for each container opener. */
const OPEN_TO_CLOSE = { "{": "}", "[": "]" } as const;

/**
 * State of a (possibly truncated) JSON document, from a real character scan.
 *
 * Raw brace *counting* is not enough: `{"a": "x}` has one `{` and one `}`, so a
 * counter-based check calls it balanced even though the document ends inside a
 * string literal.
 */
interface JsonScan {
  /** The document ends inside a string literal. */
  inString: boolean;
  /** The trailing character of the string is an unpaired backslash. */
  danglingEscape: boolean;
  /** Open containers, outermost first. */
  stack: Array<"{" | "[">;
}

function scanJSON(text: string): JsonScan {
  const stack: Array<"{" | "["> = [];
  let inString = false;
  let escaped = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (ch === "\\") {
        escaped = true;
      } else if (ch === '"') {
        inString = false;
      }
      continue;
    }
    if (ch === '"') {
      inString = true;
      continue;
    }
    if (ch === "{" || ch === "[") {
      stack.push(ch);
    } else if (ch === "}" || ch === "]") {
      stack.pop();
    }
  }

  return { inString, danglingEscape: inString && escaped, stack };
}

/**
 * Estimate whether tool-call arguments JSON is truncated.
 *
 * True when the document does not parse AND either a string literal or a
 * container is still open at the end.
 */
export function isTruncatedJSON(text: string): boolean {
  if (!text) return false;
  try {
    JSON.parse(text);
    return false; // Parsed fine
  } catch {
    const scan = scanJSON(text);
    return scan.inString || scan.stack.length > 0;
  }
}

/** Remove a string-literal key at the end of `text` (`{"a"` -> `{`). */
function dropTrailingKey(text: string): string {
  if (!text.endsWith('"')) return text;

  // Walk back to the opening quote of the trailing literal.
  let i = text.length - 2;
  while (i >= 0) {
    if (text[i] === '"') {
      let backslashes = 0;
      let j = i - 1;
      while (j >= 0 && text[j] === "\\") {
        backslashes++;
        j--;
      }
      if (backslashes % 2 === 0) break; // Not escaped: this is the opener.
    }
    i--;
  }
  if (i < 0) return text;

  return text.slice(0, i).trimEnd();
}

/**
 * Drop an incomplete trailing member: `{"a": 1,` -> `{"a": 1`,
 * `{"a":` -> `{}`, `{"a": 1, "b":` -> `{"a": 1}`.
 *
 * Without this, appending closers turns `{"a":` into `{"a":}`, which is still
 * invalid JSON, so the whole repair used to be a no-op.
 */
function dropDanglingMember(text: string): string {
  let out = text.trimEnd();

  // Bounded loop: every branch strictly shortens `out`.
  for (let guard = 0; guard < 8; guard++) {
    if (out.endsWith(",")) {
      out = out.slice(0, -1).trimEnd();
      continue;
    }
    if (out.endsWith(":")) {
      out = dropTrailingKey(out.slice(0, -1).trimEnd());
      continue;
    }
    break;
  }

  return out;
}

/**
 * Attempt to repair truncated JSON by closing the open string and containers.
 *
 * Handles the DeepSeek max_tokens cut, which almost always lands *mid-string*
 * (`{"path": "/Users/me/repo/sr`) rather than neatly between tokens.
 */
export function repairTruncatedJSON(text: string): {
  repaired: string;
  fixed: boolean;
} {
  if (!isTruncatedJSON(text)) {
    return { repaired: text, fixed: false };
  }

  let result = text;

  // 1. Close an unterminated string literal. Appending closers first would
  //    place them inside the string and leave the document invalid.
  let scan = scanJSON(result);
  if (scan.inString) {
    result += scan.danglingEscape ? '\\"' : '"';
    scan = scanJSON(result);
  }

  // 2. Drop an incomplete trailing member, which would otherwise become
  //    `{"a": }` once the closers are appended.
  result = dropDanglingMember(result);

  // 3. Close the containers that are still open, innermost first, using the
  //    closer that matches each opener. The previous version emitted every
  //    `]` before every `}`, so `{"a":[{"b":1` became `{"a":[{"b":1]}}`.
  scan = scanJSON(result);
  for (let i = scan.stack.length - 1; i >= 0; i--) {
    result += OPEN_TO_CLOSE[scan.stack[i]];
  }

  // 4. Trailing commas left behind by the cut.
  result = result.replace(/,\s*([}\]])/g, "$1");

  try {
    JSON.parse(result);
    return { repaired: result, fixed: true };
  } catch {
    return { repaired: text, fixed: false };
  }
}

/* ------------------------------------------------------------------ */
/*  Pass 2: scavenge — find tool calls in reasoning / text            */
/* ------------------------------------------------------------------ */

/**
 * Scavenge tool calls that DeepSeek "leaked" into reasoning_content or
 * message content (outside the tool_calls array).
 */
export function scavengeToolCalls(
  content: string | null | undefined,
): ToolCallRepairInput[] {
  if (!content) return [];

  const found: ToolCallRepairInput[] = [];

  // Pattern 1: tool_use blocks inside <think> tags
  // DeepSeek often emits tool call JSON inside reasoning blocks
  const thinkPattern = /<think>([\s\S]*?)<\/think>/g;
  let match: RegExpExecArray | null;

  while ((match = thinkPattern.exec(content)) !== null) {
    const thinkContent = match[1];
    const scavenged = extractToolCallsFromText(thinkContent);
    found.push(...scavenged);
  }

  // Pattern 2: tool calls embedded in markdown code fences
  const fencePattern = /```(?:json)?\s*\n?([\s\S]*?)```/g;
  while ((match = fencePattern.exec(content)) !== null) {
    const fenceContent = match[1];
    const scavenged = extractToolCallsFromText(fenceContent);
    found.push(...scavenged);
  }

  // Pattern 3: bare tool calls inline in the response text
  if (!thinkPattern.test(content)) {
    // Only if we didn't already find through thinks
    const inlineScavenged = extractToolCallsFromText(content);
    found.push(...inlineScavenged);
  }

  return found;
}

function extractToolCallsFromText(text: string): ToolCallRepairInput[] {
  const found: ToolCallRepairInput[] = [];

  // Look for JSON objects with function/name/arguments patterns
  const jsonPattern = /(?:```json\s*)?(\{[\s\S]*?"function"[\s\S]*?\})/g;
  let m: RegExpExecArray | null;
  while ((m = jsonPattern.exec(text)) !== null) {
    try {
      const obj = JSON.parse(m[1]);
      if (obj.function?.name && obj.function?.arguments) {
        found.push({
          id: obj.id || `scavenged-${found.length}`,
          type: "function",
          function: {
            name: obj.function.name,
            arguments:
              typeof obj.function.arguments === "string"
                ? obj.function.arguments
                : JSON.stringify(obj.function.arguments),
          },
        });
      }
    } catch {
      // Not valid JSON, skip
    }
  }

  // Also look for bare {"name": "...", "arguments": {...}} patterns
  const barePattern = /\{"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\}/g;
  while ((m = barePattern.exec(text)) !== null) {
    try {
      const args = JSON.parse(m[2]);
      found.push({
        id: `scavenged-bare-${found.length}`,
        type: "function",
        function: {
          name: m[1],
          arguments: JSON.stringify(args),
        },
      });
    } catch {
      // Skip
    }
  }

  return found;
}

/* ------------------------------------------------------------------ */
/*  Pass 3: storm — detect and break identical call repeats            */
/* ------------------------------------------------------------------ */

/**
 * Detect call-storms: same (tool, args) tuple within a sliding window.
 * Returns the number of storms detected (the excess calls to suppress).
 */
export function detectCallStorm(
  calls: ToolCallRepairInput[],
  windowSize = 5,
): { clean: ToolCallRepairInput[]; stormCount: number } {
  if (calls.length < 2) return { clean: calls, stormCount: 0 };

  const clean: ToolCallRepairInput[] = [calls[0]];
  let stormCount = 0;

  for (let i = 1; i < calls.length; i++) {
    const prev = calls.slice(Math.max(0, i - windowSize), i);
    const isDuplicate = prev.some(
      (p) =>
        p.function.name === calls[i].function.name &&
        p.function.arguments === calls[i].function.arguments,
    );

    if (isDuplicate) {
      stormCount++;
      // Suppress this call — don't add to clean
    } else {
      clean.push(calls[i]);
    }
  }

  return { clean, stormCount };
}

/* ------------------------------------------------------------------ */
/*  Pipeline                                                            */
/* ------------------------------------------------------------------ */

export interface RepairOptions {
  /** Max params before schema flattening triggers (default: 10). */
  maxParams?: number;
  /** Max nesting depth before flattening triggers (default: 2). */
  maxDepth?: number;
  /** Sliding window for call-storm detection (default: 5). */
  stormWindow?: number;
}

export interface RepairResult {
  repaired: ToolCallRepairInput[];
  scavenged: ToolCallRepairInput[];
  stormCount: number;
  truncatedFixed: number;
}

/**
 * Full repair pipeline for a set of tool calls.
 *
 * 1. Scavenge — recover tool calls leaked into reasoning_content
 * 2. Repair truncated JSON in arguments
 * 3. Detect and break call-storms
 *
 * Schema flattening is handled at tool registration time (not runtime).
 */
export function repairToolCalls(
  calls: ToolCallRepairInput[],
  reasoningContent?: string | null,
  options: RepairOptions = {},
): RepairResult {
  const { stormWindow = 5 } = options;

  const scavenged: ToolCallRepairInput[] = reasoningContent
    ? scavengeToolCalls(reasoningContent)
    : [];

  const allCalls = [...calls, ...scavenged];

  // Repair truncated JSON in arguments
  let truncatedFixed = 0;
  for (const call of allCalls) {
    const { repaired, fixed } = repairTruncatedJSON(call.function.arguments);
    if (fixed) {
      call.function.arguments = repaired;
      truncatedFixed++;
    }
  }

  // Detect call-storms
  const { clean, stormCount } = detectCallStorm(allCalls, stormWindow);

  return {
    repaired: clean,
    scavenged,
    stormCount,
    truncatedFixed,
  };
}
