/**
 * Tests for pi-reasonix core modules.
 *
 * Run: node --test test/*.test.mjs
 *
 * Tests import from the compiled JS (dist/src/) since Node ESM
 * cannot resolve .ts files directly without a loader.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

/* ------------------------------------------------------------------ */
/*  cache-first tests                                                   */
/* ------------------------------------------------------------------ */

describe("PrefixGuard", () => {
  it("stabilises messages placing system first", async () => {
    const { PrefixGuard } = await import("../dist/src/cache-first.js");
    const guard = new PrefixGuard();

    const result = guard.stabilise([
      { role: "user", content: "hello" },
      { role: "system", content: "you are a bot" },
    ]);

    assert.equal(result.messages[0].role, "system");
    assert.equal(result.messages[1].role, "user");
    assert(result.prefixHash.length > 0);
  });

  it("produces stable hash for same prefix", async () => {
    const { PrefixGuard } = await import("../dist/src/cache-first.js");
    const guard = new PrefixGuard();

    const r1 = guard.stabilise([
      { role: "system", content: "you are a bot" },
      { role: "user", content: "hi" },
    ]);
    const r2 = guard.stabilise([
      { role: "system", content: "you are a bot" },
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello!" },
    ]);

    assert.equal(r1.prefixHash, r2.prefixHash);
  });

  it("detects changed prefix hash", async () => {
    const { PrefixGuard } = await import("../dist/src/cache-first.js");
    const guard = new PrefixGuard();

    const r1 = guard.stabilise([
      { role: "system", content: "prefix a" },
    ]);
    guard.reset();
    const r2 = guard.stabilise([
      { role: "system", content: "prefix b" },
    ]);

    assert.notEqual(r1.prefixHash, r2.prefixHash);
  });
});

describe("fastHash", () => {
  it("produces consistent results", async () => {
    const { fastHash } = await import("../dist/src/cache-first.js");
    assert.equal(fastHash("hello"), fastHash("hello"));
    assert.notEqual(fastHash("hello"), fastHash("world"));
  });
});

/* ------------------------------------------------------------------ */
/*  repair tests                                                        */
/* ------------------------------------------------------------------ */

describe("repairTruncatedJSON", () => {
  it("leaves valid JSON unchanged", async () => {
    const { repairTruncatedJSON } = await import("../dist/src/repair.js");
    const result = repairTruncatedJSON('{"a": 1, "b": 2}');
    assert.equal(result.fixed, false);
    assert.equal(result.repaired, '{"a": 1, "b": 2}');
  });

  it("repairs unterminated JSON", async () => {
    const { repairTruncatedJSON } = await import("../dist/src/repair.js");
    const result = repairTruncatedJSON('{"a": 1, "b": {"c": 2}');
    assert.equal(result.fixed, true);
    const parsed = JSON.parse(result.repaired);
    assert.deepEqual(parsed, { a: 1, b: { c: 2 } });
  });

  it("repairs unterminated array in JSON", async () => {
    const { repairTruncatedJSON } = await import("../dist/src/repair.js");
    const result = repairTruncatedJSON('{"items": [1, 2, 3');
    assert.equal(result.fixed, true);
    const parsed = JSON.parse(result.repaired);
    assert.deepEqual(parsed.items, [1, 2, 3]);
  });
});

describe("scavengeToolCalls", () => {
  it("extracts tool calls from think blocks", async () => {
    const { scavengeToolCalls } = await import("../dist/src/repair.js");
    const result = scavengeToolCalls(
      '<think>I need to search for the file. {"function": {"name": "search", "arguments": {"query": "test"}}}</think>',
    );
    assert.equal(result.length, 1);
    assert.equal(result[0].function.name, "search");
  });
});

describe("detectCallStorm", () => {
  it("suppresses duplicate tool calls", async () => {
    const { detectCallStorm } = await import("../dist/src/repair.js");
    const result = detectCallStorm([
      { id: "1", function: { name: "read", arguments: '{"path":"a"}' } },
      { id: "2", function: { name: "read", arguments: '{"path":"a"}' } },
      { id: "3", function: { name: "read", arguments: '{"path":"b"}' } },
      { id: "4", function: { name: "read", arguments: '{"path":"a"}' } },
    ]);

    // calls[1] duplicates calls[0] (both path=a), calls[3] also duplicates calls[0]
    // calls[2] has path=b which is different — kept
    assert.equal(result.stormCount, 2);
    assert.equal(result.clean.length, 2);
  });
});

describe("isTruncatedJSON", () => {
  it("detects complete JSON", async () => {
    const { isTruncatedJSON } = await import("../dist/src/repair.js");
    assert.equal(isTruncatedJSON('{"a": 1}'), false);
  });

  it("detects truncated JSON", async () => {
    const { isTruncatedJSON } = await import("../dist/src/repair.js");
    assert.equal(isTruncatedJSON('{"a": 1, "b": {"c"'), true);
  });
});

/* ------------------------------------------------------------------ */
/*  cost-control tests                                                  */
/* ------------------------------------------------------------------ */

describe("compactToolResult", () => {
  it("truncates large tool results", async () => {
    const { compactToolResult } = await import("../dist/src/cost-control.js");
    const longContent = "x".repeat(50000);
    const result = compactToolResult(
      { role: "tool", content: longContent, tool_call_id: "1" },
      3000,
    );

    assert(result.content && result.content.length < longContent.length);
    assert(result.content && result.content.includes("[content truncated:"));
  });

  it("skips small tool results", async () => {
    const { compactToolResult } = await import("../dist/src/cost-control.js");
    const result = compactToolResult(
      { role: "tool", content: "short", tool_call_id: "1" },
      3000,
    );
    assert.equal(result.content, "short");
  });
});

describe("estimateTokens", () => {
  it("estimates tokens from string length", async () => {
    const { estimateTokens } = await import("../dist/src/cost-control.js");
    assert.equal(estimateTokens("hello world"), 3);
    assert.equal(estimateTokens(""), 0);
  });
});
