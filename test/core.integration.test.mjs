/**
 * Integration tests for pi-reasonix extension.
 *
 * Tests the full extension factory: event wiring, payload transformation,
 * prefix stabilization, and the /reasonix-status command registration.
 *
 * Run: node --test test/*.integration.test.mjs
 *
 * These tests import from the compiled JS (dist/) to avoid Node ESM
 * limitations with import type / .ts resolution.
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { PrefixGuard, AppendOnlyLog, fastHash, isDeepSeekProvider } from "../dist/src/cache-first.js";
import { repairTruncatedJSON, scavengeToolCalls, repairToolCalls } from "../dist/src/repair.js";
import { compactToolResults, estimateContextUsage } from "../dist/src/cost-control.js";

/** Minimal shape for test messages. */
function msg(overrides) {
  return { role: "user", content: "", ...overrides };
}

/* ------------------------------------------------------------------ */
/*  Fake ExtensionAPI for testing extension wiring                     */
/* ------------------------------------------------------------------ */

function createMockAPI() {
  const captured = [];
  const handlers = new Map();

  return {
    api: {
      on: (event, handler) => {
        captured.push({ type: "on", event });
        handlers.set(event, handler);
      },
      registerCommand: (name, opts) => {
        captured.push({ type: "registerCommand", name, description: opts.description });
        handlers.set(`cmd:${name}`, opts.handler);
      },
      registerTool: () => {
        captured.push({ type: "registerTool" });
      },
      _handlers: handlers,
    },
    captured,
  };
}

/* ------------------------------------------------------------------ */
/*  Extension Factory Tests                                            */
/* ------------------------------------------------------------------ */

describe("Extension factory wiring", () => {
  it("registers all lifecycle hooks", async () => {
    const { api, captured } = createMockAPI();
    const ext = (await import("../dist/extensions/index.js")).default;
    ext(api);

    const events = captured
      .filter((c) => c.type === "on")
      .map((c) => c.event);

    assert(events.includes("before_provider_request"), "missing before_provider_request hook");
    assert(events.includes("after_provider_response"), "missing after_provider_response hook");
    assert(events.includes("turn_end"), "missing turn_end hook");
    assert(events.includes("session_start"), "missing session_start hook");
  });

  it("registers the /reasonix-status command", async () => {
    const { api, captured } = createMockAPI();
    const ext = (await import("../dist/extensions/index.js")).default;
    ext(api);

    const cmd = captured.find(
      (c) => c.type === "registerCommand" && c.name === "reasonix-status",
    );
    assert(cmd, "missing reasonix-status command");
    assert(cmd.description.includes("cache"), "description should mention cache");
  });

  it("handler signatures are compatible with ExtensionAPI types", async () => {
    const { api } = createMockAPI();
    const ext = (await import("../dist/extensions/index.js")).default;
    assert.doesNotThrow(() => ext(api));
  });
});

/* ------------------------------------------------------------------ */
/*  Payload Transformation Tests                                       */
/* ------------------------------------------------------------------ */

describe("before_provider_request payload transformation", () => {
  it("stabilises message order: system first", () => {
    const guard = new PrefixGuard();
    const messages = [
      msg({ role: "user", content: "hello" }),
      msg({ role: "system", content: "you are helpful" }),
      msg({ role: "user", content: "how are you?" }),
    ];

    const result = guard.stabilise(messages);
    assert.equal(result.messages[0].role, "system");
    assert.equal(result.messages[1].role, "user");
    assert.equal(result.messages[2].role, "user");
  });

  it("produces stable prefix hash across turns", () => {
    const guard = new PrefixGuard();

    const t1 = guard.stabilise([
      msg({ role: "system", content: "you are helpful" }),
      msg({ role: "user", content: "hi" }),
    ]);

    const t2 = guard.stabilise([
      msg({ role: "system", content: "you are helpful" }),
      msg({ role: "user", content: "hi" }),
      msg({ role: "assistant", content: "hello!" }),
      msg({ role: "user", content: "what's next?" }),
    ]);

    assert.equal(t1.prefixHash, t2.prefixHash);
  });

  it("changes prefix hash when system prompt changes", () => {
    const guard = new PrefixGuard();

    const t1 = guard.stabilise([
      msg({ role: "system", content: "old system prompt" }),
    ]);
    guard.reset();
    const t2 = guard.stabilise([
      msg({ role: "system", content: "new system prompt" }),
    ]);

    assert.notEqual(t1.prefixHash, t2.prefixHash);
  });

  it("compacts oversized tool results", () => {
    const guard = new PrefixGuard();
    const messages = [
      msg({ role: "system", content: "you are helpful" }),
      msg({ role: "user", content: "read the file" }),
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "1",
          function: { name: "read", arguments: '{"path":"big.txt"}' },
        }],
      },
      {
        role: "tool",
        content: "x".repeat(50000),
        tool_call_id: "1",
      },
    ];

    const stabilised = guard.stabilise(messages);
    const compacted = compactToolResults(stabilised.messages);

    const toolMsg = compacted.compacted.find((m) => m.role === "tool");
    assert(toolMsg);
    assert(toolMsg.content && toolMsg.content.length < 50000);
    assert(toolMsg.content.includes("[content truncated:"));
  });
});

describe("AppendOnlyLog validation", () => {
  it("rejects truncated message logs", () => {
    const log = new AppendOnlyLog();

    assert.equal(log.validate([msg({ role: "system" }), msg({ role: "user", content: "hi" })]), true);
    assert.equal(log.validate([msg({ role: "system" })]), false);
  });

  it("accepts appended message logs", () => {
    const log = new AppendOnlyLog();

    assert.equal(log.validate([msg({ role: "system" }), msg({ role: "user", content: "hi" })]), true);
    assert.equal(log.validate([
      msg({ role: "system" }),
      msg({ role: "user", content: "hi" }),
      msg({ role: "assistant", content: "hello" }),
    ]), true);
  });
});

/* ------------------------------------------------------------------ */
/*  Non-DeepSeek passthrough                                           */
/* ------------------------------------------------------------------ */

describe("DeepSeek model detection", () => {
  it("detects deepseek.com URLs", () => {
    assert.equal(isDeepSeekProvider("https://api.deepseek.com"), true);
    assert.equal(isDeepSeekProvider("https://api.deepseek.com/v1"), true);
    assert.equal(isDeepSeekProvider("https://api.openai.com/v1"), false);
    assert.equal(isDeepSeekProvider("http://localhost:11434"), false);
  });
});

/* ------------------------------------------------------------------ */
/*  End-to-end: repair pipeline                                        */
/* ------------------------------------------------------------------ */

describe("repairToolCalls integration", () => {
  it("scavenges tool calls from reasoning content and repairs truncated JSON", () => {
    const result = repairToolCalls(
      [
        { id: "1", function: { name: "read", arguments: '{"path": "a.txt"' } },
      ],
      '<think>{"function": {"name": "search", "arguments": {"q": "test"}}}</think>',
    );

    assert.equal(result.scavenged.length, 1);
    assert.equal(result.scavenged[0].function.name, "search");
    assert.equal(result.truncatedFixed, 1);
    const parsed = JSON.parse(result.repaired[0].function.arguments);
    assert.equal(parsed.path, "a.txt");
  });

  it("detects and suppresses call storms", () => {
    const result = repairToolCalls([
      { id: "1", function: { name: "read", arguments: '{"path":"x"}' } },
      { id: "2", function: { name: "read", arguments: '{"path":"x"}' } },
      { id: "3", function: { name: "read", arguments: '{"path":"x"}' } },
      { id: "4", function: { name: "read", arguments: '{"path":"y"}' } },
    ]);

    assert.equal(result.stormCount, 2);
    assert.equal(result.repaired.length, 2);
    assert.equal(result.repaired[0].id, "1");
    assert.equal(result.repaired[1].function.arguments, '{"path":"y"}');
  });
});

/* ------------------------------------------------------------------ */
/*  Context estimation                                                 */
/* ------------------------------------------------------------------ */

describe("estimateContextUsage", () => {
  it("estimates total tokens from messages", () => {
    const messages = [
      msg({ role: "system", content: "sys" }),
      msg({ role: "user", content: "hello world" }),
      {
        role: "assistant",
        content: null,
        tool_calls: [{
          id: "1",
          function: { name: "read", arguments: '{"path":"file.txt"}' },
        }],
      },
    ];

    const tokens = estimateContextUsage(messages);
    assert(typeof tokens === "number");
    assert(tokens > 0);
  });
});
