/**
 * Tests for the session footer rendering module (src/footer.ts).
 *
 * Run: npm test  (builds dist/ first, then runs node --test)
 */

import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

describe("formatTokenCount", () => {
  it("abbreviates to M units", async () => {
    const { formatTokenCount } = await import("../dist/src/footer.js");
    assert.equal(formatTokenCount(0), "0M");
    assert.equal(formatTokenCount(1234), "0.001M");
    assert.equal(formatTokenCount(12_345), "0.01M");
    assert.equal(formatTokenCount(1_234_567), "1.23M");
    assert.equal(formatTokenCount(86_801_920), "86.8M");
  });
});

describe("formatCacheFooter", () => {
  it("shows placeholder before any request", async () => {
    const { formatCacheFooter } = await import("../dist/src/footer.js");
    assert.equal(formatCacheFooter(undefined), "⚡️ --");
  });

  it("renders a compact one-line summary without a tok suffix", async () => {
    const { formatCacheFooter } = await import("../dist/src/footer.js");
    const out = formatCacheFooter({
      totalRequests: 572,
      hitRequests: 560,
      cachedInputTokens: 86_801_920,
      totalInputTokens: 90_008_079,
    });
    assert.equal(out, "⚡️ 560/572 · 86.8M/90.0M (96%)");
  });

  it("appends non-zero write/repair counters", async () => {
    const { formatCacheFooter } = await import("../dist/src/footer.js");
    const out = formatCacheFooter({
      totalRequests: 572,
      hitRequests: 560,
      cachedInputTokens: 86_801_920,
      totalInputTokens: 90_008_079,
      cacheWriteInputTokens: 200_000,
      callsRepaired: 2,
      stormsSuppressed: 1,
      resultsCompacted: 3,
    });
    assert.equal(out, "⚡️ 560/572 · 86.8M/90.0M (96%) · w0.20M · 🔧2 · 🌀1 · ⚙3");
  });

  it("omits zero counters to keep a quiet session short", async () => {
    const { formatCacheFooter } = await import("../dist/src/footer.js");
    const out = formatCacheFooter({
      totalRequests: 572,
      hitRequests: 560,
      cachedInputTokens: 86_801_920,
      totalInputTokens: 90_008_079,
    });
    assert.equal(out, "⚡️ 560/572 · 86.8M/90.0M (96%)");
  });

  it("omits the percentage until input tokens exist", async () => {
    const { formatCacheFooter } = await import("../dist/src/footer.js");
    const out = formatCacheFooter({
      totalRequests: 1,
      hitRequests: 1,
      cachedInputTokens: 0,
      totalInputTokens: 0,
    });
    assert.equal(out, "⚡️ 1/1 · 0M/0M");
  });
});
