/**
 * footer — session-scoped cache stats + one-line TUI footer rendering.
 *
 * The footer renders the CURRENT session's cache usage (hit/total requests
 * and input tokens) plus repair/compaction counters, in the spirit of
 * pi-cache-optimizer's session mode. Counters reset on /reload or Pi restart
 * — no on-disk persistence.
 */

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

/** Session-scoped counters backing the one-line footer. */
export interface CacheFooterStats {
  totalRequests: number;
  hitRequests: number;
  /** Tokens served from the provider's disk cache (cache reads). */
  cachedInputTokens: number;
  totalInputTokens: number;
  /** Tokens written to the provider's disk cache for future requests. */
  cacheWriteInputTokens?: number;
  /** Repair-pipeline counters (only shown when non-zero). */
  callsRepaired?: number;
  stormsSuppressed?: number;
  resultsCompacted?: number;
}

/* ------------------------------------------------------------------ */
/*  Footer rendering                                                   */
/* ------------------------------------------------------------------ */

/**
 * Abbreviate a token count to a compact "M" string, matching the style of
 * pi-cache-optimizer's footer (0.002M / 1.23M / 12.3M).
 */
export function formatTokenCount(value: number): string {
  const millions = Math.max(0, Math.round(value)) / 1_000_000;
  if (millions === 0) return "0M";
  if (millions < 0.001) return `${millions.toFixed(4)}M`;
  if (millions < 0.01) return `${millions.toFixed(3)}M`;
  if (millions >= 10) return `${millions.toFixed(1)}M`;
  return `${millions.toFixed(2)}M`;
}

/**
 * One-line footer text, e.g.
 * `⚡️ 560/572 · 86.80M/90.01M (96%) · w0.20M · 🔧2 🌀1 ⚙3`.
 *
 * Optional counters are appended only when non-zero so a quiet session keeps
 * the line short. Returns `⚡️ --` before the first recorded request so the
 * status bar shows the banner is live (and reasonix is active) immediately.
 */
export function formatCacheFooter(stats: CacheFooterStats | undefined): string {
  if (!stats || stats.totalRequests === 0) return "⚡️ --";
  const percent =
    stats.totalInputTokens > 0
      ? ` (${Math.round((stats.cachedInputTokens / stats.totalInputTokens) * 100)}%)`
      : "";
  const extras: string[] = [];
  if ((stats.cacheWriteInputTokens ?? 0) > 0) {
    extras.push(`w${formatTokenCount(stats.cacheWriteInputTokens!)}`);
  }
  if ((stats.callsRepaired ?? 0) > 0) extras.push(`🔧${stats.callsRepaired}`);
  if ((stats.stormsSuppressed ?? 0) > 0) extras.push(`🌀${stats.stormsSuppressed}`);
  if ((stats.resultsCompacted ?? 0) > 0) extras.push(`⚙${stats.resultsCompacted}`);
  const tail = extras.length > 0 ? ` · ${extras.join(" · ")}` : "";
  return (
    `⚡️ ${stats.hitRequests}/${stats.totalRequests} · ` +
    `${formatTokenCount(stats.cachedInputTokens)}/${formatTokenCount(stats.totalInputTokens)}${percent}${tail}`
  );
}
