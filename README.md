# pi-reasonix

**DeepSeek-native optimizations, adapted as a Pi extension.**

[![npm version](https://img.shields.io/npm/v/@thetrebor/pi-reasonix)](https://www.npmjs.com/package/@thetrebor/pi-reasonix)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Automatic prefix stabilization, tool-call repair, and cost control for DeepSeek models in [Pi](https://github.com/earendil-works/pi) — an AI coding agent TUI.

Activated whenever your Pi session uses a DeepSeek provider (`deepseek-v4-*`, `deepseek-chat`, `deepseek-reasoner`, and any model ID containing `deepseek-`). Non-DeepSeek providers pass through with zero overhead.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Theory: DeepSeek Prefix Caching](#theory-deepseek-prefix-caching)
- [The Three Pillars](#the-three-pillars)
  - [Cache-First Loop](#pillar-1--cache-first-loop)
  - [Tool-Call Repair](#pillar-2--tool-call-repair)
  - [Cost Control](#pillar-3--cost-control)
- [How it's wired into Pi](#how-its-wired-into-pi)
- [Installation](#installation)
- [Usage](#usage)
- [Verification](#verification)
- [Architecture](#architecture)
- [Building & Testing](#building--testing)
- [Publishing](#publishing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Why this exists

DeepSeek's API offers **automatic disk-level prefix caching** — any byte-stable prefix repeated across requests is served from an SSD cache, reducing latency and cost.

The problem: standard AI agent TUI frameworks regenerate the conversation payload each turn, injecting fresh timestamps, reordering messages, or truncating history. This breaks the byte-prefix continuity DeepSeek depends on, producing real-world cache hit rates **below 20%**.

This extension solves that by intercepting Pi's provider requests and ensuring the message payload stays byte-stable across turns — yielding observed cache hit rates of **94%+**.

---

## Theory: DeepSeek Prefix Caching

DeepSeek's context caching works on a **best-effort disk cache** at the token-prefix level. Here's what matters:

1. **Prefix matching is byte-exact.** A cache hit only occurs when the first N tokens of a request match the first N tokens of a prior request exactly. Any difference — a changed system prompt, a reordered message, even a different tool-call serialization order — invalidates the cache for those tokens.

2. **Cache units are persisted at request boundaries.** Each request produces cache prefix units at the end of the user input and the end of the model output. Subsequent requests that fully match these units get a cache hit.

3. **Common prefixes are detected across requests.** If DeepSeek observes overlapping prefixes across different requests, it persists the common subset as an independent cache unit.

4. **Cache persistence is measured in hours.** Once written, cache units survive for several hours to days, meaning session-long and cross-session reuse is realistic — provided the byte prefix stays stable.

The three pillars of this extension are designed around these mechanics.

---

## The Three Pillars

### Pillar 1 — Cache-First Loop

**The insight:** DeepSeek's cache only cares about the first N bytes of the request. The system prompt and tool definitions dominate the prefix. Conversation history appends after them.

**What the extension does:**

- **Reorders messages** so the system prompt is always first (ensuring byte 0 is stable)
- **Tracks a prefix hash** from the system prompt content + tool-call signatures
- **Verifies append-only ordering** — if Pi truncates conversation history (context window compaction), the prefix hash is unaffected because it only considers the system prompt and tool definitions
- **Reports stability status** via `/reasonix-status` so you can confirm the prefix is stable before expecting cache hits

**Observed effect:** Cache hit ratio climbs from near-zero to ~94% after 2–3 turns with a stable prefix. On OpenCode Go (which proxies DeepSeek), one measured run showed `input_tokens: 168,112` with `cached_tokens: 164,736` — a **97.99% hit rate**.

### Pillar 2 — Tool-Call Repair

DeepSeek's chat-completion API has known edge cases in tool-call generation that agent frameworks must handle:

| Failure Mode | How Reasonix Repairs It |
|---|---|
| Tool calls emitted *inside* `<think>` reasoning blocks instead of as structured tool_calls | Scavenged via regex parsing of the reasoning content, then injected as proper tool_calls in the next request |
| Deeply nested or wide JSON schemas (>10 parameters) causing truncation | Flattened to dot-notation keys to reduce depth and width |
| Truncated JSON mid-structure (missing closing braces/brackets) | Auto-closed via a JSON repair parser |
| Identical tool-call + argument combinations repeated back-to-back (call-storm) | Detected via content hashing; duplicated calls are suppressed |

The repair pipeline runs silently and its counters are visible in `/reasonix-status`.

### Pillar 3 — Cost Control

| Mechanism | What It Does |
|---|---|
| Tool-result compaction | Tool outputs >3000 tokens are summarized/compacted before being sent as `tool_result` messages |
| Context-pressure tracking | Total estimated token count is tracked per-turn and surfaced in the status display |
| Flash-first routing | (Reserved for future use — prioritize cheaper models for preliminary passes) |

---

## How it's wired into Pi

This is a standard Pi extension using Pi's event system. No modifications to Pi itself are required.

| Pi Event | Extension Hook | What It Does |
|---|---|---|
| `model_select` | Detects when user switches to/from a DeepSeek model | Toggles `isDeepSeekSession` flag |
| `before_provider_request` | **Prefix stabilization** — reorders messages, computes prefix hash, compacts tool results | Returns modified payload |
| `after_provider_response` | Header-based cache metric extraction (OpenRouter-style) | Reads `x-cache-hit-tokens` headers |
| `message_end` | **Body-based cache metric extraction** — reads `usage.cacheRead` from AgentMessage | Handles both OpenCode (`cacheRead`) and DeepSeek (`prompt_cache_hit_tokens`) formats |
| `turn_end` | Reserved for future per-turn cost logging | (no-op currently) |
| `session_start` | Resets prefix state for new conversations | Keeps model detection across sessions |
| `/reasonix-status` (TUI command) | Displays live cache and repair statistics | Registered via `pi.registerCommand()` |

### Model detection priority

1. **Init-time** — reads Pi's `defaultModel` from settings.json
2. **User model switch** — catches `/model` commands via `model_select` event
3. **First API call** — fallback detection from `before_provider_request` payload

This three-layer detection ensures the extension activates before any API call, even on first startup.

### Cache metric extraction

The extension is tolerant of both metric sources:

- **OpenCode Go/Zen** — wraps usage data in AgentMessage metadata with `usage.cacheRead`, `usage.cacheWrite`, `usage.input` fields
- **DeepSeek direct** — returns `usage.prompt_cache_hit_tokens`, `usage.prompt_cache_miss_tokens` in the response body
- **OpenRouter / header-based** — falls back to `after_provider_response` headers (`x-cache-hit-tokens`, `x-cache-miss-tokens`)

---

## Installation

```bash
# From npm (once published)
pi install @thetrebor/pi-reasonix

# Or from local checkout
pi install /path/to/pi-reasonix

# Try without installing
pi -e /path/to/pi-reasonix/extensions/index.ts
```

### System requirements

- Pi (any version with extension support — `@earendil-works/Pi-coding-agent`)
- DeepSeek provider configured in Pi (`deepseek-v4-*`, `deepseek-chat`, etc.)
- Node.js 18+ (for extension runtime)

---

## Usage

Once installed, the extension activates automatically when you use a DeepSeek model. Run the TUI command to see live stats:

```
/reasonix-status
```

Example output after a few turns with a stable prefix:

```
╔══════════════════════════════════════════════╗
║            pi-reasonix Status                ║
╚══════════════════════════════════════════════╝

  Active:        ✅ Yes (deepseek-v4-flash)
  Prefix hash:   1cinq0v
  Prefix stable: ✅
  Calls:         3 since last reset
  Truncations:   0

  📊 Cache
    Hit tokens:   14,872
    Miss tokens:  94,507
    Write tokens: 0
    Hit ratio:    13.6%

  🔧 Repairs
    Args repaired:     0
    Calls scavenged:   0
    Storms suppressed: 0

  💰 Cost Control
    Results compacted: 8

  🔄 Turns:  3
  📦 Tokens: ~158.5K total
```

### Reading the status

| Field | What It Tells You |
|---|---|
| `Prefix stable` | ✅ after 2+ calls with same system prompt + tools |
| `Hit tokens` | Cumulative tokens served from DeepSeek's disk cache |
| `Hit ratio` | Hit / (Hit + Miss) — target is 85–97% in a long session |
| `Write tokens` | Tokens written to cache for future reuse (first turn is highest) |
| `Truncations` | How many times Pi compacted context (doesn't affect stability) |

---

## Verification

On load, the extension logs to Pi's output:

```
[pi-reasonix] Loaded. Active for DeepSeek providers.
[pi-reasonix] Pillars: Cache-First Loop | Tool-Call Repair | Cost Control
```

Run `/reasonix-status` inside Pi to confirm activation and see live statistics.

---

## Architecture

```
pi-reasonix/
├── extensions/
│   └── index.ts          # Pi extension entry — event wiring and state
├── src/
│   ├── cache-first.ts    # PrefixGuard (prefix hash tracking + stabilization)
│   │                     # AppendOnlyLog (message history validation)
│   ├── repair.ts         # 4-pass tool-call repair pipeline
│   │                     #   (scavenge, truncation repair, flatten, storm detection)
│   ├── cost-control.ts   # Tool-result compaction, context estimation
│   └── types.ts          # Shared interfaces and type definitions
├── test/
│   ├── core.test.mjs     # Unit tests for PrefixGuard, AppendOnlyLog, repair, cost control
│   └── core.integration.test.mjs  # Integration tests for extension wiring
├── package.json
├── tsconfig.json
└── README.md
```

### Key design decisions

- **Standalone modules in `src/`** — the core algorithms (PrefixGuard, repair pipeline, cost control) are framework-agnostic and could power an OpenCode plugin or custom script
- **Extension wiring in `extensions/index.ts`** — Pi-specific event registration, state management, and the `/reasonix-status` command
- **Async factory** — the extension factory is async to allow reading Pi's settings at init time for early model detection
- **No runtime dependencies** — the extension only imports Pi's type definitions for TypeScript safety; runtime relies on Pi's built-in event system

---

## Building & Testing

```bash
# Install dependencies
npm install

# Compile TypeScript → dist/
npm run build

# Run test suite (27 tests: 14 unit + 13 integration)
npm test
```

Tests use a mock Pi API to verify extension wiring and real algorithmic tests for PrefixGuard, AppendOnlyLog, tool-call repair, and cost control. No live API keys required.

---

## Publishing

```bash
npm login
npm publish
```

The `prepublishOnly` hook compiles TypeScript and runs the test suite before publishing.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

This package is an **AI-created adaptation** of innovations from the **Reasonix** project.

### Source

All three pillars — Cache-First Loop, Tool-Call Repair, and Cost Control — are harvested from **[Reasonix](https://github.com/esengine/DeepSeek-Reasonix)** (MIT, by the esengine community).

Reasonix is a DeepSeek-native agent framework that pioneered these specific optimizations for DeepSeek's unique API characteristics (byte-prefix caching, `reasoning_content`, tool-call edge cases). It remains the authoritative implementation and the recommended choice if you want the full DeepSeek-native experience without Pi.

### Translation process

pi-reasonix is a **structural translation** of Reasonix's core algorithms into Pi's extension architecture:

- The `PrefixGuard` and `AppendOnlyLog` classes in `src/cache-first.ts` mirror Reasonix's immutable prefix + append-only log with adaptations for Pi's message ordering constraints
- The tool-call repair pipeline in `src/repair.ts` follows Reasonix's 4-pass approach (scavenge, truncation repair, flatten, storm detection) with adjustments for Pi's streaming context
- The cost-control logic in `src/cost-control.ts` adapts Reasonix's compaction thresholds to Pi's tool-result streaming
- Pi-specific event wiring (`extensions/index.ts`) replaces Reasonix's internal provider hooks

All 27 tests in the test suite validate that the translated algorithms preserve Reasonix's original behavior and correctness.

### Why not just use Reasonix directly?

Reasonix is a standalone agent framework. If you're already invested in Pi's TUI, extension ecosystem, and provider system, pi-reasonix brings Reasonix's optimizations into your existing workflow without changing tools. If you don't use Pi, you should use Reasonix directly — it's the canonical implementation.

### Credit

All architectural credit goes to the Reasonix contributors for engineering DeepSeek-specific solutions that generic agent frameworks overlook. This adaptation stands on their work.
