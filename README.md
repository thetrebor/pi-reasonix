# pi-reasonix

**DeepSeek-native optimizations for Pi.**  
Harvested from [Reasonix](https://github.com/esengine/DeepSeek-Reasonix) — the leading DeepSeek-native agent framework.

## What it does

Three optimizations that activate automatically when your Pi session uses a DeepSeek provider:

### Pillar 1 — Cache-First Loop
DeepSeek's automatic prefix caching only activates when the **exact byte prefix** matches across turns. Standard agent loops rebuild the prompt each turn, hitting cache at <20%. This extension stabilizes the prefix — system prompt + tool specs stay byte-identical, conversation history is append-only. Expected cache hit rate: **~94%** (vs 47% baseline).

### Pillar 2 — Tool-Call Repair
DeepSeek has known failure modes that generic frameworks don't handle:

| Failure mode | Fix |
|---|---|
| Tool calls leaked inside `<think>` blocks | Scavenged via regex + JSON parser |
| Deeply nested / wide schemas (>10 params) | Flattened to dot notation |
| Truncated JSON mid-structure | Auto-closed braces/brackets |
| Identical tool+args repeats (call-storm) | Detected and suppressed |

### Pillar 3 — Cost Control
- Tool results >3000 tokens auto-compacted at turn end
- Context pressure detection (proactive at 40%, emergency at 80%)

## Status

```
/reasonix-status
```

Shows cache hit ratio, repair counters, context usage, and session stats.

## Install

```bash
# From npm (once published)
pi install npm:@thetrebor/pi-reasonix

# Or from local checkout
pi install /path/to/pi-reasonix
```

### Try without installing

```bash
pi -e /path/to/pi-reasonix/extensions/index.ts
```

### Manual placement

Copy or symlink into pi's auto-discovered extension directories:

```bash
# Global (all projects)
ln -s /path/to/pi-reasonix/extensions ~/.pi/agent/extensions/pi-reasonix

# Or project-local
ln -s /path/to/pi-reasonix/extensions .pi/extensions/pi-reasonix
```

## Verification

The extension logs on load:

```
[pi-reasonix] Loaded. Active for DeepSeek providers.
[pi-reasonix] Cache-First Loop | Tool-Call Repair | Cost Control
```

Run `/reasonix-status` inside pi to see live stats.

## How it works

| Hook | What it does |
|---|---|
| `before_provider_request` | Detects DeepSeek model, stabilises message prefix, compacts tool results, tracks context pressure |
| `after_provider_response` | Reads `x-cache-hit-tokens` / `x-cache-miss-tokens` from response headers |
| `tool_call` | Repairs truncated JSON in tool call arguments |
| `session_start` | Resets prefix state for a clean session |

## Architecture

```
pi-reasonix/
├── extensions/
│   └── index.ts          # Pi extension entry (event wiring)
├── src/
│   ├── cache-first.ts    # PrefixGuard + AppendOnlyLog
│   ├── repair.ts         # Tool-call repair pipeline
│   ├── cost-control.ts   # Turn-end compaction + context pressure
│   └── types.ts          # Shared types and interfaces
├── package.json          # Pi package manifest
└── README.md
```

## Building

```bash
npm install
npm run build    # TypeScript compilation
npm test         # Run tests (TBD)
```

## Publishing

```bash
npm login
npm publish
```

## License

MIT

## Acknowledgements

This package harvests innovations from [Reasonix](https://github.com/esengine/DeepSeek-Reasonix) (MIT, by the esengine community). All architectural credit goes to the Reasonix contributors for engineering DeepSeek-specific solutions that generic agent frameworks overlook.
