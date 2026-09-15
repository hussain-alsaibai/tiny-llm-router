# tiny-llm-router

> **Zero-dependency model router for Python AI agents.** Pick the right model per role — fast for chat, deep for think, vision-capable for images — with cost caps, latency guards, and circuit-breakers. One file. MIT.

```bash
pip install tiny-llm-router
# or: copy tiny_llm_router.py into your project
```

![version 0.1.0](https://img.shields.io/badge/version-0.1.0-blue)

```python
from tiny_llm_router import Router, RoleRule, Task

router = Router([
    RoleRule("code",    "gpt-4o-mini", max_cost_per_1k=0.5),
    RoleRule("think",   "o1-mini"),
    RoleRule("vision",  "gpt-4o"),
    RoleRule("long",    "gemini-2.5-flash"),
    RoleRule("json",    "haiku"),
    RoleRule("default", "gpt-4o-mini"),
], fallback="haiku")

model = router.pick(Task(role="code", prompt="def hello(): return 1"))
print(model)   # gpt-4o-mini

# After each real call:
router.record(model, latency_ms=120, cost_usd=0.0023)
```

## Why?

The agent model landscape is no longer "pick one". Every serious agent today
routes between 4–6 models:

| Model | Role | Why |
|-------|------|-----|
| `gpt-4o-mini` / `haiku` | fast chat | $0.15/M input, sub-second |
| `gpt-4o` | vision | best multimodal bang/buck |
| `o1-mini` / `nemotron-3` | think | heavy reasoning when it matters |
| `qwen3-coder` | code | code-specialized |
| `gemini-2.5-flash` | long context | 1M+ tokens, cheap |
| `kimi-k2.6` | default | free-tier, fast, decent |

The hard part is **the routing logic itself** — cost caps, latency SLAs,
fallback chains, and breaking a flaky provider after N errors. Tiny-llm-router
gives you all of that in 280 LOC.

## Features

- **Role-based picking** — `Task(role=...)` selects the right model
- **Heuristic inference** — auto-detect code/long/json/vision when caller skips `.role`
- **Cost caps** — `max_cost_per_1k` per role rule, with pricing table
- **Latency guards** — `max_latency_p95_ms` against observed p95
- **Circuit-breakers** — auto-skip models with N errors in window
- **Statistics** — rolling avg latency/cost + last-20 p95 proxy
- **Zero dependencies** — pure stdlib, copy-pasteable
- **Interop** — `make_tiny_llm_adapter(router, tiny_llm)` for auto-routing

## Use with tiny-llm

```python
import tiny_llm
from tiny_llm_router import Router, RoleRule, Task, make_tiny_llm_adapter

router = Router([
    RoleRule("code",    "gpt-4o-mini"),
    RoleRule("think",   "o1-mini"),
    RoleRule("vision",  "gpt-4o"),
    RoleRule("default", "haiku"),
], fallback="haiku")

adapter = make_tiny_llm_adapter(router, tiny_llm)
resp = adapter.chat_auto(Task(role="code", prompt="def fib(n): ..."))
print(resp, "via", adapter.last_model)
```

## Why "tiny"?

The OpenClaw ecosystem is built on zero-dep, single-file MIT libraries:

- `tiny-llm` — multi-provider LLM client
- `tiny-cost` — token cost + budget enforcement
- `tiny-mcp-client` — MCP server consumer
- `tiny-mcp-server` — MCP server author
- `tiny-mcp-registry` — server health discovery
- `tiny-event-bus`, `tiny-router`, `tiny-config`, `tiny-log`, `tiny-validator`,
  `tiny-cli`, `tiny-event-emitter`, `tiny-semaphore`, `tiny-rate-limiter`,
  `tiny-circuit-breaker`, `tiny-bounty`, `tiny-budget`, `tiny-prompt-cache`,
  `tiny-semantic-cache`, `tiny-trace`, `tiny-eval`, `tiny-store`, `tiny-task`,
  `tiny-workflow-engine`, and 30+ more.

`tiny-llm-router` slots in as the **brain above `tiny-llm`** — once you have
multi-provider access, the next decision is *which* provider per call.

## Compatibility

- Python 3.8 – 3.13
- No C extensions, no native code
- Thread-safe (single internal `_STAT_LOCK`)

## License

MIT — copy-paste, modify, ship.
