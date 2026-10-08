# pi (Python)

An idiomatic Python port of the TypeScript packages **`@earendil-works/pi-ai`** and
**`@earendil-works/pi-agent-core`**.

| TypeScript | Python |
| --- | --- |
| `WebStromProjects/pi/packages/ai` | [`packages/ai/pi_ai`](packages/ai/pi_ai) |
| `WebStromProjects/pi/packages/agent` | [`packages/agent/pi_agent`](packages/agent/pi_agent) |

Every non-generated TypeScript source file has a Python counterpart — see
[PORTING_STATUS.md](PORTING_STATUS.md) for the coverage report and the handful of
deliberate deviations.

## Install and test

```bash
uv sync
uv run pytest                 # 909 passed, 670 skipped
uv run python scripts/porting_status.py
```

The 670 skips are upstream network-gated E2E tests (they run when the matching
`*_API_KEY` is exported), so the default suite is fully offline and deterministic.

## Use

```python
from pi_ai import create_models, SimpleStreamOptions
from pi_ai.providers.all import all_providers
from pi_ai.types import Context, UserMessage

models = create_models()
for provider in all_providers():
    models.set_provider(provider)

model = models.get_model("anthropic", "claude-fable-5")
message = await models.complete_simple(
    model,
    Context(messages=[UserMessage(content="hello", timestamp=0)]),
    SimpleStreamOptions(api_key="..."),
)
print(message.content[0].text)
```

```python
from pi_agent import Agent, AgentOptions

agent = Agent(AgentOptions(model=model, convert_to_llm=convert))
await agent.prompt("hello")
```

## Documentation

* [PORTING.md](PORTING.md) — the porting conventions: data model, naming, async,
  cancellation, streams, and the `Literal`-vs-`StrEnum` rule.
* [TESTPORT.md](TESTPORT.md) — how the upstream test suite maps onto pytest.
* [PORTING_STATUS.md](PORTING_STATUS.md) — what is ported, what is folded, and what
  deliberately differs from the TypeScript.

## Layout

```
packages/ai/pi_ai/        core data model, Models registry, auth, provider adapters
packages/ai/pi_ai/api/    one module per wire protocol (4 protocols)
packages/ai/pi_ai/providers/  9 built-in provider factories + the verbatim model catalogs
packages/ai/tests/        pytest port of packages/ai/test
packages/agent/pi_agent/  agent loop, agent runtime, transport proxy
packages/agent/tests/     pytest port of packages/agent/test
scripts/porting_status.py coverage report
```

## Scope of this fork

This fork keeps a deliberately small protocol surface:

* **APIs** — `openai-responses`, `anthropic-messages`, `google-generative-ai`,
  `pi-messages`. The `openai-completions` adapter was removed; providers that used it
  (`deepseek`, `moonshotai`, `moonshotai-cn`, `xiaomi`) now use `openai-responses`.
* **Providers** — `anthropic`, `google`, `openai`, `deepseek`, `minimax`, `minimax-cn`,
  `moonshotai`, `moonshotai-cn`, `xiaomi`.
* **Chat models only** — image generation and classification are removed.
* **Authentication** — API keys only; OAuth login/refresh is removed.

See [PORTING.md](PORTING.md) §5a for the precise removals.
