# pi-ai (Python)

A Python port of `@earendil-works/pi-ai` (upstream: `packages/ai`).

```python
from pi_ai import create_models
from pi_ai.providers.all import all_providers

models = create_models()
for provider in all_providers():
    models.set_provider(provider)

model = models.get_model("anthropic", "claude-fable-5")
message = await models.complete_simple(model, {"messages": [{"role": "user", "content": "hi"}]})
print(message.content[0].text)
```

Module topology mirrors the TypeScript source one-to-one:

| Upstream | Here |
| --- | --- |
| `src/types.ts` | `pi_ai/types.py` |
| `src/models.ts` | `pi_ai/models.py` |
| `src/api/*.ts` | `pi_ai/api/*.py` |
| `src/auth/**` | `pi_ai/auth/**` |
| `src/providers/*.ts` | `pi_ai/providers/*.py` |
| `src/utils/*.ts` | `pi_ai/utils/*.py` |
| `src/providers/data/*.json` | `pi_ai/providers/data/*.json` (verbatim) |

See `PORTING.md` at the repository root for the porting conventions.
