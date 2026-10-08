# Porting status

Snapshot of the TypeScript → Python port. Regenerate the machine-checked part with:

```bash
uv run python scripts/porting_status.py
```

## Fork scope (read first)

This fork deliberately narrows the ported surface, so the upstream coverage numbers below
do **not** apply verbatim any more:

* APIs kept: `openai-responses`, `anthropic-messages`, `google-generative-ai`,
  `pi-messages`. `openai-completions` was removed and its providers (`deepseek`,
  `moonshotai`, `moonshotai-cn`, `xiaomi`) moved to `openai-responses`.
* Providers kept: `anthropic`, `google`, `openai`, `deepseek`, `minimax`, `minimax-cn`,
  `moonshotai`, `moonshotai-cn`, `xiaomi`. Every other provider factory, catalog JSON and
  test was deleted.
* Image generation and classification were deleted (`ImageApi`, `ClassifierApi`,
  `ImageModel`, `ClassifierModel`, `pi_ai.images*`, `Models.generate_images`/`classify`).
* Only API-key authentication remains; OAuth (`pi_ai/auth/oauth/`, `pi_ai/oauth.py`,
  `pi_ai/bun_oauth.py`) was deleted.

Deleted upstream modules are intentional gaps, recorded by `scripts/porting_status.py`.

## Source coverage

The upstream figures below describe the original full port, before this fork's pruning.

| | TypeScript | Python |
| --- | --- | --- |
| Source files | 199 | 153 |
| Source lines | 28,790 | 35,559 |
| Test files | 177 | 183 |
| Test lines | 45,583 | 39,559 |

```
TypeScript source files considered : 197
carried by a Python module        : 137
intentionally folded              : 60
NOT YET PORTED                    : 0
```

**Every one of the 197 non-generated TypeScript source files is now carried by a Python
module.** The 60 folded files are *not* gaps:

* `api/*.lazy.ts` (18) — a lazy wrapper is a one-line `lazy_api(lazy_load(...))` call site
  in Python, so no `*_lazy.py` module exists.
* `providers/*.models.ts` (42) — generated shims over `providers/data/*.json`; the JSON is
  copied verbatim and `pi_ai/providers/catalog.py` flattens it on demand.

See `PORTING.md` §10a for both folds.

## Test coverage

`.venv/bin/python -m pytest packages/ai/tests packages/agent/tests -q` from the repository root:

```
909 passed, 670 skipped
```

* **670 skips are upstream network-gated E2E tests.** The TypeScript suite gates the same
  tests on `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …; the port keeps the identical gate so
  the suite stays offline and deterministic. Export any of those variables and those tests
  run.
* The upstream vertex `xfail` no longer exists: the Google Vertex adapter was removed by
  this fork.

Test parity with upstream is intentionally reduced by this fork: the test files covering
the removed protocols, providers, image/classification and OAuth were deleted together with
their implementations. `packages/agent` remains complete: 4/4 test files plus its
`test/utils` helpers.

## What is deliberately different

Everything here is documented in the affected module's docstring.

### Transport

| Upstream | Port |
| --- | --- |
| `globalThis.fetch` / the `openai`, `anthropic`, `@google/genai` SDKs | `httpx.AsyncClient` built by `pi_ai.utils.http.create_client` |
| SDK error objects (`APIError`, `ApiError`) | module-local `ProviderHTTPError` / `AnthropicAPIError` / `GoogleApiError` shaped like the SDK error (`.status`, `.headers`, unwrapped `.error`) so `retry_provider_request` and `normalize_provider_error` behave identically |

### Runtime shape

* **Cancellation** — `AbortSignal`/`AbortController` are ported 1:1 rather than rewritten
  around `asyncio.Task.cancel`, so every ported module stays comparable with its source.
* **`stream()` needs a running event loop.** `lazy_stream` starts its worker with
  `asyncio.ensure_future`, so providers must be called from async code (TypeScript could
  call them from anywhere).
* **Node-only surfaces** have no counterpart and are not simulated: the Bun sandbox
  `process.env` fallback in `provider-env.ts` and the browser `node:fs` guards.
* **`compat.ts`** lazy-loads the sibling modules it star-re-exports, because
  `env-api-keys` and the API adapters are separate modules in Python rather than one flat
  namespace.

## Design decisions worth knowing

* **`Literal` vs `StrEnum`.** A closed union becomes a `Literal` alias unless the runtime
  uses its members as values; `StopReason` and friends stay `StrEnum`. `get_args()` derives
  the runtime tuple from the alias, so `KNOWN_PROVIDERS` is a one-line cross-check against
  `all_providers()` (both 9 in this fork). See `PORTING.md` §5.
* **`normalize_context()` is idempotent.** The TypeScript survives being called twice
  because a missing `systemPrompt` reads as `undefined`; Python needs the optional fields
  probed explicitly. `compat.stream()` depends on it.
* **Data model.** Dataclasses with snake_case fields and camelCase JSON aliases
  (`pi_ai.utils.serde`), so the 910 KB of generated catalog JSON round-trips losslessly.

## Bugs the ported tests found and that were fixed

The ported suite is the evidence that the port is faithful; writing it surfaced seven real
defects, all fixed:

1. `ModelsImpl._apply_auth` returned no request options when the caller passed none, so
   `models.complete(model, context)` sent unauthenticated requests.
2. The Anthropic adapter's GitHub Copilot path called helpers that no longer existed after
   they were de-duplicated into `api/github_copilot_headers.py`.
3. `normalize_context()` was not idempotent, so `compat.stream()` crashed on its Cloudflare
   fallback (the TypeScript survives because missing fields read as `undefined`).
4. `safe_json_stringify` used Python's spaced JSON separators where `JSON.stringify` is
   compact.
5. `ModelsImpl.get_all_models` had no `getModels()` fallback for legacy providers that omit
   `getAllModels`.
6. `ModelsImpl` passed bare `StreamOptions` to adapters that read provider-specific fields
   (`tool_choice`, `service_tier`, `client`, `deferred`), which Python turned into an
   `AttributeError` where JavaScript yields `undefined`.
7. `resolve_google_thinking_level` rendered `"null"` instead of the mapped value in its
   error message, and `convert_responses_tools` collapsed an explicit `strict: null` to the
   `false` default. There is now a `UNSET` sentinel that keeps `undefined` and `null`
   distinct, matching the JavaScript.

A pre-abort request also no longer schedules its work at all (see `AbortSignal`).

## Open work

1. `packages/agent/examples/mcp-codemode` is **not ported**: it imports
   `@earendil-works/pi-mcp` and `@earendil-works/pi-codemode`, neither of which has a
   Python counterpart in this repository.
2. `packages/ai/scripts/*.ts` (the catalog generators) are not ported; the generated JSON
   they produce is shipped verbatim (pruned to the nine providers), and the two tests that
   exercise the generators themselves are reported rather than faked.
3. This fork's removed protocols (OpenAI Chat Completions, Bedrock Converse, Azure OpenAI
   Responses, Google Vertex, Mistral Conversations, OpenAI Codex Responses, Cloudflare,
   TypeSafe System One, LlamaCpp classify, OpenRouter images), removed providers and removed
   OAuth are intentional and not considered open work. See the fork-scope note at the top.
