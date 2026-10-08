# pi → Python porting conventions

This repository is a faithful Python port of two TypeScript packages:

| TypeScript source | Python target |
| --- | --- |
| `WebStromProjects/pi/packages/ai` | `packages/ai/pi_ai` |
| `WebStromProjects/pi/packages/agent` | `packages/agent/pi_agent` |

The port keeps **behaviour and module topology** identical so the two codebases can be
diffed file-by-file, while the *surface* is idiomatic Python.

Source tree (read these when porting):

```
/Users/huangzd00/Projects/WebStromProjects/pi/packages/ai/src
/Users/huangzd00/Projects/WebStromProjects/pi/packages/ai/test
/Users/huangzd00/Projects/WebStromProjects/pi/packages/agent/src
/Users/huangzd00/Projects/WebStromProjects/pi/packages/agent/test
```

## 1. Naming

| TypeScript | Python |
| --- | --- |
| `src/foo/bar-baz.ts` | `pi_ai/foo/bar_baz.py` |
| `fooBar()` function | `foo_bar()` |
| `foo-bar.ts` test | `test/tests/foo_bar_test.py` |
| `FooBar` type / interface | `FooBar` dataclass or `Protocol` |
| `CONSTANT_CASE` const | `CONSTANT_CASE` module constant |

Private helpers that TypeScript marks with a leading `_` convention or module privacy use a
leading underscore in Python.

## 2. Data model: dataclasses with snake_case + JSON aliases

All structural types are `@dataclass` with **snake_case** field names. The original
camelCase wire/JSON name is recorded in field metadata so that reading the generated model
catalog and (de)serialising messages stays lossless:

```python
@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = field(default=0, metadata={"alias": "cacheRead"})
```

Use `pi_ai.utils.serde` for conversion:

```python
from pi_ai.utils.serde import to_json, from_json

payload = to_json(message)          # dataclass -> JSON-compatible dict/list/scalars
message = from_json(AssistantMessage, payload)
```

`to_json` **omits dataclass fields whose value is `None`** (mirroring how
`JSON.stringify` drops `undefined`). `None` inside a plain `dict` or `list` is kept, because
there it means JSON `null`.

Unions of `@dataclass` content blocks are resolved from the `type` discriminator key.
Register a block type with the `@content_block("text")` decorator from
`pi_ai.utils.serde`.

## 3. Async

Every operation that performs I/O is `async def`. Imports that TypeScript does dynamically
(`await import(...)`) become `importlib.import_module(...)` executed inside the async function.

## 4. Cancellation

TypeScript `AbortSignal` / `AbortController` are ported 1:1 by
`pi_ai.utils.abort.AbortSignal` / `AbortController` (same names, same semantics:
`.aborted`, `.reason`, `.throw_if_aborted()`, `await signal.wait()`,
`signal.add_listener(cb)`, `signal.any(signals)`, `signal.timeout(ms)`).

Helpers:

* `operation_signal(signal)` — never-`None` signal for optional public parameters.
* `race_with_abort_signal(awaitable, signal)` — stop waiting when aborted while still
  observing the abandoned coroutine so a later exception is never "never retrieved".

Prefer `async with` / `try/finally` cleanup; the port must not leak tasks.

## 5. TypeScript unions and open string types

The rule is *does the runtime ever use a member as a value?*

| Shape | Python |
| --- | --- |
| Closed union used as **values** in code (`default=`, `==`, assignment) | `StrEnum` |
| Closed union that is **type-only** / documentation | `Literal[...]` alias (+ `tuple` constant when runtime discovery helps) |
| Open union written upstream as `KnownX \| (string & {})` | plain `str`, with the known set as a `Literal` alias named `KnownX` |
| Errors with a stable code | `StrEnum` (`ModelsErrorCode`) |

`Literal` is the direct analogue of a TypeScript string-literal union: no class, no import
cost, and the same exhaustiveness information for a type checker.  An enum is only worth its
ceremony when something actually reads a member.  For example, in `pi_ai/types.py`:

```python
KnownApi: TypeAlias = Literal[
    "openai-responses", "anthropic-messages", "google-generative-ai", "pi-messages"
]
KNOWN_APIS: tuple[KnownApi, ...] = get_args(KnownApi)   # one source of truth
Api: TypeAlias = str                                    # KnownApi | (string & {})

class StopReason(StrEnum):                              # 41 runtime uses as a value
    STOP = "stop"
    TOOL_USE = "toolUse"
```

`get_args()` derives the runtime constant from the alias, so the two cannot drift.
Deriving `KNOWN_PROVIDERS` this way is also a cheap cross-check against the provider
factories: it must stay equal to `all_providers()`.

* Structural object types (`ProviderStreams`, `ApiKeyAuth`, `CredentialStore`,
  `ModelsStore`, `AuthContext`, `Provider`, `Models`) → `typing.Protocol`.
  Concrete implementations are plain classes that need not inherit the Protocol.
* `Record<string, T>` → `dict[str, T]`; `readonly T[]` → `Sequence[T]`; `T[]` → `list[T]`.
* `JsonValue` → `pi_ai.types.JsonValue` (recursive alias).

## 5a. Reduced protocol surface (this fork)

This fork deliberately narrows the upstream surface:

* **`KnownApi`** is the closed set of four wire protocols above. The
  `openai-completions` adapter (`pi_ai/api/openai_completions.py`) is gone; the providers
  that used it (`deepseek`, `moonshotai`, `moonshotai-cn`, `xiaomi`) now serve their models
  through the `openai-responses` adapter, and their catalog `api` fields say
  `openai-responses`.
* **`KnownProvider`** is the closed set of nine providers above; every other provider
  factory, catalog JSON and test was deleted.
* **Image generation and classification are gone** — `ImageApi`/`ClassifierApi`,
  `ImageModel`/`ClassifierModel`, `Models.generate_images`/`classify`,
  `ProviderImages`/`ProviderClassifier`, `pi_ai.images*` and their catalogs.
* **Only API-key authentication exists** — no OAuth credential, login, refresh or
  provider-auth method; `pi_ai/auth/oauth/`, `pi_ai/oauth.py` and `pi_ai/bun_oauth.py` were
  deleted, and `ProviderAuth` only carries `api_key`.

Everything else follows the upstream porting conventions below.

## 6. Errors

`ModelsError(Exception)` with a `.code` attribute; codes match
`ModelsErrorCode` in `pi_ai/utils/models_error.py`. Provider/stream failures are **not**
raised across an `AssistantMessageEventStream` — they are encoded as an `AssistantMessage`
with `stop_reason="error"` or `"aborted"` plus `error_message`, exactly like TypeScript.

## 7. Tool schemas

TypeBox/`TSchema` is replaced by plain JSON Schema (`dict[str, Any]`). The
`Type.StringEnum(...)` helper becomes `pi_ai.utils.typebox_helpers.string_enum(...)`
returning the equivalent JSON Schema dict. Validation uses `jsonschema`.

## 8. Streams

`AssistantMessageEventStream` is ported as-is: `push(event)`, `end(result=None)`,
`async for event in stream`, `await stream.result()`. Event payloads are the
`AssistantMessageEvent` dataclasses from `pi_ai/types.py`. Build them with the small
constructors in `pi_ai/utils/event_stream.py` (`event_start`, `text_delta`, …) instead of
hand-writing dataclasses, to keep the emit sites readable.

## 9. HTTP

Use `httpx.AsyncClient`. A caller-supplied `fetch` option maps to an
`httpx.AsyncBaseTransport`-like callable `FetchFunction` (see `pi_ai/types.py`); default is
`httpx`. SSE parsing is ported by hand, as in the TypeScript, into
`pi_ai/utils/event_stream.py`'s sibling modules (`pi_ai/utils/sse.py`).

## 10. Generated data is copied, not translated

`src/providers/data/*.json` is pure data. It is copied **verbatim** to
`pi_ai/providers/data/`, pruned to the nine retained providers. The generated `.models.ts`
shims are replaced by `pi_ai/providers/catalog.py`, which flattens the JSON into chat model
catalogs with `from_json`.

## 10a. Module topology: the intentional fold

Python cannot express one upstream shape directly, so one TypeScript file is folded
into a neighbour. `scripts/porting_status.py` records it:

* `src/api/*.lazy.ts` (18 files) — the lazy wrapper's only job is to defer an
  `await import()`. In Python that is a one-line call site, so the provider factories write
  `lazy_api(lazy_load("pi_ai.api.<module>"))` instead of a `*_lazy.py` module.

## 11. Tests

Tests live in `packages/ai/tests` and `packages/agent/tests`, mirroring the TypeScript
`test/` file names (`anthropic-sse-parsing.test.ts` → `tests/anthropic_sse_parsing_test.py`).
Written with `pytest` (`pytest-asyncio` for async). Vitest mocks (`vi.fn()`, `vi.mock()`)
become small hand-written fakes / `monkeypatch`. Tests must be deterministic and must not
touch the network.

## 12. Definition of done

* `uv run pytest` passes from the repository root.
* `uv run python -c "import pi_ai, pi_agent"` succeeds.
* Every module of the TypeScript source has a Python counterpart, or a documented reason
  why it does not (see `PORTING_STATUS.md`).
