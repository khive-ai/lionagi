# LNDL — Language Network Directive Language

LNDL is a structured-output protocol where the model emits tagged text that
mixes natural reasoning, tool calls, and typed values in a single response.
The framework parses the tags and assembles the typed result.

It's an **alternative to JSON-mode structured output**. They both target the
same goal — get a `BaseModel` back from the LLM — but they take different
paths and have different sweet spots.

---

## Quick Start

```python
from pydantic import BaseModel
from lionagi import Branch, iModel

class Answer(BaseModel):
    n: int

branch = Branch(chat_model=iModel(model="openai/gpt-5.4-mini"))

def multiply(a: int, b: int) -> int:
    return a * b

out = await branch.operate(
    instruction="What is 7 * 8?",
    tools=multiply,
    actions=True,
    lndl=True,                    # <-- enable LNDL
    response_format=Answer,
)
print(out.n)  # 56
```

Under the hood the model emits something like:

```text
<lact a1>multiply(a=7, b=8)</lact>
OUT{n: [a1]}
```

The framework parses the tags, executes `multiply`, binds the result to
`n`, and validates the assembled value against `Answer`.

---

## When to use LNDL vs JSON

Use the table below; both modes coexist and you flip per call via `lndl=`.

| Situation | Pick |
|-----------|------|
| Many tool calls per response (≥3) | **LNDL** |
| Provider portability matters (GPT/Claude/Gemini/Ollama all behave the same) | **LNDL** |
| Tasks that interleave reasoning + tools + structured output | **LNDL** |
| Long multi-step research/exploration with `lndl_rounds=N` | **LNDL** |
| Single tool call (or none) + simple schema | JSON |
| Deeply nested schemas (3+ levels of `BaseModel`) | JSON |
| Schemas with `Literal[...]` discriminated unions | JSON |
| Flat classification / extraction / labelling | JSON |

**Rule of thumb:** if the model would otherwise emit ≥3 separate tool-call
turns, LNDL collapses them into one and saves the round-trip cost. For a
single classification call, JSON is strictly cheaper.

---

## Modes

### Single-round LNDL (`lndl=True`)

The model emits one LNDL response. The framework parses, executes any
`<lact>` calls, assembles the OUT block, and validates against
`response_format`. On parse/validation failure, optional `lndl_retries`
re-prompts the model with the error.

```python
out = await branch.operate(
    instruction="…",
    response_format=Report,
    tools=tools,
    actions=True,
    lndl=True,
    lndl_retries=1,    # default 0; one retry on failure
)
```

### Multi-round LNDL (`lndl_rounds=N`)

The model gets N chat turns to gather information before committing OUT.
In intermediate rounds it can issue `<lact>` calls without an OUT block;
tool results land in chat history; later rounds reason over them. The
loop stops at the first valid OUT.

```python
out = await branch.operate(
    instruction="Explore the codebase and produce a report.",
    response_format=Report,
    tools=tools,
    actions=True,
    lndl=True,
    lndl_rounds=4,
)
```

Use multi-round when the answer **depends on tool output the model can't
predict in advance** (codebase exploration, multi-step research, anything
where intermediate results steer the next call).

### ReActStream + LNDL

`branch.ReActStream(lndl=True, ...)` adds an outer ReAct loop on top of
LNDL. Each ReAct beat is its own `operate(lndl=True, lndl_rounds=N)` call;
the ReAct framework decides whether to extend based on `extension_needed`.

```python
async for analysis in branch.ReActStream(
    instruct={"instruction": "…"},
    response_format=Report,
    tools=tools,
    lndl=True,
    lndl_rounds=3,           # rounds inside each beat
    max_extensions=3,        # outer ReAct beats
):
    # each yield is a per-beat analysis;
    # the final yield is the full Report
    ...
```

Use this when you want the framework to decide when to stop, rather than
hard-capping with `lndl_rounds`.

---

## Syntax Reference

```text
<lvar spec.field alias>value</lvar>            # bind a literal value
<lvar alias>value</lvar>                       # raw scalar (alias only)
<lact spec.field alias>fn(arg="val")</lact>    # bind a tool-call result
<lact alias>fn(arg="val")</lact>               # tool call (alias only)

OUT{spec: [alias1, alias2], scalar: [alias]}  # explicit form
OUT{alias1, alias2}                            # shortcut form
OUT{spec: [[a, b], [c, d]]}                   # nested groups for list[Model]
```

**Critical rules:**

- Aliases must be **unique within one response.** Reusing `a` twice causes
  a parse error. The framework's example uses 2-character aliases (a1, b1,
  c1, …) — copy that pattern.
- `<lact>` arguments must be literal values. `<lact j>add(b, e)</lact>` is
  invalid because `b` and `e` are aliases, not values.
- Tags are **siblings**, never nested. `<lvar><lact>…</lact></lvar>` is
  invalid.
- Only aliases in OUT{} are committed. Lacts not in OUT{} are scratch
  (zero-cost planning, never executed in single-round mode).

For the full grammar see the system prompt in
[`lionagi/lndl/prompt.py`][lndl-prompt].

---

## Diagnostics: `LndlTrace`

LNDL provides opt-in telemetry. Pass an `LndlTrace` and the framework will
record one entry per LNDL parse attempt (including retries and multi-round
continuations).

```python
from lionagi.lndl import LndlTrace, classify_chunk, classify_result

trace = LndlTrace()
out = await branch.operate(
    instruction="…",
    response_format=Report,
    lndl=True,
    lndl_rounds=4,
    trace=trace,                 # <-- opt-in telemetry
)

print(trace.summary())
# LndlTrace(4 rounds | health={'clean': 4, 'malformed': 0, 'no_out': 0}
#  (100% clean) | outcomes={'continue': 1, 'success': 3})

# Per-round detail
for r in trace.rounds:
    print(r.outcome, r.schema, r.health.status, r.error)
```

### What the trace tells you

| Field | Meaning |
|-------|---------|
| `outcome` | What the framework decided this round: `success`, `continue`, `retry`, `failed`, `exhausted` |
| `schema` | The `BaseModel` class targeted this round |
| `error` | Validation/parse error if not success |
| `actions_executed` | Number of `<lact>` calls actually run |
| `health.status` | Syntactic health of the chunk: `clean`, `malformed`, `no_out` |
| `raw` | The model's literal LNDL emission |

### Three layers of classification

LNDL gives you three complementary views of "did it work?":

| Function | Question answered |
|----------|------------------|
| `classify_chunk(text)` | Did the model write valid LNDL syntax? |
| `LndlRoundRecord.outcome` | What did the framework do with it? |
| `classify_result(value)` | What did the user actually get? |

A response can be `clean` (good syntax) but `failed` (schema mismatch) and
return a `dict` (validation fell back).

### Default = zero overhead

`trace=None` is the default. No `LndlTrace` instance, no work, no cost.

---

## Failure Modes (and how the trace surfaces them)

### Duplicate alias collision

```text
LNDL syntax error: Parse error at line 41, column 117: Duplicate alias 'i'
```

The model used the same alias twice in one response. Almost always
single-letter aliases (`a`, `i`, `r`) on schemas with many fields. The
framework's example aliases are 2-character (a1, b1, …) precisely to
discourage this; encourage the model in your prompt to copy the pattern.

### Schema-friction validation failure

```text
Validation against ReportResponse failed: 9 validation errors for ReportResponse
```

The model emitted clean LNDL but the values don't match the schema's
types. Most common with `Literal[...]` fields, deeply nested optionals, or
`list[Model]` with required fields the model treats as optional. Trim the
schema or set `lndl_retries=1`.

### Fix-mode contagion

After one retry, the model can shift into "describe my plan" mode and
stop issuing `<lact>` tool calls in subsequent rounds. The trace shows
`actions=0` on every later round.

The reparse prompt now explicitly says "preserve the SAME tool calls" but
this is still possible on weaker models. If you see it consistently, drop
`lndl_retries` to 0 and let the LndlTrace `failed` outcomes surface
loudly.

### First-attempt JSON fallback

Model emits JSON `{"…": "…"}` instead of LNDL on the first turn. Caught
by retry; usually one round of cost. Mitigated by the LNDL system prompt
injection but not eliminated.

### Tool-name hallucination

```text
Error invoking action 'tool': Function tool is not registered.
```

Model emits `<lact a>tool(...)</lact>` — the literal string `tool`, not a
real function name. Action invocation rejects it; chat history gets
polluted with an error response. Make sure your registered tools have
distinctive names.

---

## Best Practices

### Schema design

- **Prefer scalars + flat `list[Model]`.** Avoid 3+ levels of nesting.
- **Avoid `Literal[...]` and discriminated unions** — model will pick
  out-of-set values often enough to bite you.
- **Optional fields with `default_factory=list`** are silently treated as
  optional by Pydantic; combine with `Field(...)` (no default) when you
  need them required.
- **Field descriptions become prompt instructions.** Use them to nudge
  behavior: `Field(description="Cite path:line from real tool output. Emit [] if no search was run.")`

### Prompt design

- **Inject the LNDL system prompt automatically** — the framework does
  this when `lndl=True`; idempotent; restored on exit.
- **Don't overload the user instruction with schema hints** — let the
  rendered schema example do that work.
- **Tell the model how many rounds it has** — multi-round mode does this
  automatically via the continuation builder.

### Using the trace

- **Always pass a trace in development.** It's free and tells you exactly
  why a run failed.
- **Tune `lndl_rounds` based on retry rate.** If you see ≥30% retry
  outcomes, lower `lndl_rounds` and lean on `lndl_retries` instead.
- **Track citation accuracy or schema completeness** as quality metrics
  parallel to the trace.

### Provider tuning

- **gpt-5.4-mini** — 80% reliable on flat schemas; struggles with
  nested. Good for cost-sensitive workloads.
- **gpt-5.4 / claude-sonnet-4.6** — 95%+ reliable; recommended for
  production.
- **qwen3.6-27b** — surprisingly strong on LNDL given its size; matches
  gpt-5.4-mini quality at 2x latency.

---

## API Surface

### Public exports from `lionagi.lndl`

```python
from lionagi.lndl import (
    # Diagnostics (opt-in)
    LndlTrace,           # container, pass via trace=
    LndlRoundRecord,     # per-round entry
    LndlChunkHealth,     # syntactic verdict
    classify_chunk,      # text → LndlChunkHealth
    classify_result,     # value → "ok"|"str"|"dict"|"empty"
    extract_lndl_chunks, # messages → list[str]

    # Round-outcome ADT (used internally; surfaced for advanced use)
    Success, Continue, Retry, Failed, Exhausted, RoundOutcome,

    # Errors
    LNDLError, ParseError, MissingFieldError, MissingLvarError,
    MissingOutBlockError, TypeMismatchError, AmbiguousMatchError,

    # Lower-level (rarely needed)
    Lexer, Parser, Token, TokenType,
    assemble, collect_actions, collect_notes,
    LNDL_SYSTEM_PROMPT, get_lndl_system_prompt,
)
```

### Branch methods that accept LNDL params

```python
# Single-shot LNDL operate
out = await branch.operate(
    instruction=...,
    response_format=Schema,
    lndl=True,
    lndl_rounds=N,        # default 1; >1 = multi-round
    lndl_retries=K,       # default 0; K retries on parse/validation failure
    trace=trace,          # default None; LndlTrace() to record
)

# Streaming ReAct + LNDL
async for analysis in branch.ReActStream(
    instruct={...},
    response_format=Schema,
    lndl=True,
    lndl_rounds=N,
    lndl_retries=K,
    trace=trace,
    max_extensions=M,     # outer ReAct cap
):
    ...

# Pull raw LNDL from messages (no trace required)
chunks = branch.lndl_chunks(since=start_idx)
```

---

## Cookbook examples

- [`cookbooks/lndl/_cases_runner.py`][cases] — 15 cases covering scalars,
  lists, nested models, dicts, unions, multi-tool chains, and retry recovery
- [`cookbooks/lndl/explore_agent.py`][explore] — read-only codebase agent,
  multi-round LNDL with `CodingToolkit.READ_ONLY`
- [`cookbooks/lndl/chain_analysis.py`][chain] — phased pipeline (4 phases,
  each its own `operate(lndl=True)` on a shared branch)
- [`cookbooks/lndl/react_analysis.py`][react] — `ReActStream(lndl=True)`
  with `LndlTrace` for full diagnostic visibility

[lndl-prompt]: https://github.com/khive-ai/lionagi/blob/main/lionagi/lndl/prompt.py
[cases]: https://github.com/khive-ai/lionagi/blob/main/cookbooks/lndl/_cases_runner.py
[explore]: https://github.com/khive-ai/lionagi/blob/main/cookbooks/lndl/explore_agent.py
[chain]: https://github.com/khive-ai/lionagi/blob/main/cookbooks/lndl/chain_analysis.py
[react]: https://github.com/khive-ai/lionagi/blob/main/cookbooks/lndl/react_analysis.py
