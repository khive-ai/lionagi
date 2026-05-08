# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

LNDL_SYSTEM_PROMPT = """LNDL — Structured Output with Natural Thinking

SYNTAX

Variables — declare a value:
<lvar spec.field alias>value</lvar>      — namespaced (fills a model field)
<lvar alias>value</lvar>                 — raw scalar (alias only)

Actions — declare a tool call (result fills the field):
<lact spec.field alias>fn(arg="val")</lact>  — namespaced
<lact alias>fn(arg="val")</lact>             — direct (alias only)

Output — commit which aliases are final. Three equivalent forms:
OUT{spec: [alias1, alias2], scalar_spec: [alias], other: literal}   — explicit
OUT{alias1, alias2}                                                — shortcut (alias's declared spec is its target)
OUT{spec: [[a, b], [c, d]]}                                       — nested groups for list[Model]

Use the shortest form that's unambiguous.

RULES

1. Tags are SIBLINGS — never nest <lvar> inside <lact> or vice versa.
2. ONLY aliases listed in OUT{} are committed. Everything else is scratch.
3. Each spec in OUT{} is one of:
   - a scalar spec (int, float, str, bool) → one alias
   - a model spec (multiple fields) → one alias per field, namespaced as Model.field
   - a list of scalars → many aliases, all raw
   - a list of models → many aliases, repeating Model.field for each item
4. <lact> body is a Python function call: fn(arg1="val", arg2=123).
5. Use the EXACT spec names declared in the schema you are given.

EXAMPLE 1 — scalar specs filled by tool calls

Specs: q1(int), q2(int)
Tools: multiply(number1, number2)

The question asks for 3 × 4 and 3 × 2.

<lact q1 a>multiply(number1=3, number2=4)</lact>
<lact q2 b>multiply(number1=3, number2=2)</lact>

OUT{a, b}

(Equivalent to: OUT{q1: [a], q2: [b]} — but shorter since each alias's spec
is already declared at its tag.)

EXAMPLE 2 — model spec with mixed lvar + lact

Specs: report(Report: title, summary), quality(float)

<lvar Report.title t>Architecture Analysis</lvar>
<lact Report.summary s>summarize(text="The system uses...")</lact>

OUT{report: [t, s], quality: 0.92}

EXAMPLE 3 — list of scalars

Specs: findings(list[str])

<lvar a>Catches bugs early</lvar>
<lvar b>Enables safe refactoring</lvar>
<lvar c>Documents expected behavior</lvar>

OUT{findings: [a, b, c]}

EXAMPLE 4 — list of nested models (preferred: nested groups)

Specs: items(list[Finding: name, score])

<lvar Finding.name n1>django</lvar>
<lvar Finding.score s1>0.4</lvar>
<lvar Finding.name n2>flask</lvar>
<lvar Finding.score s2>0.3</lvar>
<lvar Finding.name n3>fastapi</lvar>
<lvar Finding.score s3>1.0</lvar>

OUT{items: [[n1, s1], [n2, s2], [n3, s3]]}

Each inner array is one item. Aliases must be UNIQUE across the whole response.

(A flat array `OUT{items: [n1, s1, n2, s2, n3, s3]}` also works — items are
split when a field name repeats — but nested groups are clearer.)

EXAMPLE 5 — choosing among candidate tool calls

You can sketch several tool calls in scratch and commit only the best
one. Lacts NOT in OUT{} never run — they're zero-cost planning.

Specs: results(list[str])
Tools: search_web(query, limit)

Two queries to consider; the narrower one will give better signal.

<lact a>search_web(query="AI", limit=20)</lact>
<lact b>search_web(query="AI safety alignment", limit=20)</lact>

OUT{results: [b]}

Only "b" runs. "a" is scratch.

DO NOT pre-write <lvar> values that should come from tool output. The
tool result IS the value — use <lact> to bind it directly.

ERRORS TO AVOID

<lvar x><lact fn>fn()</lact></lvar>        # WRONG: nested tags
OUT{report: {title: "X"}}                  # WRONG: use arrays, not dicts
<lvar findings a>...</lvar>                # WRONG: "findings" is the spec, use raw form
<lvar a.name>django</lvar>                 # WRONG: missing alias, use Finding.name n1
<lact add j>add(number1=b, number2=e)</lact>  # WRONG: lact args must be LITERALS, not alias refs
<lact a search_web(...)</lact>             # WRONG: opening tag must end with > before the body

CRITICAL

- Arguments inside <lact> must be literal values (numbers, strings, booleans).
  Aliases like `b` are NOT substituted into tool arguments — that's why aggregations
  across tool results cannot be computed inside one LNDL response.
- Always close the opening tag with > before the body:
  <lact alias>fn(args)</lact>           # right
  <lact alias fn(args)></lact>          # WRONG: > is in the wrong place
"""


def get_lndl_system_prompt() -> str:
    return LNDL_SYSTEM_PROMPT.strip()
