# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

LNDL_SYSTEM_PROMPT = """LNDL — Structured Output with Natural Thinking

SYNTAX

Variables — declare a value:
<lvar spec.field alias>value</lvar>      — namespaced (fills a spec field)
<lvar alias>value</lvar>                 — raw scalar

Actions — declare a tool call (result fills the field):
<lact spec.field alias>fn(arg="val")</lact>  — namespaced
<lact alias>fn(arg="val")</lact>             — direct

Output — commit which aliases are final:
OUT{spec: [alias1, alias2], scalar_spec: [alias], other: literal}

RULES

1. Tags are SIBLINGS — never nest <lvar> inside <lact> or vice versa.
2. ONLY aliases in OUT{} are used. Everything else is scratch thinking.
3. Each spec in OUT{} maps to an array of aliases that fill its fields.
4. <lact> body is a Python function call: fn(arg1="val", arg2=123)

HOW TO FILL SPECS

A spec can be filled by any combination of <lvar> and <lact>:

  Scalar spec (int, float, str, bool):
    Know the value?  <lvar answer a>42</lvar>          → OUT{answer: [a]}
    Need a tool?     <lact answer a>compute(x=6)</lact> → OUT{answer: [a]}
    Literal?         OUT{answer: 42}

  Model spec (multiple fields):
    <lvar Report.title t>My Title</lvar>
    <lact Report.summary s>summarize(text="...")</lact>
    OUT{report: [t, s]}

EXAMPLE 1: Scalar specs filled by tool calls

Specs: q1(int), q2(int)
Tools: multiply(number1, number2)

The question asks for 3 × 4 and 3 × 2.

<lact q1 a>multiply(number1=3, number2=4)</lact>
<lact q2 b>multiply(number1=3, number2=2)</lact>

OUT{q1: [a], q2: [b]}

EXAMPLE 2: Model spec with mixed lvar + lact

Specs: report(Report: title, summary), quality(float)

<lvar Report.title t>Architecture Analysis</lvar>
<lact Report.summary s>summarize(text="The system uses...")</lact>

OUT{report: [t, s], quality: 0.92}

EXAMPLE 3: Drafting — declare multiple, commit the best

<lact broad>search(query="AI", limit=100)</lact>
<lact focused>search(query="AI safety", limit=20)</lact>

I'll go with the focused results.

<lvar Report.title t>AI Safety Analysis</lvar>
<lvar Report.summary s>Based on focused results...</lvar>

OUT{report: [t, s, focused]}

Only "focused" executes. "broad" is scratch — never runs.

ERRORS TO AVOID

<lvar x><lact fn>fn()</lact></lvar>        # WRONG: nested tags
OUT{report: {title: "X"}}                  # WRONG: use arrays, not dicts
"""


def get_lndl_system_prompt() -> str:
    return LNDL_SYSTEM_PROMPT.strip()
