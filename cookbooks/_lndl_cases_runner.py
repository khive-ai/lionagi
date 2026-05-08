"""Runner for LNDL test cases. Executes each case, prints raw + parsed output.

Run with: uv run python cookbooks/_lndl_cases_runner.py
Set CASE=N to run a single case (e.g. CASE=3).
"""

import asyncio
import os
import traceback

from pydantic import BaseModel

from lionagi import Branch, iModel

MODEL = os.environ.get("LNDL_MODEL", "openai/gpt-5.4-mini")
ONLY = os.environ.get("CASE")  # "1", "2", ... or None for all


def multiply(number1: float, number2: float) -> float:
    """Multiply two numbers."""
    return number1 * number2


def add(number1: float, number2: float) -> float:
    """Add two numbers."""
    return number1 + number2


def divide(numerator: float, denominator: float) -> float:
    """Divide numerator by denominator."""
    return numerator / denominator


def subtract(number1: float, number2: float) -> float:
    """Subtract number2 from number1."""
    return number1 - number2


def summarize(text: str) -> str:
    """Return a short summary of the given text."""
    return f"Summary: {text[:60]}"


def search_web(query: str, limit: int = 3) -> list[str]:
    """Search the web (stub). Returns ``limit`` mock results."""
    return [f"[{query!r}] Result {i+1}" for i in range(limit)]


def score_relevance(claim: str, evidence: str) -> float:
    """Score how strongly evidence supports a claim. Stub returning 0.5..0.95."""
    h = abs(hash((claim, evidence))) % 100
    return 0.5 + (h / 100) * 0.45


def fetch_price(symbol: str) -> float:
    """Stub: return a deterministic mock price for a stock symbol."""
    h = abs(hash(symbol)) % 1000
    return round(50.0 + h * 0.37, 2)


def fetch_volume(symbol: str) -> int:
    """Stub: return a deterministic mock daily volume."""
    return abs(hash(symbol)) % 1_000_000


def fresh_branch(system: str = "You produce clean LNDL output.", tools=None):
    return Branch(
        system=system,
        tools=tools,
        chat_model=iModel(model=MODEL),
    )


def banner(label: str):
    bar = "=" * 70
    print(f"\n{bar}\n{label}\n{bar}")


def show(out, branch=None, label="result"):
    print(f"  type: {type(out).__name__}")
    print(f"  value: {out!r}")
    if branch and len(branch.messages) > 2:
        raw = getattr(branch.messages[2].content, "assistant_response", None)
        if raw:
            print("  raw assistant LNDL:")
            for line in str(raw).split("\n"):
                print(f"    | {line}")


def expect(label: str, cond: bool, detail: str = ""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {label}{(' — ' + detail) if detail else ''}")
    return cond


# ---------------------------------------------------------------------------
# Case 1 — Scalar specs filled by tool calls
# ---------------------------------------------------------------------------
async def case_1():
    banner("Case 1 — scalar specs filled by <lact>")

    class Answer(BaseModel):
        q1: int
        q2: int

    branch = fresh_branch(tools=multiply)
    out = await branch.operate(
        instruction="Compute 3*4 for q1 and 3*2 for q2 using the multiply tool.",
        actions=True,
        lndl=True,
        response_format=Answer,
    )
    show(out, branch)
    expect("returned a model (not dict)", not isinstance(out, dict))
    expect("has q1 field", hasattr(out, "q1") if not isinstance(out, dict) else "q1" in out)
    expect("has q2 field", hasattr(out, "q2") if not isinstance(out, dict) else "q2" in out)
    expect(
        "has action_responses",
        hasattr(out, "action_responses") if not isinstance(out, dict) else "action_responses" in out,
    )


# ---------------------------------------------------------------------------
# Case 2 — Pure structured extraction (no tools, no actions)
# ---------------------------------------------------------------------------
async def case_2():
    banner("Case 2 — pure structured extraction (no tools)")

    class Person(BaseModel):
        name: str
        age: int

    branch = fresh_branch()
    out = await branch.operate(
        instruction="Extract: 'Alice is thirty years old.'",
        lndl=True,
        response_format=Person,
    )
    show(out, branch)
    expect("returned Person model", isinstance(out, Person))
    if isinstance(out, Person):
        expect("name is Alice", out.name.lower().startswith("alice"))
        expect("age is 30", out.age == 30)


# ---------------------------------------------------------------------------
# Case 3 — Single nested model with mixed lvar + lact
# ---------------------------------------------------------------------------
async def case_3():
    banner("Case 3 — nested Report with lvar + lact")

    class Report(BaseModel):
        title: str
        summary: str

    class Container3(BaseModel):
        report: Report

    branch = fresh_branch(tools=summarize)
    out = await branch.operate(
        instruction=(
            "Build a Report. Title is 'AI Safety Analysis'. "
            "For the summary, call summarize with text='AI safety studies alignment risks'."
        ),
        actions=True,
        lndl=True,
        response_format=Container3,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        report = getattr(out, "report", None)
        expect("has report field", report is not None)
        if report is not None:
            expect("report is Report instance", isinstance(report, Report))
            expect("report has title", bool(getattr(report, "title", None)))
            expect("report has summary", bool(getattr(report, "summary", None)))


# ---------------------------------------------------------------------------
# Case 4 — list[str]
# ---------------------------------------------------------------------------
async def case_4():
    banner("Case 4 — list[str]")

    class FindingsList(BaseModel):
        findings: list[str]

    branch = fresh_branch()
    out = await branch.operate(
        instruction="List three key benefits of unit testing as findings.",
        lndl=True,
        response_format=FindingsList,
    )
    show(out, branch)
    expect("returned FindingsList", isinstance(out, FindingsList))
    if isinstance(out, FindingsList):
        expect("findings is list", isinstance(out.findings, list))
        expect("findings has 3 items", len(out.findings) == 3, f"got {len(out.findings)}")


# ---------------------------------------------------------------------------
# Case 5 — list[NestedModel]
# ---------------------------------------------------------------------------
async def case_5():
    banner("Case 5 — list[Finding] with score")

    class Finding(BaseModel):
        name: str
        score: float

    class Catalog(BaseModel):
        items: list[Finding]

    branch = fresh_branch()
    out = await branch.operate(
        instruction=(
            "Rate three Python web frameworks (django, flask, fastapi) "
            "on a 0-1 scale based on async support."
        ),
        lndl=True,
        response_format=Catalog,
    )
    show(out, branch)
    expect("returned Catalog", isinstance(out, Catalog))
    if isinstance(out, Catalog):
        expect("items is list", isinstance(out.items, list))
        expect("items has 3 entries", len(out.items) == 3, f"got {len(out.items)}")
        if out.items:
            expect("first item is Finding", isinstance(out.items[0], Finding))


# ---------------------------------------------------------------------------
# Case 6 — Optional / nullable field
# ---------------------------------------------------------------------------
async def case_6():
    banner("Case 6 — optional / nullable")

    class OptDoc(BaseModel):
        required: str
        optional: str | None = None

    branch = fresh_branch()
    out = await branch.operate(
        instruction="Set required to 'hello'. Leave optional unset.",
        lndl=True,
        response_format=OptDoc,
    )
    show(out, branch)
    expect("returned OptDoc", isinstance(out, OptDoc))
    if isinstance(out, OptDoc):
        expect("required == 'hello'", out.required.lower().strip() == "hello", repr(out.required))
        expect("optional is None", out.optional is None, repr(out.optional))


# ---------------------------------------------------------------------------
# Case 7 — Reason + actions + LNDL
# ---------------------------------------------------------------------------
async def case_7():
    banner("Case 7 — reason + actions + LNDL")

    class Q(BaseModel):
        q: int

    branch = fresh_branch(tools=multiply)
    out = await branch.operate(
        instruction="What is 5 * 7? Use the multiply tool. Explain your reasoning briefly.",
        reason=True,
        actions=True,
        lndl=True,
        response_format=Q,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        expect("has q", hasattr(out, "q"))
        expect("has reason", hasattr(out, "reason") and getattr(out, "reason", None) is not None)
        expect("has action_responses", hasattr(out, "action_responses"))


# ---------------------------------------------------------------------------
# Case 8 — Mixed types: nested + scalar + list
# ---------------------------------------------------------------------------
async def case_8():
    banner("Case 8 — mixed nested + scalar + list")

    class Report(BaseModel):
        title: str
        summary: str

    class Container8(BaseModel):
        report: Report
        score: float
        tags: list[str]

    branch = fresh_branch()
    out = await branch.operate(
        instruction=(
            "Build a Container with: a Report (title='Climate Update', "
            "summary='Carbon levels rising'), score=0.85, "
            "and tags=['climate', 'science', 'urgent']."
        ),
        lndl=True,
        response_format=Container8,
    )
    show(out, branch)
    expect("returned Container8", isinstance(out, Container8))
    if isinstance(out, Container8):
        expect("report is Report", isinstance(out.report, Report))
        expect("score is float", isinstance(out.score, float))
        expect("tags is list", isinstance(out.tags, list))


# ---------------------------------------------------------------------------
# Case 9 — dict[str, float]
# ---------------------------------------------------------------------------
async def case_9():
    banner("Case 9 — dict[str, float]")

    class Metrics(BaseModel):
        metrics: dict[str, float]

    branch = fresh_branch()
    out = await branch.operate(
        instruction="Report metrics: precision=0.9, recall=0.85, f1=0.87.",
        lndl=True,
        response_format=Metrics,
    )
    show(out, branch)
    expect("returned Metrics", isinstance(out, Metrics))
    if isinstance(out, Metrics):
        expect("metrics is dict", isinstance(out.metrics, dict))
        expect("has 3 entries", len(out.metrics) == 3, f"got {len(out.metrics)}")


# ---------------------------------------------------------------------------
# Case 10 — Union type
# ---------------------------------------------------------------------------
async def case_10():
    banner("Case 10 — int | str union")

    class Either(BaseModel):
        result: int | str

    branch = fresh_branch()
    out = await branch.operate(
        instruction="Set result to the integer 42.",
        lndl=True,
        response_format=Either,
    )
    show(out, branch)
    expect("returned Either", isinstance(out, Either))


# ===========================================================================
# COMPLEX CASES — what LNDL is actually for: many tools, deep schemas, single API call
# ===========================================================================


# ---------------------------------------------------------------------------
# Case 11 — Research pipeline
#   3 nested model types, 3 tools, single call assembles everything.
# ---------------------------------------------------------------------------
async def case_11():
    banner("Case 11 — research pipeline (3 tools, 3 nested models)")

    class Evidence(BaseModel):
        snippet: str
        score: float

    class Finding(BaseModel):
        claim: str
        evidence: list[str]
        relevance: float

    class Summary(BaseModel):
        verdict: str
        key_points: list[str]

    class ResearchReport(BaseModel):
        query: str
        findings: list[Finding]
        summary: Summary
        confidence: float

    branch = fresh_branch(
        system="You are a research assistant. Use tools to gather evidence.",
        tools=[search_web, score_relevance, summarize],
    )
    out = await branch.operate(
        instruction=(
            "Research 'is unit testing worth the cost'. Produce two findings, "
            "each with a claim and 2 evidence snippets (use search_web). "
            "Score each finding's relevance with score_relevance. "
            "Write a summary with a verdict and 3 key points (use summarize for the verdict). "
            "Set overall confidence between 0 and 1."
        ),
        actions=True,
        lndl=True,
        response_format=ResearchReport,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        expect("has query", bool(getattr(out, "query", None)))
        expect("findings is list", isinstance(getattr(out, "findings", None), list))
        if isinstance(getattr(out, "findings", None), list):
            expect("at least 1 finding", len(out.findings) >= 1)
            if out.findings:
                f0 = out.findings[0]
                expect("finding is Finding instance", isinstance(f0, Finding))
        expect("has summary", isinstance(getattr(out, "summary", None), Summary))
        expect(
            "summary has key_points list",
            isinstance(getattr(getattr(out, "summary", None), "key_points", None), list),
        )
        expect("has action_responses", hasattr(out, "action_responses"))
        if hasattr(out, "action_responses"):
            expect(
                "executed multiple actions",
                len(out.action_responses or []) >= 2,
                f"got {len(out.action_responses or [])}",
            )


# ---------------------------------------------------------------------------
# Case 12 — Multi-step arithmetic (chain of tool calls)
# ---------------------------------------------------------------------------
async def case_12():
    banner("Case 12 — multi-step arithmetic (4 tools, chained)")

    class Step(BaseModel):
        description: str
        result: float

    class Calculation(BaseModel):
        problem: str
        steps: list[Step]
        answer: float

    branch = fresh_branch(
        system="You are a careful math tutor.",
        tools=[add, multiply, divide, subtract],
    )
    out = await branch.operate(
        instruction=(
            "Compute (3 + 4) * 5 - 6 / 2 step by step. "
            "Use add for 3+4, multiply for *5, divide for 6/2, subtract for the final. "
            "Show one Step per tool call with a description and the numeric result. "
            "Set 'problem' to the original expression and 'answer' to the final number."
        ),
        actions=True,
        lndl=True,
        response_format=Calculation,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        expect("has problem", bool(getattr(out, "problem", None)))
        expect("steps is list", isinstance(getattr(out, "steps", None), list))
        if isinstance(getattr(out, "steps", None), list):
            expect("at least 4 steps", len(out.steps) >= 4, f"got {len(out.steps)}")
        expect(
            "answer is correct (32.0)",
            getattr(out, "answer", None) == 32.0,
            f"got {getattr(out, 'answer', None)}",
        )


# ---------------------------------------------------------------------------
# Case 13 — Stock dashboard (per-symbol tool calls, no aggregation)
# ---------------------------------------------------------------------------
async def case_13():
    banner("Case 13 — stock dashboard (per-symbol tool calls)")

    class Quote(BaseModel):
        symbol: str
        price: float
        volume: int

    class Dashboard(BaseModel):
        as_of: str
        quotes: list[Quote]

    branch = fresh_branch(
        system="You build market dashboards using tools.",
        tools=[fetch_price, fetch_volume],
    )
    out = await branch.operate(
        instruction=(
            "Build a dashboard for AAPL, MSFT, GOOG. For each symbol: "
            "set its symbol, call fetch_price for price, call fetch_volume for volume. "
            "Set as_of='2026-05-07'."
        ),
        actions=True,
        lndl=True,
        response_format=Dashboard,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        expect("has as_of", bool(getattr(out, "as_of", None)))
        expect("quotes is list", isinstance(getattr(out, "quotes", None), list))
        if isinstance(getattr(out, "quotes", None), list):
            expect("3 quotes", len(out.quotes) == 3, f"got {len(out.quotes)}")
            if out.quotes:
                q0 = out.quotes[0]
                expect("quote is Quote instance", isinstance(q0, Quote))
                expect("quote price > 0", q0.price > 0)
                expect("quote volume > 0", q0.volume > 0)


# ---------------------------------------------------------------------------
# Case 14 — Heterogeneous schema (nested + list[str] inside model + optional)
# ---------------------------------------------------------------------------
async def case_14():
    banner("Case 14 — heterogeneous schema (nested + nested list[str] + optional)")

    class Section(BaseModel):
        heading: str
        body: str
        bullets: list[str] | None = None

    class Article(BaseModel):
        title: str
        sections: list[Section]
        word_count: int
        tags: list[str]

    branch = fresh_branch(system="You write structured articles.")
    out = await branch.operate(
        instruction=(
            "Write a 3-section article about 'Why unit tests matter'. "
            "Each section has a heading, a body paragraph, and a bullets list with 2-3 items. "
            "Set title, word_count to a reasonable estimate, "
            "and tags to ['testing', 'quality', 'engineering']."
        ),
        lndl=True,
        response_format=Article,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        expect("title set", bool(getattr(out, "title", None)))
        expect("sections is list", isinstance(getattr(out, "sections", None), list))
        if isinstance(getattr(out, "sections", None), list):
            expect("3 sections", len(out.sections) == 3, f"got {len(out.sections)}")
            if out.sections:
                s0 = out.sections[0]
                expect("section is Section instance", isinstance(s0, Section))
                expect("section.bullets is list", isinstance(s0.bullets, list))
        expect("word_count is int", isinstance(getattr(out, "word_count", None), int))
        expect("tags has 3 items", len(getattr(out, "tags", [])) == 3)


# ---------------------------------------------------------------------------
# Case 15 — retries handle malformed LNDL
#   We deliberately give the model a confusing instruction that often produces
#   ill-formed output, then let lndl_retries=2 recover.
# ---------------------------------------------------------------------------
async def case_15():
    banner("Case 15 — lndl_retries recovers from malformed output")

    class Plan(BaseModel):
        title: str
        steps: list[str]
        priority: int

    branch = fresh_branch(
        system="Respond using LNDL. If your output is malformed, fix it on retry."
    )
    out = await branch.operate(
        instruction=(
            "Make a Plan with title='Migrate to Postgres', "
            "3 steps (back up DB, run migration, smoke test), priority=1."
        ),
        lndl=True,
        lndl_retries=2,
        response_format=Plan,
    )
    show(out, branch)
    is_model = not isinstance(out, dict)
    expect("returned a model (not dict)", is_model)
    if is_model:
        expect("has title", bool(getattr(out, "title", None)))
        expect("steps is list with 3 items", len(getattr(out, "steps", [])) == 3)
        expect("priority is int", isinstance(getattr(out, "priority", None), int))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
CASES = [
    ("1", case_1),
    ("2", case_2),
    ("3", case_3),
    ("4", case_4),
    ("5", case_5),
    ("6", case_6),
    ("7", case_7),
    ("8", case_8),
    ("9", case_9),
    ("10", case_10),
    ("11", case_11),
    ("12", case_12),
    ("13", case_13),
    ("14", case_14),
    ("15", case_15),
]


async def main():
    for label, fn in CASES:
        if ONLY and label != ONLY:
            continue
        try:
            await fn()
        except Exception as e:
            print(f"  [ERROR] case {label}: {e!r}")
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
