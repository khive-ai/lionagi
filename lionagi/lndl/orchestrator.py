# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""LNDL orchestration: multi-round drive loops on top of generate / structure / act.

Two entrypoints, one for each operate mode:

- `run_with_tools`: react-style loop where the model emits `<lvar>` / `<lact>`
  and an `OUT{}` block, lacts referenced in OUT{} are executed via the `act`
  operation, results substituted, then OUT{} resolves into a validated model.
  Bypasses the structure pipeline so we can inspect lacts before resolution.

- `run_continuation`: thin wrapper around `ctx.conduct("structure", ...)`
  that catches `MissingOutBlockError` and drives continuation rounds, used
  when the model produced narrative + lvars but never committed an OUT{}.

Both share continuation-prompt construction and last-assistant-text readout.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from lionagi.service.toolkit import ToolKit
from lionagi.rules import Validator
from lionagi.ln.types._sentinel import Unset
from lionagi.protocols.messages import Message
from lionagi.protocols.messages.action_request import (
    ActionRequestContent as ActionRequest,
)

from .ast import Lvar
from .errors import MissingOutBlockError
from .fuzzy import normalize_lndl_text, parse_lndl_fuzzy
from .lexer import Lexer
from .parser import Parser
from .resolver import resolve_references_prefixed
from .types import LactMetadata, LvarMetadata, RLvarMetadata

if TYPE_CHECKING:
    from lionagi.operations.generate import GenerateParams
    from lionagi.operations.round_outcome import RoundOutcome
    from lionagi.operations.structure import StructureParams
    from lionagi.session.context import RequestContext
    from lionagi.ln.types import Operable

__all__ = (
    "build_continuation_prompt",
    "extract_lvars",
    "run_continuation",
    "run_with_tools",
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def extract_lvars(
    text: str,
) -> tuple[dict[str, LvarMetadata | RLvarMetadata], dict[str, LactMetadata]]:
    """Best-effort partial parse: pull lvars/lacts from text even without OUT{}.

    Returns ({alias: LvarMetadata|RLvarMetadata}, {alias: LactMetadata}).
    Returns empty dicts if the text is not parseable as LNDL.
    """
    text = normalize_lndl_text(text)
    try:
        program = Parser(Lexer(text).tokenize(), source_text=text).parse()
    except Exception:
        return {}, {}
    return _program_to_metadata(program)


def build_continuation_prompt(
    round_num: int,
    *,
    max_rounds: int,
    last_error: str | None = None,
) -> str:
    """Continuation message for the next LNDL round.

    Tells the model how many rounds remain and feeds back any prior parse or
    resolve error so it can self-correct. Prior assistant text and tool
    results already live in chat history; this only adds round metadata.
    """
    remaining = max_rounds - round_num - 1
    parts = [f"Round {round_num + 1} of {max_rounds} ({remaining} remaining)."]
    if last_error:
        parts.append(f"Previous round failed: {last_error}")
    elif remaining <= 1:
        parts.append("FINAL ROUND — produce your OUT{} block now with what you have.")
    elif remaining <= 2:
        parts.append("Running low on rounds. Synthesize soon and produce OUT{}.")
    else:
        parts.append("Continue.")
    parts.append("Produce an OUT{} block when ready, using <lvar>/<lact> as needed.")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# run_with_tools — react loop
# ---------------------------------------------------------------------------


async def run_with_tools(
    *,
    operable: Operable,
    structure: type[BaseModel],
    gen_params: GenerateParams,
    validator: Validator,
    session: Any,
    branch: Any,
    ctx: RequestContext,
    max_rounds: int = 3,
    persist: bool = True,
    toolkits: list[ToolKit] | None = None,
) -> Any:
    """LNDL react loop: generate → parse → execute lacts → resolve OUT{}.

    Each round returns a RoundOutcome the loop matches on. Outcomes:
      Success(out)  → return out
      Continue(_)   → next round (no OUT{} yet)
      Retry(err, _) → next round, feed error to model
      Failed(err)   → re-raise
    Loop exhaustion raises MissingOutBlockError with the last error attached.
    """
    last_error: str | None = None
    last_outcome: RoundOutcome | None = None

    for round_num in range(max_rounds):
        outcome = await _run_round(
            round_num=round_num,
            operable=operable,
            structure=structure,
            gen_params=gen_params,
            validator=validator,
            session=session,
            branch=branch,
            ctx=ctx,
            last_error=last_error,
            persist=persist,
            is_last_round=(round_num == max_rounds - 1),
            max_rounds=max_rounds,
            toolkits=toolkits,
        )
        last_outcome = outcome
        match outcome:
            case Success(output):  # type: ignore[misc]
                return output
            case Failed(err):  # type: ignore[misc]
                raise err
            case Retry(err, _):  # type: ignore[misc]
                last_error = err
            case Continue(_):  # type: ignore[misc]
                last_error = None
            case _:
                break

    final_err = (
        getattr(last_outcome, "error", None)
        or getattr(last_outcome, "last_error", None)
        or last_error
    )
    suffix = f". Last error: {final_err}" if final_err else ""
    raise MissingOutBlockError(
        f"LNDL operate exhausted {max_rounds} rounds without OUT{{}}{suffix}"
    )


async def _run_round(
    *,
    round_num: int,
    operable: Operable,
    structure: type[BaseModel],
    gen_params: GenerateParams,
    validator: Validator,
    session: Any,
    branch: Any,
    ctx: Any,
    last_error: str | None,
    persist: bool,
    is_last_round: bool,
    max_rounds: int,
    toolkits: list[ToolKit] | None,
) -> RoundOutcome:
    """One round: generate → parse text → execute referenced lacts → resolve OUT{}."""
    if round_num == 0:
        round_gen = gen_params
    else:
        prompt = build_continuation_prompt(
            round_num, max_rounds=max_rounds, last_error=last_error
        )
        round_gen = gen_params.with_updates(copy_containers="deep", primary=prompt)

    text = await _generate_text(round_gen, ctx, persist=persist)
    text = normalize_lndl_text(text)

    try:
        program = Parser(Lexer(text).tokenize(), source_text=text).parse()
    except Exception as parse_err:
        if not is_last_round:
            return Retry(
                error=(
                    "Invalid LNDL syntax. Use <lvar alias>value</lvar> or "
                    "<lvar Model.field alias>value</lvar>. "
                    'Do NOT use XML attributes (name="...", type="...").'
                )
            )
        # Last round: try fuzzy parse before giving up.
        try:
            output = parse_lndl_fuzzy(text, operable, scratchpad=branch.scratchpad)
            validated = await validator.validate(
                output.fields, operable, structure=structure
            )
            return Success(validated)
        except Exception:
            return Failed(parse_err)

    lvars, lacts = _program_to_metadata(program)
    has_out = program.out_block is not None

    out_refs: set[str] = set()
    if has_out:
        for v in program.out_block.fields.values():
            if isinstance(v, list):
                out_refs.update(v)

    lacts_to_execute = {a: m for a, m in lacts.items() if a in out_refs}
    if lacts_to_execute:
        results = await _execute_lacts(
            lacts_to_execute, session, branch, ctx, toolkits=toolkits
        )
        for alias, result in results.items():
            lvars[alias] = RLvarMetadata(local_name=alias, value=_to_str(result))
            lacts.pop(alias, None)

    if not has_out:
        return Continue(notes_committed=())

    try:
        output = resolve_references_prefixed(
            program.out_block.fields, lvars, lacts, operable
        )
        result = {
            name: value.model_dump() if isinstance(value, BaseModel) else value
            for name, value in output.fields.items()
        }
        return Success(structure.model_validate(result))
    except Exception as resolve_err:
        return Retry(error=f"OUT{{}} resolution failed: {resolve_err}")


def _program_to_metadata(
    program: Any,
) -> tuple[dict[str, LvarMetadata | RLvarMetadata], dict[str, LactMetadata]]:
    """Pull lvar / lact metadata dicts off a parsed Program."""
    lvars: dict[str, LvarMetadata | RLvarMetadata] = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            lvars[lvar.alias] = LvarMetadata(
                model=lvar.model,
                field=lvar.field,
                local_name=lvar.alias,
                value=lvar.content,
            )
        else:
            lvars[lvar.alias] = RLvarMetadata(local_name=lvar.alias, value=lvar.content)
    lacts = {
        lact.alias: LactMetadata(
            model=lact.model,
            field=lact.field,
            local_name=lact.alias,
            call=lact.call,
        )
        for lact in program.lacts
    }
    return lvars, lacts


# ---------------------------------------------------------------------------
# run_continuation — wrap structure.conduct with retry
# ---------------------------------------------------------------------------


async def run_continuation(
    *,
    structure_params: StructureParams,
    ctx: RequestContext,
    session: Any,
    branch: Any,
    use_gen_params: GenerateParams,
    max_rounds: int,
) -> Any:
    """Run structure() with continuation rounds on MissingOutBlockError.

    When the model emits lvars + narrative but no OUT{} block, parse raises
    MissingOutBlockError. We catch it, build a continuation prompt, and
    retry up to max_rounds times.
    """
    current = structure_params
    for round_num in range(max_rounds + 1):
        try:
            return await ctx.conduct("structure", current)
        except MissingOutBlockError:
            if round_num >= max_rounds:
                raise

            response_text = _last_assistant_text(session, branch)
            if response_text is None or "<lvar" not in response_text:
                raise

            lvars, _ = extract_lvars(response_text)
            if not lvars:
                raise

            prompt = build_continuation_prompt(round_num, max_rounds=max_rounds)
            new_gen = use_gen_params.with_updates(
                copy_containers="deep", primary=prompt, instruction=Unset
            )
            current = replace(current, generate_params=new_gen)

    raise MissingOutBlockError("LNDL continuation loop exhausted without result")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _generate_text(
    gen_params: GenerateParams,
    ctx: Any,
    *,
    persist: bool,
) -> str:
    """Generate one response, persist as message if requested, return text."""
    from lionagi.operations.utils import ReturnAs

    if persist:
        msg = await ctx.conduct(
            "generate",
            gen_params.with_updates(
                copy_containers="deep", return_as=ReturnAs.MESSAGE
            ),
        )
        session = await ctx.get_session()
        branch = await ctx.get_branch()
        session.add_message(msg, branches=branch)
        return msg.content.response

    return await ctx.conduct(
        "generate",
        gen_params.with_updates(copy_containers="deep", return_as=ReturnAs.TEXT),
    )


async def _execute_lacts(
    lacts: dict[str, LactMetadata],
    session: Any,
    branch: Any,
    ctx: Any,
    *,
    toolkits: list[ToolKit] | None,
) -> dict[str, Any]:
    """Execute the lacts referenced from OUT{} via the act operation.

    Returns {alias: tool_result_or_error_marker}. Tool errors come back as
    "<tool_error: ...>" strings so the resolver can still substitute them
    into RLvar values without aborting OUT{} resolution.
    """
    from lionagi.operations.act import ActParams
    from lionagi.libs.parse import parse_function_call

    action_messages: list[Message] = []
    alias_by_msg_id: dict[str, str] = {}

    for alias, meta in lacts.items():
        parsed = parse_function_call(meta.call)
        req = ActionRequest.create(
            function=parsed["tool"], arguments=parsed.get("arguments", {})
        )
        msg = Message(content=req)
        action_messages.append(msg)
        alias_by_msg_id[str(msg.id)] = alias
        session.add_message(msg, branches=branch)

    if not action_messages:
        return {}

    responses = await ctx.conduct(
        "act", ActParams(action_requests=action_messages, toolkits=toolkits)
    )

    results: dict[str, Any] = {}
    for resp in responses:
        session.add_message(Message(content=resp), branches=branch)
        alias = alias_by_msg_id.get(resp.request_id, "")
        if alias:
            results[alias] = (
                resp.result if resp.success else f"<tool_error: {resp.error}>"
            )
    return results


def _to_str(value: Any) -> str:
    """Stringify a tool result for embedding in a synthetic RLvar value."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        from lionagi.libs.schema import minimal_yaml

        return minimal_yaml(value)
    return str(value)


def _last_assistant_text(session: Any, branch: Any) -> str | None:
    """Find the most recent assistant response text in this branch, if any."""
    if branch is None or not branch.order:
        return None
    for msg_id in reversed(branch.order):
        msg = session.messages.get(msg_id, None)
        if msg is None:
            continue
        content = getattr(msg, "content", None)
        if content is None:
            continue
        response = getattr(content, "response", None)
        if isinstance(response, str):
            return response
    return None


# Late imports to keep RoundOutcome resolution lazy (operate.py imports us).
from lionagi.operations.round_outcome import (  # noqa: E402
    Continue,
    Failed,
    Retry,
    Success,
)
