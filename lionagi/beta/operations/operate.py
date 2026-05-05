# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operate: top-level agent operation chain.

Handler signature: operate(params, ctx) -> validated model instance

Full pipeline:
  1. Compose request structure from operable (inject action spec if needed)
  2. Structure: generate -> parse -> validate (produces typed model)
  3. Act: extract and execute action_requests, persist messages
  4. Compose response structure, merge action_results, validate

Action spec injection:
  If invoke_actions=True AND tool_schemas are present AND the branch
  has "action" in its capabilities, the action_requests spec is injected
  into the operable before composing the request structure. This lets
  the LLM produce structured tool calls alongside regular output fields.

Runtime spec injection is intentionally narrow: only the action spec is
injected automatically based on explicit capability declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from lionagi.beta.rules import Validator
from lionagi.ln.types import ModelConfig, Params
from lionagi.ln.types._sentinel import MaybeUnset, Unset
from lionagi.protocols.messages import Message
from lionagi.beta.resource.toolkit import ToolKit

from .act import ActParams
from .generate import GenerateParams
from .round_outcome import Continue, Failed, Retry, RoundOutcome, Success
from .specs import Action, ActionResult, get_action_result_spec, get_action_spec
from .structure import StructureParams

if TYPE_CHECKING:
    from lionagi.beta.core.message.common import CustomParser
    from lionagi.beta.resource.imodel import iModel
    from lionagi.beta.session.context import RequestContext
    from lionagi.ln.types import Operable

__all__ = ("OperateParams", "operate")


def _extract_lvars_from_text(
    text: str,
) -> tuple[dict, dict]:
    """Partial-parse LNDL text — extract lvars/lacts even when OUT{} is absent.

    Returns (lvars, lacts) keyed by alias. Used by the no-tool LNDL continuation
    loop to find `note.X` commits in a partial response.
    """
    from lionagi.beta.lndl.ast import Lvar
    from lionagi.beta.lndl.fuzzy import normalize_lndl_text
    from lionagi.beta.lndl.lexer import Lexer
    from lionagi.beta.lndl.parser import Parser
    from lionagi.beta.lndl.types import LactMetadata, LvarMetadata, RLvarMetadata

    text = normalize_lndl_text(text)
    try:
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        program = parser.parse()
    except Exception:
        return {}, {}

    lvars: dict = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            lvars[lvar.alias] = LvarMetadata(
                model=lvar.model,
                field=lvar.field,
                local_name=lvar.alias,
                value=lvar.content,
            )
        else:
            lvars[lvar.alias] = RLvarMetadata(
                local_name=lvar.alias,
                value=lvar.content,
            )

    lacts: dict = {}
    for lact in program.lacts:
        lacts[lact.alias] = LactMetadata(
            model=lact.model,
            field=lact.field,
            local_name=lact.alias,
            call=lact.call,
        )

    return lvars, lacts


@dataclass(frozen=True, slots=True)
class OperateParams(Params):
    """Parameters for the operate pipeline; StructureParams is built internally after runtime spec composition."""

    _config = ModelConfig(sentinel_additions=frozenset({"none", "empty"}))

    generate_params: GenerateParams
    operable: Operable
    validator: Validator = field(default_factory=Validator)

    capabilities: set[str] | None = None
    persist: bool = True

    invoke_actions: bool = False
    action_strategy: Literal["sequential", "concurrent"] = "concurrent"
    max_concurrent: int | None = None
    throttle_period: float | None = None

    auto_fix: bool = True
    strict: bool = True

    parse_imodel: MaybeUnset[iModel | str] = Unset
    parse_imodel_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_parser: CustomParser | None = None
    similarity_threshold: float = 0.85
    max_retries: int = 3
    fill_mapping: dict[str, Any] | None = None
    fill_value: Any = Unset

    max_lndl_rounds: int = 3

    toolkits: list[ToolKit] | None = None


async def operate(params: OperateParams, ctx: RequestContext) -> Any:
    """Operate handler: compose -> structure -> act -> merge.

    Returns a validated model instance. If actions were invoked,
    the response structure includes action_results.

    When structure_format is LNDL and the model produces lvars but no OUT{}
    block, the lvars are persisted to branch.scratchpad and a continuation
    round is requested (up to max_lndl_rounds times).
    """
    session = await ctx.get_session()
    branch = await ctx.get_branch()

    operable = params.operable
    gen_params = params.generate_params

    # LNDL uses <lact> tags for tool calls; JSON action_requests spec must NOT be injected for lndl.
    has_tools = not gen_params.is_sentinel_field("tool_schemas")
    branch_caps = getattr(branch, "capabilities", set())
    is_lndl = gen_params.structure_format == "lndl"
    inject_actions = params.invoke_actions and has_tools and "action" in branch_caps and not is_lndl

    request_operable = operable.extend([get_action_spec()]) if inject_actions else operable
    request_structure = request_operable.compose_structure()
    use_gen_params = gen_params.with_updates(
        copy_containers="deep", request_model=request_structure
    )

    if is_lndl and params.invoke_actions and has_tools:
        # LNDL with tools: generate → extract → execute lacts → resolve with results
        structured = await _lndl_tool_operate(
            operable=request_operable,
            structure=request_structure,
            gen_params=use_gen_params,
            validator=params.validator,
            session=session,
            branch=branch,
            ctx=ctx,
            max_rounds=params.max_lndl_rounds,
            persist=params.persist,
            toolkits=params.toolkits,
        )
        return structured

    structure_params = StructureParams(
        generate_params=use_gen_params,
        validator=params.validator,
        operable=request_operable,
        structure=request_structure,
        persist=params.persist,
        capabilities=params.capabilities,
        auto_fix=params.auto_fix,
        strict=params.strict,
        parse_imodel=params.parse_imodel,
        parse_imodel_kwargs=params.parse_imodel_kwargs,
        custom_parser=params.custom_parser,
        similarity_threshold=params.similarity_threshold,
        max_retries=params.max_retries,
        fill_mapping=params.fill_mapping,
        fill_value=params.fill_value,
    )

    structured = await _structure_with_lndl_continuation(
        structure_params=structure_params,
        ctx=ctx,
        session=session,
        branch=branch,
        use_gen_params=use_gen_params,
        max_lndl_rounds=params.max_lndl_rounds,
    )

    if not inject_actions:
        return structured

    act_requests = getattr(structured, "action_requests", None)
    if not act_requests:
        return structured

    action_messages = _actions_to_messages(act_requests)
    if not action_messages:
        return structured

    for msg in action_messages:
        session.add_message(msg, branches=branch)

    act_params = ActParams(
        action_requests=action_messages,
        strategy=params.action_strategy,
        max_concurrent=params.max_concurrent,
        throttle_period=params.throttle_period,
        toolkits=params.toolkits,
    )
    action_responses = await ctx.conduct("act", act_params)

    for resp in action_responses:
        resp_msg = Message(content=resp)
        session.add_message(resp_msg, branches=branch)

    action_results = _responses_to_results(action_responses, action_messages)

    # No re-validation: structured output was validated in stage 2; action_results are execution artifacts.
    response_operable = request_operable.extend([get_action_result_spec()])
    response_structure = response_operable.compose_structure()

    data = request_operable.dump_instance(structured)
    data["action_results"] = action_results

    return response_structure(**data)


async def _lndl_tool_operate(
    operable: Any,
    structure: type,
    gen_params: GenerateParams,
    validator: Validator,
    session: Any,
    branch: Any,
    ctx: Any,
    max_rounds: int = 3,
    persist: bool = True,
    toolkits: list[ToolKit] | None = None,
) -> Any:
    """LNDL operate with tools — react loop returning a validated model.

    Each round produces a `RoundOutcome`; the loop matches on it:
      - Success(out)  → return
      - Continue(_)   → next round (note.X commits already in scratchpad)
      - Retry(err, _) → next round, feed error to model
      - Failed(err)   → re-raise
      - Exhausted(_)  → raise MissingOutBlockError

    Per-round work lives in `_run_lndl_round`. The 4 ad-hoc error paths from
    the prior implementation collapse into Retry / Failed / Exhausted variants.
    """
    from lionagi.beta.lndl.errors import MissingOutBlockError

    last_error: str | None = None
    last_outcome: RoundOutcome | None = None

    for round_num in range(max_rounds):
        outcome = await _run_lndl_round(
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

        if isinstance(outcome, Success):
            return outcome.output
        if isinstance(outcome, Failed):
            raise outcome.error
        if isinstance(outcome, Retry):
            last_error = outcome.error
            continue
        if isinstance(outcome, Continue):
            last_error = None
            continue
        # Exhausted shouldn't be returned mid-loop; treated as terminal below.
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


async def _run_lndl_round(
    round_num: int,
    operable: Any,
    structure: type,
    gen_params: GenerateParams,
    validator: Validator,
    session: Any,
    branch: Any,
    ctx: Any,
    last_error: str | None,
    persist: bool,
    is_last_round: bool,
    max_rounds: int = 3,
    toolkits: list[ToolKit] | None = None,
) -> RoundOutcome:
    """Run a single LNDL round and classify its outcome.

    Steps: generate → parse → identify note commits → execute lacts →
    promote results → commit notes → resolve OUT{} (if present).
    """
    from lionagi.beta.lndl.ast import Lvar
    from lionagi.beta.lndl.fuzzy import parse_lndl_fuzzy
    from lionagi.beta.lndl.lexer import Lexer
    from lionagi.beta.lndl.parser import Parser
    from lionagi.beta.lndl.resolver import resolve_references_prefixed
    from lionagi.beta.lndl.types import LactMetadata, LvarMetadata, RLvarMetadata

    from .utils import ReturnAs

    if round_num == 0:
        round_gen = gen_params
    else:
        continuation = _build_lndl_continuation(
            branch, round_num, last_error=last_error, max_rounds=max_rounds
        )
        round_gen = gen_params.with_updates(copy_containers="deep", primary=continuation)

    if persist:
        text = await _generate_and_persist_text(round_gen, ctx)
    else:
        text_gen = round_gen.with_updates(copy_containers="deep", return_as=ReturnAs.TEXT)
        text = await ctx.conduct("generate", text_gen)

    from lionagi.beta.lndl.fuzzy import normalize_lndl_text

    text = normalize_lndl_text(text)
    try:
        lexer = Lexer(text)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=text)
        program = parser.parse()
    except Exception as strict_err:
        if is_last_round:
            try:
                output = parse_lndl_fuzzy(text, operable, scratchpad=branch.scratchpad)
                validated = await validator.validate(output.fields, operable, structure=structure)
                return Success(validated)
            except Exception:
                return Failed(strict_err)
        return Retry(
            error=(
                "Invalid LNDL syntax. Use <lvar alias>value</lvar> or "
                "<lvar Model.field alias>value</lvar>. "
                'Do NOT use XML attributes (name="...", type="...").'
            )
        )

    lvars: dict[str, LvarMetadata | RLvarMetadata] = {}
    lacts: dict[str, LactMetadata] = {}
    for lvar in program.lvars:
        if isinstance(lvar, Lvar):
            lvars[lvar.alias] = LvarMetadata(
                model=lvar.model, field=lvar.field, local_name=lvar.alias, value=lvar.content
            )
        else:
            lvars[lvar.alias] = RLvarMetadata(local_name=lvar.alias, value=lvar.content)
    for lact in program.lacts:
        lacts[lact.alias] = LactMetadata(
            model=lact.model, field=lact.field, local_name=lact.alias, call=lact.call
        )

    has_out = program.out_block is not None

    # Only lacts referenced in OUT{} execute; everything outside OUT{} is scratch.
    # Scratchpad persistence is via the scratchpad tool, not syntax-level note.X declarations.
    out_refs: set[str] = set()
    if has_out:
        for v in program.out_block.fields.values():
            if isinstance(v, list):
                out_refs.update(v)
    lacts_to_execute = {a: meta for a, meta in lacts.items() if a in out_refs}
    lact_results: dict[str, Any] = {}
    if lacts_to_execute:
        lact_results = await _execute_lacts_to_results(
            lacts_to_execute, session, branch, ctx, toolkits=toolkits
        )

    for alias, result in lact_results.items():
        lvars[alias] = RLvarMetadata(local_name=alias, value=_result_to_str(result))
    for alias in lact_results:
        lacts.pop(alias, None)

    if not has_out:
        return Continue(notes_committed=())

    try:
        output = resolve_references_prefixed(
            program.out_block.fields,
            lvars,
            lacts,
            operable,
        )
        from pydantic import BaseModel

        result = {
            name: value.model_dump() if isinstance(value, BaseModel) else value
            for name, value in output.fields.items()
        }
        return Success(structure.model_validate(result))
    except Exception as resolve_err:
        return Retry(
            error=f"OUT{{}} resolution failed: {resolve_err}",
            note_keys=(),
        )


def _result_to_str(result: Any) -> str:
    """Convert a tool result to a string for embedding in synthetic lvars."""
    if isinstance(result, str):
        return result
    if isinstance(result, (dict, list)):
        from lionagi.libs.schema import minimal_yaml

        return minimal_yaml(result)
    return str(result)


async def _execute_lacts_to_results(
    lacts: dict[str, Any],
    session: Any,
    branch: Any,
    ctx: Any,
    toolkits: list[ToolKit] | None = None,
) -> dict[str, Any]:
    """Execute lact tool calls via act() and return {alias: result} dict.

    parse_function_call splits "code.count_lines(...)" into tool="count_lines".
    act() handles ToolKit name coercion via the toolkits parameter.
    """
    from lionagi.beta.core.message import ActionRequest
    from lionagi.beta.lndl.types import LactMetadata
    from lionagi.libs.parse import parse_function_call

    action_messages: list[Message] = []
    alias_to_msg_id: dict[str, str] = {}

    for alias, lact_meta in lacts.items():
        if not isinstance(lact_meta, LactMetadata):
            continue
        parsed = parse_function_call(lact_meta.call)
        function_name = parsed["tool"]
        arguments = parsed.get("arguments", {})
        req = ActionRequest.create(function=function_name, arguments=arguments)
        msg = Message(content=req)
        action_messages.append(msg)
        alias_to_msg_id[str(msg.id)] = alias
        session.add_message(msg, branches=branch)

    if not action_messages:
        return {}

    act_params = ActParams(
        action_requests=action_messages,
        toolkits=toolkits,
    )
    responses = await ctx.conduct("act", act_params)

    results: dict[str, Any] = {}
    for resp in responses:
        session.add_message(Message(content=resp), branches=branch)
        alias = alias_to_msg_id.get(resp.request_id, "")
        if alias:
            results[alias] = resp.result if resp.success else f"<tool_error: {resp.error}>"
    return results


async def _generate_and_persist_text(gen_params: GenerateParams, ctx: Any) -> str:
    """Generate as MESSAGE, persist, return text."""
    from .utils import ReturnAs

    msg = await ctx.conduct(
        "generate",
        gen_params.with_updates(copy_containers="deep", return_as=ReturnAs.MESSAGE),
    )
    session = await ctx.get_session()
    branch = await ctx.get_branch()
    session.add_message(msg, branches=branch)
    return msg.content.response


def _build_lndl_continuation(
    branch: Any, round_num: int, last_error: str | None = None, max_rounds: int = 3
) -> str:
    """Build the continuation prompt for the next LNDL round.

    Prior assistant text and tool results already live in chat history.
    We add: round number, error guidance, scratchpad state, and remaining
    round budget so the model knows when to synthesize.
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


async def _structure_with_lndl_continuation(
    structure_params: StructureParams,
    ctx: RequestContext,
    session: Any,
    branch: Any,
    use_gen_params: GenerateParams,
    max_lndl_rounds: int,
) -> Any:
    """Run structure with retry on MissingOutBlockError for LNDL.

    If the model produced lvars/lacts but no OUT{} block, we send a
    continuation prompt telling it to commit. No magic note.X scratchpad
    injection — the model's own tool calls handle persistence.
    """
    from lionagi.beta.lndl.errors import MissingOutBlockError

    for round_num in range(max_lndl_rounds + 1):
        try:
            return await ctx.conduct("structure", structure_params)
        except MissingOutBlockError:
            if round_num >= max_lndl_rounds:
                raise

            response_text = _get_last_assistant_text(session, branch)
            if response_text is None or "<lvar" not in response_text:
                raise

            lvars, _lacts = _extract_lvars_from_text(response_text)
            if not lvars:
                raise

            continuation_primary = _build_lndl_continuation(
                branch, round_num, max_rounds=max_lndl_rounds
            )
            continuation_gen_params = use_gen_params.with_updates(
                copy_containers="deep",
                primary=continuation_primary,
                instruction=Unset,
            )
            structure_params = structure_params.__class__(
                generate_params=continuation_gen_params,
                validator=structure_params.validator,
                operable=structure_params.operable,
                structure=structure_params.structure,
                persist=structure_params.persist,
                capabilities=structure_params.capabilities,
                auto_fix=structure_params.auto_fix,
                strict=structure_params.strict,
                parse_imodel=structure_params.parse_imodel,
                parse_imodel_kwargs=structure_params.parse_imodel_kwargs,
                custom_parser=structure_params.custom_parser,
                similarity_threshold=structure_params.similarity_threshold,
                max_retries=structure_params.max_retries,
                fill_mapping=structure_params.fill_mapping,
                fill_value=structure_params.fill_value,
            )

    raise MissingOutBlockError("LNDL continuation loop exhausted without result")


def _get_last_assistant_text(session: Any, branch: Any) -> str | None:
    """Get the response text from the last assistant message in the branch."""
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
        if response is not None and isinstance(response, str):
            return response
    return None


def _actions_to_messages(act_requests: list) -> list[Message]:
    from lionagi.beta.core.message import ActionRequest

    messages: list[Message] = []
    for req in act_requests:
        if isinstance(req, Action):
            content = ActionRequest.create(function=req.function, arguments=req.arguments)
            messages.append(Message(content=content))
        elif isinstance(req, dict):
            content = ActionRequest.create(
                function=req.get("function", ""),
                arguments=req.get("arguments", {}),
            )
            messages.append(Message(content=content))
    return messages


def _responses_to_results(
    action_responses: list,
    action_messages: list[Message],
) -> list[ActionResult]:
    from lionagi.beta.core.message import ActionResponse

    id_to_func: dict[str, str] = {}
    for msg in action_messages:
        content = msg.content
        if hasattr(content, "function"):
            id_to_func[str(msg.id)] = content.function

    results: list[ActionResult] = []
    for resp in action_responses:
        if isinstance(resp, ActionResponse):
            func = id_to_func.get(
                resp.request_id if not resp._is_sentinel(resp.request_id) else "",
                "",
            )
            results.append(
                ActionResult(
                    function=func,
                    result=resp.result if resp.success else None,
                    error=resp.error if not resp.success else None,
                )
            )
        elif isinstance(resp, dict):
            results.append(ActionResult.model_validate(resp))
    return results
