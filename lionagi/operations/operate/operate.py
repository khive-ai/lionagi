# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import TYPE_CHECKING, Any, Literal, Union

from pydantic import BaseModel, JsonValue

from lionagi.ln import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import Spec
from lionagi.models import FieldModel
from lionagi.protocols.generic import Progression
from lionagi.protocols.messages import (
    Instruction,
    JsonFormatter,
    LndlFormatter,
    SenderRecipient,
)

from ..fields import Instruct
from ..types import (
    ActionParam,
    ChatParam,
    HandleValidation,
    Middle,
    ParseParam,
    RunParam,
)

if TYPE_CHECKING:
    from lionagi.service.imodel import iModel
    from lionagi.session.branch import Branch, ToolRef

    from .operative import Operative


def _get_raw_assistant_response(branch: "Branch", result: Any) -> str | None:
    """Extract the raw LNDL assistant response from the result or message log."""
    if isinstance(result, str):
        return result
    raw = None
    for m in branch.messages:
        if hasattr(m.content, "assistant_response"):
            raw = m.content.assistant_response
    return raw


_LNDL_REPARSE_PROMPT = (
    "Your previous LNDL output had an error. Fix it and respond with ONLY valid LNDL.\n\n"
    "Error: {error}\n\n"
    "Original output:\n{original}\n\n"
    "Rules:\n"
    "- OUT{{}} spec names must exactly match the schema's field names\n"
    "- Every alias in OUT{{}} must have a matching <lvar> or <lact> declaration\n"
    "- Tag attributes are SPACE-separated identifiers, the body sits between > and </tag>\n"
    "- Tool arguments must be literal values, not alias references\n"
    "Respond with corrected LNDL only — no explanation."
)


async def _request_lndl_fix(
    branch: "Branch",
    original: str,
    error: str,
    *,
    chat_param: ChatParam | None = None,
    middle: Middle | None = None,
) -> str | None:
    """Ask the model to fix malformed LNDL using the existing branch context.

    Routes through the same ``middle`` (communicate / run_and_collect) that
    the outer operate call selected, so CLI/streaming paths stay consistent.
    """
    prompt = _LNDL_REPARSE_PROMPT.format(error=error, original=original[:2000])
    try:
        if middle is not None and chat_param is not None:
            raw = await middle(
                branch, prompt, chat_param, None, False, skip_validation=True
            )
            return raw if isinstance(raw, str) else (str(raw) if raw else None)
        # Fallback when middle not available (shouldn't happen in practice)
        _, resp = await branch.chat(
            instruction=prompt, return_ins_res_message=True
        )
        return resp.response
    except Exception:
        return None


def _build_lndl_continuation(
    round_num: int,
    max_rounds: int,
    last_error: str | None = None,
) -> str:
    """Build the user-side continuation message that drives the NEXT round.

    ``round_num`` is the index of the round that just completed. The
    continuation header tells the model which round it is about to run
    (1-indexed) and how many remain. Tool results from prior rounds already
    live in chat history, so we don't echo them.
    """
    next_round = round_num + 2  # 1-indexed, next round about to be driven
    remaining = max_rounds - next_round
    parts = [f"Round {next_round} of {max_rounds} ({remaining} remaining)."]
    if last_error:
        parts.append(f"Previous round failed: {last_error}")
    if remaining <= 0:
        parts.append(
            "FINAL ROUND. You MUST produce a complete OUT{} block this round. "
            "Use everything you've learned from prior tool results. Do not "
            "issue more tool calls — synthesize and commit now."
        )
    elif remaining == 1:
        parts.append(
            "Running low on rounds. Synthesize what you have and produce OUT{}."
        )
    elif not last_error:
        parts.append(
            "Continue. Read any tool results above, declare more <lvar>/<lact> "
            "as needed, and produce OUT{} when ready."
        )
    return "\n\n".join(parts)


async def _execute_program_lacts(
    branch: "Branch",
    program,
    action_param: "ActionParam | None",
) -> tuple[dict[str, Any], list[str]]:
    """Execute every <lact> in the program — regardless of OUT membership.

    Returns ``({alias: result_value}, [parse_error_messages])``.
    ActionRequest/ActionResponse messages are appended to the branch via
    ``act()`` so the next round sees them. Parse errors are collected (not
    silently swallowed) so the caller can surface them as ``Retry`` hints.
    """
    if not action_param or not program.lacts:
        return {}, []

    from lionagi.lndl._parse_function_call import parse_function_call

    from ..act.act import act
    from ..fields import ActionRequestModel

    requests: list = []
    aliases: list[str] = []
    parse_errors: list[str] = []
    for la in program.lacts:
        try:
            parsed = parse_function_call(la.call)
        except Exception as exc:
            parse_errors.append(
                f"<lact {la.alias}> parse failed: {exc} (call: {la.call!r:.80})"
            )
            continue
        requests.append(
            ActionRequestModel(
                function=parsed["operation"],
                arguments=parsed["arguments"],
            )
        )
        aliases.append(la.alias)

    if not requests:
        return {}, parse_errors

    responses = await act(branch, requests, action_param)
    responses = list(responses or [])
    # Preserve alias→response identity *before* dropping Nones. Filtering first
    # would shift later results onto earlier aliases when an action returns
    # None (e.g. permission-denied or skipped tools).
    results: dict[str, Any] = {}
    for alias, resp in zip(aliases, responses, strict=False):
        if resp is None:
            continue
        results[alias] = resp.output if hasattr(resp, "output") else resp
    return results, parse_errors


async def _try_finalize_lndl_once(
    *,
    branch: "Branch",
    raw: str,
    chat_param: ChatParam,
    action_param: ActionParam | None,
    operative: Union["Operative", None],
    model_class: type[BaseModel] | None,
) -> tuple[Any, str | None]:
    """One pass: parse → assemble → execute → validate.

    Returns ``(result, error)``. If ``error`` is non-None, the caller can
    retry with a fix prompt.
    """
    from pydantic import ValidationError

    from lionagi.lndl import (
        Lexer,
        Parser,
        assemble,
        collect_actions,
        normalize_lndl_text,
        replace_actions,
    )

    raw = normalize_lndl_text(raw)
    try:
        lexer = Lexer(raw)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=raw)
        program = parser.parse()
    except Exception as e:
        return raw, f"LNDL syntax error: {e}"

    if not program.out_block:
        return raw, "Missing OUT{} block — declare your final aliases there."

    target_type = chat_param.response_format
    output = assemble(program, target_type)
    if not output:
        return raw, "OUT{} is empty after parsing."

    # Required-field check: every non-optional, non-action field on the target
    # must appear in OUT{} (or be filled by execution later).
    target_cls = (
        target_type
        if isinstance(target_type, type)
        else type(target_type) if target_type else None
    )
    if target_cls is not None and hasattr(target_cls, "model_fields"):
        skip_action = {"action_required", "action_requests", "action_responses"}
        missing = []
        for fname, finfo in target_cls.model_fields.items():
            if fname in skip_action:
                continue
            if not finfo.is_required():
                continue
            if fname not in output:
                missing.append(fname)
        if missing:
            return raw, (
                f"OUT{{}} missing required field(s): {', '.join(missing)}. "
                f"Every required spec must appear in OUT{{}}."
            )

    placeholders = collect_actions(output)
    action_requests = []
    action_responses_models = []
    if placeholders and action_param is None:
        # invoke_actions=False: replace ActionCall placeholders with their
        # raw call strings so the output can validate against the target
        # model. Without this, fields contain ActionCall objects that fail
        # Pydantic validation and leak internal types to the caller.
        raw_calls = {ac.name: ac.raw_call for ac in placeholders}
        output = replace_actions(output, raw_calls)
    elif placeholders and action_param is not None:
        from ..act.act import act
        from ..fields import ActionRequestModel

        action_requests = [
            ActionRequestModel(function=ac.function, arguments=ac.arguments)
            for ac in placeholders
        ]
        responses = list(await act(branch, action_requests, action_param) or [])
        # Pair before filtering so a None response doesn't shift downstream
        # placeholders onto the wrong alias.
        results_by_name = {}
        kept_responses: list[Any] = []
        for ac, resp in zip(placeholders, responses, strict=False):
            if resp is None:
                continue
            results_by_name[ac.name] = resp.output if hasattr(resp, "output") else resp
            kept_responses.append(resp)
        action_responses_models = kept_responses
        output = replace_actions(output, results_by_name)

    if action_requests:
        output["action_requests"] = action_requests
    if action_responses_models:
        output["action_responses"] = action_responses_models

    if operative is not None and operative.base_type is not None:
        try:
            return operative.response_type.model_validate(output), None
        except ValidationError as ve:
            return (
                output,
                f"Validation against {operative.base_type.__name__} failed: {ve}",
            )
        except Exception as e:
            return output, f"Validation error: {e}"
    elif model_class is not None:
        try:
            return model_class.model_validate(output), None
        except ValidationError as ve:
            return output, f"Validation against {model_class.__name__} failed: {ve}"
        except Exception as e:
            return output, f"Validation error: {e}"

    return output, None


async def _finalize_lndl(
    *,
    branch: "Branch",
    result: Any,
    chat_param: ChatParam,
    action_param: ActionParam | None,
    operative: Union["Operative", None],
    model_class: type[BaseModel] | None,
    lndl_retries: int = 0,
    handle_validation: HandleValidation = "return_value",
    middle: Middle | None = None,
) -> Any:
    """Assemble LNDL output with optional retry (single-round mode).

    On parse/validation failure, sends the error back to the model and asks
    for a fix, up to ``lndl_retries`` extra attempts. Routes retries through
    the same ``middle`` as the original call for CLI/streaming consistency.
    """
    raw = _get_raw_assistant_response(branch, result)
    if not raw:
        return result

    last_output: Any = result
    last_error: str | None = None
    attempts = lndl_retries + 1

    for attempt in range(attempts):
        out, err = await _try_finalize_lndl_once(
            branch=branch,
            raw=raw,
            chat_param=chat_param,
            action_param=action_param,
            operative=operative,
            model_class=model_class,
        )
        if err is None:
            return out
        last_output = out
        last_error = err
        if attempt + 1 >= attempts:
            break
        fixed = await _request_lndl_fix(
            branch, raw, err, chat_param=chat_param, middle=middle
        )
        if not fixed:
            break
        raw = fixed

    return _apply_lndl_handle_validation(
        last_output, last_error, handle_validation, target=model_class
    )


def _apply_lndl_handle_validation(
    output: Any,
    error: str | None,
    policy: HandleValidation,
    *,
    target: type[BaseModel] | None,
) -> Any:
    """Apply ``handle_validation`` to a possibly-failed LNDL final result."""
    if error is None:
        return output
    match policy:
        case "return_value":
            return output
        case "return_none":
            return None
        case "raise":
            target_name = getattr(target, "__name__", repr(target))
            raise ValueError(f"LNDL failed to produce valid {target_name}: {error}")
    return output


def _schema_field_hint(target_type: Any) -> str | None:
    """Render a one-line reminder of the schema's field names.

    Used by the final-round continuation to nudge the model toward the
    correct OUT{} structure when it has been writing only scratch.
    """
    cls = (
        target_type
        if isinstance(target_type, type)
        else type(target_type) if target_type else None
    )
    if cls is None or not hasattr(cls, "model_fields"):
        return None
    skip = {"action_required", "action_requests", "action_responses"}
    names = [n for n in cls.model_fields if n not in skip]
    if not names:
        return None
    return "Schema fields to fill in OUT{}: " + ", ".join(names)


async def _run_lndl_react(
    *,
    branch: "Branch",
    result: Any,
    chat_param: ChatParam,
    action_param: ActionParam | None,
    operative: Union["Operative", None],
    model_class: type[BaseModel] | None,
    max_rounds: int,
    handle_validation: HandleValidation = "return_value",
    middle: Middle | None = None,
) -> Any:
    """Multi-round LNDL — ReAct flavor.

    Each round:
      1. parse the most recent assistant response
      2. execute ALL <lact>s (regardless of OUT membership) so tool results
         land in chat history before the next round
      3. collect any <lvar note.X> declarations into the cross-round scratchpad
      4. classify the round outcome — Success / Continue / Retry — and either
         return, advance with a continuation prompt, or feed the error back

    The first round consumes the assistant response that ``operate`` already
    triggered. Subsequent rounds dispatch through ``middle`` (the same
    communicate/run_and_collect selected by the outer operate call) so CLI
    and streaming paths are honoured.
    """
    from lionagi.lndl import Continue, Failed, Retry, Success

    raw = _get_raw_assistant_response(branch, result)
    if not raw:
        return result

    if middle is None:
        from ..communicate.communicate import communicate

        middle = communicate

    scratchpad: dict[str, Any] = {}
    last_error: str | None = None
    last_partial: Any = result

    for round_num in range(max_rounds):
        outcome = await _run_one_lndl_round(
            branch=branch,
            raw=raw,
            chat_param=chat_param,
            action_param=action_param,
            operative=operative,
            model_class=model_class,
            scratchpad=scratchpad,
        )

        if isinstance(outcome, Success):
            return outcome.output
        if isinstance(outcome, Failed):
            raise outcome.error
        if isinstance(outcome, Retry):
            last_error = outcome.error
        elif isinstance(outcome, Continue):
            last_error = None
            last_partial = result

        if round_num + 1 >= max_rounds:
            break

        # Drive the next round through the same middle (communicate or
        # run_and_collect) that the outer operate() used. This ensures
        # CLI/streaming models go through the right path and that messages
        # are persisted by the middle itself — no manual add_message needed.
        cont = _build_lndl_continuation(round_num, max_rounds, last_error=last_error)
        is_final = (round_num + 2) >= max_rounds
        if is_final:
            hint = _schema_field_hint(chat_param.response_format)
            if hint:
                cont = cont + "\n\n" + hint
        try:
            # parse_param=None: LNDL handles its own parsing per round.
            # skip_validation=True: we validate in _run_one_lndl_round.
            raw = await middle(
                branch, cont, chat_param, None, False, skip_validation=True
            )
            if not isinstance(raw, str):
                raw = str(raw) if raw is not None else ""
        except Exception as e:
            last_error = f"chat failed: {e}"
            break

    return _apply_lndl_handle_validation(
        last_partial,
        last_error
        or "Multi-round LNDL exhausted without producing a valid OUT{} block.",
        handle_validation,
        target=model_class,
    )


async def _run_one_lndl_round(
    *,
    branch: "Branch",
    raw: str,
    chat_param: ChatParam,
    action_param: ActionParam | None,
    operative: Union["Operative", None],
    model_class: type[BaseModel] | None,
    scratchpad: dict[str, Any],
):
    """Run a single LNDL round and return a RoundOutcome.

    Mutates ``scratchpad`` in-place when <lvar note.X> declarations are seen
    so the next round can reference them.
    """
    from pydantic import ValidationError

    from lionagi.lndl import (
        Continue,
        Failed,
        Lexer,
        Parser,
        Retry,
        Success,
        assemble,
        collect_actions,
        collect_notes,
        normalize_lndl_text,
        replace_actions,
    )

    _ = Continue  # variant referenced only for type-doc clarity

    raw = normalize_lndl_text(raw)

    try:
        lexer = Lexer(raw)
        tokens = lexer.tokenize()
        parser = Parser(tokens, source_text=raw)
        program = parser.parse()
    except Exception as e:
        return Retry(error=f"LNDL syntax error: {e}")

    # Promote note.X lvars into scratchpad — visible to this round's OUT and
    # any later round's continuation.
    for k, v in collect_notes(program).items():
        scratchpad[k] = v

    # Eagerly execute every lact this round. Tool messages persist on the
    # branch so the next round's chat sees them.
    action_results, lact_errors = await _execute_program_lacts(
        branch, program, action_param
    )

    if not program.out_block:
        if lact_errors:
            # No OUT{} yet AND some lacts failed to parse — surface the
            # errors as a Retry so the continuation prompt tells the model
            # what went wrong, rather than silently producing a round with
            # no tool results and no error signal.
            return Retry(error="; ".join(lact_errors))
        # No commit yet — model is still thinking. Continue.
        return Continue()

    target_type = chat_param.response_format
    output = assemble(
        program, target_type, action_results=action_results, scratchpad=scratchpad
    )
    if not output:
        return Retry(error="OUT{} is empty after parsing.")

    # Required-field check — only fail if a required spec is missing.
    target_cls = (
        target_type
        if isinstance(target_type, type)
        else type(target_type) if target_type else None
    )
    if target_cls is not None and hasattr(target_cls, "model_fields"):
        skip_action = {"action_required", "action_requests", "action_responses"}
        missing = []
        for fname, finfo in target_cls.model_fields.items():
            if fname in skip_action:
                continue
            if not finfo.is_required():
                continue
            if fname not in output:
                missing.append(fname)
        if missing:
            return Retry(
                error=(
                    f"OUT{{}} missing required field(s): {', '.join(missing)}. "
                    f"Every required spec must appear in OUT{{}}."
                )
            )

    # Any leftover ActionCall placeholders from `assemble` belong to lacts
    # that didn't execute (parse error etc). Substitute with the executed
    # results we already have on hand.
    if action_results:
        output = replace_actions(output, action_results)

    # Surface any unresolved placeholders (lacts that failed parsing).
    leftover = collect_actions(output)
    if leftover:
        return Retry(
            error=(
                "Some <lact> calls did not execute: "
                f"{[a.name for a in leftover]}. Check that arguments are literal "
                f"values and the function name matches a registered tool."
            )
        )

    if operative is not None and operative.base_type is not None:
        try:
            return Success(operative.response_type.model_validate(output))
        except ValidationError as ve:
            return Retry(
                error=f"Validation against {operative.base_type.__name__} failed: {ve}"
            )
        except Exception as e:
            return Failed(e)
    if model_class is not None:
        try:
            return Success(model_class.model_validate(output))
        except ValidationError as ve:
            return Retry(
                error=f"Validation against {model_class.__name__} failed: {ve}"
            )
        except Exception as e:
            return Failed(e)

    return Success(output)


def prepare_operate_kw(
    branch: "Branch",
    *,
    instruct: Instruct = None,
    instruction: Instruction | JsonValue = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    progression: Progression = None,
    imodel: "iModel" = None,  # deprecated
    chat_model: "iModel" = None,
    invoke_actions: bool = True,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    parse_model: "iModel" = None,
    skip_validation: bool = False,
    handle_validation: HandleValidation = "return_value",
    tools: "ToolRef" = None,
    operative: "Operative" = None,
    response_format: type[BaseModel] = None,
    lndl: bool = False,
    lndl_retries: int = 0,
    lndl_rounds: int = 1,
    actions: bool = False,
    reason: bool = False,
    call_params: AlcallParams = None,
    action_strategy: Literal["sequential", "concurrent"] = "concurrent",
    verbose_action: bool = False,
    field_models: list[FieldModel | Spec] = None,
    operative_model: type[BaseModel] = None,  # deprecated
    request_model: type[BaseModel] = None,  # deprecated
    include_token_usage_to_model: bool = False,
    clear_messages: bool = False,
    stream_persist: bool = False,
    persist_dir: str | None = None,
    middle: Middle | None = None,
    **kwargs,
) -> dict:
    # Handle deprecated parameters
    if operative_model:
        warnings.warn(
            "Parameter 'operative_model' is deprecated and will be removed in v0.21.0. "
            "Use 'response_format'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if request_model:
        warnings.warn(
            "Parameter 'request_model' is deprecated and will be removed in v0.21.0. "
            "Use 'response_format'.",
            DeprecationWarning,
            stacklevel=2,
        )
    if imodel:
        warnings.warn(
            "Parameter 'imodel' is deprecated and will be removed in v0.21.0. Use 'chat_model'.",
            DeprecationWarning,
            stacklevel=2,
        )

    if (
        (operative_model and response_format)
        or (operative_model and request_model)
        or (response_format and request_model)
    ):
        raise ValueError(
            "Cannot specify multiple of: operative_model, response_format, request_model"
        )

    response_format = response_format or operative_model or request_model
    chat_model = chat_model or imodel or branch.chat_model
    parse_model = parse_model or chat_model

    # Convert dict-based instructions
    if isinstance(instruct, dict):
        instruct = Instruct(**instruct)

    instruct = instruct or Instruct(
        instruction=instruction,
        guidance=guidance,
        context=context,
    )

    if reason:
        instruct.reason = True
    if actions:
        instruct.actions = True
        if action_strategy:
            instruct.action_strategy = action_strategy

    # Convert field_models to Spec if needed
    fields_dict = None
    if field_models:
        fields_dict = {}
        for fm in field_models:
            # Convert FieldModel to Spec
            if isinstance(fm, FieldModel):
                spec = fm.to_spec()
            elif isinstance(fm, Spec):
                spec = fm
            else:
                raise TypeError(f"Expected FieldModel or Spec, got {type(fm)}")

            if spec.name:
                fields_dict[spec.name] = spec

    # Build Operative if needed
    operative = None
    _has_actions = instruct.actions or actions
    _actions_for_request = _has_actions and not lndl  # LNDL: no action fields in prompt
    _actions_for_response = _has_actions  # but still need them in response model
    _need_operative = instruct.reason or _has_actions or fields_dict or response_format
    if _need_operative and (
        response_format or _has_actions or fields_dict or instruct.reason
    ):
        from .step import Step

        operative = Step.request_operative(
            base_type=response_format,
            reason=instruct.reason,
            actions=_has_actions,  # always True when actions requested (for response model)
            fields=fields_dict,
        )
        operative = Step.respond_operative(operative)

    # LNDL: prompt uses the request model so reason/custom fields render,
    # but the LNDL renderer skips framework action_* fields.
    # JSON: prompt uses the request model (with action_required/action_requests)
    if operative:
        final_response_format = operative.request_type
    else:
        final_response_format = response_format
    # Choose ChatParam vs RunParam. RunParam is required when the middle
    # streams via run() (CLI endpoints, explicit stream_persist, or when
    # caller passes a middle that needs persist_dir). Defaulting to
    # RunParam for CLI endpoints keeps the call sites free of path plumbing.
    is_cli = bool(getattr(chat_model, "is_cli", False))
    use_run_param = is_cli or stream_persist or persist_dir is not None

    param_cls = RunParam if use_run_param else ChatParam
    param_kw = dict(
        guidance=instruct.guidance,
        context=instruct.context,
        sender=sender or branch.user or "user",
        recipient=recipient or branch.id,
        response_format=final_response_format,
        formatter=LndlFormatter if lndl else JsonFormatter,
        progression=progression,
        tool_schemas=tool_schemas,
        images=images,
        image_detail=image_detail,
        plain_content=None,
        include_token_usage_to_model=include_token_usage_to_model,
        imodel=chat_model,
        imodel_kw=kwargs,
    )
    if use_run_param:
        param_kw["stream_persist"] = stream_persist
        if persist_dir is not None:
            param_kw["persist_dir"] = persist_dir
    chat_param = param_cls(**param_kw)

    parse_param = None
    if final_response_format and not skip_validation:
        from ..parse.parse import get_default_call

        parse_param = ParseParam(
            response_format=final_response_format,
            fuzzy_match_params=FuzzyMatchKeysParams(),
            handle_validation=handle_validation,
            alcall_params=get_default_call(),
            imodel=parse_model,
            imodel_kw={},
            formatter=chat_param.formatter,
        )

    action_param = None
    if invoke_actions and (instruct.actions or actions):
        from ..act.act import _get_default_call_params

        action_param = ActionParam(
            action_call_params=call_params or _get_default_call_params(),
            tools=tools,
            strategy=action_strategy or instruct.action_strategy or "concurrent",
            suppress_errors=True,
            verbose_action=verbose_action,
        )

    lndl_prior_system = _ensure_lndl_system_prompt(branch) if lndl else _NO_RESTORE

    return {
        "instruction": instruct.instruction,
        "chat_param": chat_param,
        "parse_param": parse_param,
        "action_param": action_param,
        "handle_validation": handle_validation,
        "invoke_actions": invoke_actions,
        "skip_validation": skip_validation,
        "clear_messages": clear_messages,
        "operative": operative,
        "middle": middle,
        "lndl_retries": lndl_retries if lndl else 0,
        "lndl_rounds": max(1, lndl_rounds) if lndl else 1,
        "_lndl_prior_system": lndl_prior_system,
    }


# Sentinel returned by ``_ensure_lndl_system_prompt`` when no restore is
# needed (LNDL prompt was already present, or branch had no system message
# we mutated). Distinct from ``None``, which means "restore to no system".
_NO_RESTORE = object()


def _ensure_lndl_system_prompt(branch: "Branch") -> Any:
    """Inject the LNDL syntax block into the branch system message.

    Returns a token for ``_restore_lndl_system_prompt``:
      * ``_NO_RESTORE``  — no mutation happened (marker already present, etc.)
      * the prior ``System`` instance (or ``None``) — restore to this on exit

    Searches the existing system message for the LNDL marker; if absent,
    appends the prompt. The mutation is restored once the operate call
    returns, so non-LNDL calls on the same branch don't see this prompt.
    """
    from lionagi.lndl.prompt import LNDL_SYSTEM_PROMPT

    marker = "LNDL — Structured Output with Natural Thinking"
    existing_system = branch.msgs.system
    existing_text: str = ""
    if existing_system is not None:
        try:
            existing_text = existing_system.content.system_message or ""
        except Exception:
            existing_text = ""

    if marker in existing_text:
        return _NO_RESTORE  # already injected — nothing to do or restore

    if not existing_text.strip():
        new_text = LNDL_SYSTEM_PROMPT.strip()
    else:
        new_text = existing_text.rstrip() + "\n\n" + LNDL_SYSTEM_PROMPT.strip()

    new_system = branch.msgs.create_system(system=new_text)
    branch.msgs.set_system(new_system)
    return existing_system  # may be None — caller restores faithfully


def _restore_lndl_system_prompt(branch: "Branch", token: Any) -> None:
    """Undo a previous ``_ensure_lndl_system_prompt`` injection.

    Called from a ``finally`` so the LNDL system prompt does not leak into
    later non-LNDL calls on the same branch.
    """
    if token is _NO_RESTORE:
        return
    if token is None:
        # We injected over an empty/missing system. Best we can do without
        # exposing a "delete system" API on MessageManager is to overwrite
        # with an empty system, then drop it from the progression. Since
        # the API surface only supports ``set_system`` (replace), we leave
        # an empty system in place — operationally indistinguishable from
        # "no system" for downstream callers.
        empty = branch.msgs.create_system(system="")
        branch.msgs.set_system(empty)
        return
    branch.msgs.set_system(token)


async def operate(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_param: ChatParam,
    action_param: ActionParam | None = None,
    parse_param: ParseParam | None = None,
    handle_validation: HandleValidation = "return_value",
    invoke_actions: bool = True,
    skip_validation: bool = False,
    clear_messages: bool = False,
    reason: bool = False,
    field_models: list[FieldModel | Spec] | None = None,
    operative: Union["Operative", None] = None,
    middle: Middle | None = None,
    lndl_retries: int = 0,
    lndl_rounds: int = 1,
    _lndl_prior_system: Any = _NO_RESTORE,
) -> BaseModel | dict | str | None:
    """Execute operation with optional action handling.

    Args:
        branch: Branch instance
        instruction: Instruction or JSON value
        chat_param: Chat parameters
        action_param: Action parameters
        parse_param: Parse parameters
        handle_validation: Validation handling strategy
        invoke_actions: Whether to invoke actions
        skip_validation: Whether to skip validation
        clear_messages: Whether to clear messages
        reason: Whether to include reasoning
        field_models: List of FieldModel or Spec objects
        operative: Operative instance

    Returns:
        Result of operation
    """
    try:
        return await _operate_inner(
            branch,
            instruction=instruction,
            chat_param=chat_param,
            action_param=action_param,
            parse_param=parse_param,
            handle_validation=handle_validation,
            invoke_actions=invoke_actions,
            skip_validation=skip_validation,
            clear_messages=clear_messages,
            reason=reason,
            field_models=field_models,
            operative=operative,
            middle=middle,
            lndl_retries=lndl_retries,
            lndl_rounds=lndl_rounds,
        )
    finally:
        _restore_lndl_system_prompt(branch, _lndl_prior_system)


async def _operate_inner(
    branch: "Branch",
    instruction: JsonValue | Instruction,
    chat_param: ChatParam,
    action_param: ActionParam | None = None,
    parse_param: ParseParam | None = None,
    handle_validation: HandleValidation = "return_value",
    invoke_actions: bool = True,
    skip_validation: bool = False,
    clear_messages: bool = False,
    reason: bool = False,
    field_models: list[FieldModel | Spec] | None = None,
    operative: Union["Operative", None] = None,
    middle: Middle | None = None,
    lndl_retries: int = 0,
    lndl_rounds: int = 1,
) -> BaseModel | dict | str | None:
    _cctx = chat_param
    _is_lndl_pre = chat_param.formatter is LndlFormatter
    if _is_lndl_pre:
        # LNDL owns its own parse / retry / handle_validation pipeline in
        # ``_finalize_lndl`` and ``_run_lndl_react``. Skip the JSON-style
        # parse layer to avoid (a) double-parsing and (b) the parse layer
        # spending an LLM-side retry call before our LNDL finalizer ever runs.
        _pctx = None
    else:
        _pctx = (
            parse_param.with_updates(handle_validation="return_value")
            if parse_param
            else ParseParam(
                response_format=chat_param.response_format,
                imodel=branch.parse_model,
                handle_validation="return_value",
                formatter=chat_param.formatter,
            )
        )

    # Update tool schemas
    if tools := (action_param.tools or True) if action_param else None:
        tool_schemas = branch.acts.get_tool_schema(tools=tools)
        _cctx = _cctx.with_updates(tool_schemas=tool_schemas)

    # Extract model class
    model_class = None
    if chat_param.response_format is not None:
        if isinstance(chat_param.response_format, type) and issubclass(
            chat_param.response_format, BaseModel
        ):
            model_class = chat_param.response_format
        elif isinstance(chat_param.response_format, BaseModel):
            model_class = type(chat_param.response_format)

    # Convert field_models to fields dict
    fields_dict = None
    if field_models:
        fields_dict = {}
        for fm in field_models:
            if isinstance(fm, FieldModel):
                spec = fm.to_spec()
            elif isinstance(fm, Spec):
                spec = fm
            else:
                raise TypeError(f"Expected FieldModel or Spec, got {type(fm)}")

            if spec.name:
                fields_dict[spec.name] = spec

    # Create operative if needed
    _is_lndl = chat_param.formatter is LndlFormatter
    _actions_here = bool(action_param) and not _is_lndl
    if not operative and (model_class or _actions_here or fields_dict):
        from .step import Step

        operative = Step.request_operative(
            base_type=model_class,
            reason=reason,
            actions=_actions_here,
            fields=fields_dict,
        )
        operative = Step.respond_operative(operative)

        # Update contexts — use request_type (excludes action_responses)
        request_fmt = operative.request_type or model_class
        if request_fmt:
            _cctx = _cctx.with_updates(response_format=request_fmt)
            if _pctx is not None:
                _pctx = _pctx.with_updates(response_format=request_fmt)

    if middle is None:
        if isinstance(_cctx, RunParam) or getattr(branch.chat_model, "is_cli", False):
            from ..run.run import run_and_collect

            middle = run_and_collect
        else:
            from ..communicate.communicate import communicate

            middle = communicate

    result = await middle(
        branch,
        instruction,
        _cctx,
        _pctx,
        clear_messages,
        skip_validation=skip_validation,
    )

    if skip_validation:
        return result

    _is_lndl = chat_param.formatter is LndlFormatter

    # LNDL handles its own validation/action execution via _finalize_lndl.
    # Skip the JSON-path early validation gate.
    if not _is_lndl and model_class and not isinstance(result, model_class):
        match handle_validation:
            case "return_value":
                return result
            case "return_none":
                return None
            case "raise":
                expected_name = getattr(model_class, "__name__", repr(model_class))
                received_snippet = repr(result)[:200]
                raise ValueError(
                    f"Failed to parse LLM response into '{expected_name}'. "
                    f"Received (truncated): {received_snippet}. "
                    f"Hint: verify the model supports structured JSON output "
                    f"(e.g. response_format / function-calling) for this provider."
                )

    # LNDL path: always parse+assemble the raw LNDL output, even when
    # invoke_actions=False. LNDL parsing is how we produce the structured
    # result — skipping it returns raw text. When invoke_actions=False,
    # pass action_param=None so <lact> placeholders remain unexecuted.
    if _is_lndl:
        _act = action_param if invoke_actions else None
        # Use _cctx (not the original chat_param) — it has tool_schemas and
        # the updated response_format from the operative, so continuations
        # and retries reflect the actual request context.
        if lndl_rounds and lndl_rounds > 1:
            return await _run_lndl_react(
                branch=branch,
                result=result,
                chat_param=_cctx,
                action_param=_act,
                operative=operative,
                model_class=model_class,
                max_rounds=lndl_rounds,
                handle_validation=handle_validation,
                middle=middle,
            )
        return await _finalize_lndl(
            branch=branch,
            result=result,
            chat_param=_cctx,
            action_param=_act,
            operative=operative,
            model_class=model_class,
            lndl_retries=lndl_retries,
            handle_validation=handle_validation,
            middle=middle,
        )

    if not invoke_actions:
        return result

    # JSON path: look for action_requests on the parsed result
    if model_class:
        requests = getattr(result, "action_requests", None)
    elif isinstance(result, dict):
        requests = result.get("action_requests")
    else:
        requests = None

    action_response_models = None
    if action_param and requests is not None:
        from ..act.act import act

        action_response_models = await act(branch, requests, action_param)

    if not action_response_models:
        return result

    # Filter None values
    action_response_models = [r for r in action_response_models if r is not None]

    if not action_response_models:
        return result

    if not model_class:
        # Dict response: merge action_responses in. Raw-text results stay
        # untouched (text has no structured slot for action_responses).
        if isinstance(result, dict):
            result["action_responses"] = action_response_models
        return result

    # If we have model_class, we must have operative (created at line 268)
    # First set the response_model to the existing result
    operative.response_model = result
    # Then update it with action_responses
    operative.update_response_model(data={"action_responses": action_response_models})
    return operative.response_model
