from typing import Literal

from pydantic import JsonValue

from lionagi.models import FieldModel
from lionagi.protocols.messages.manager import Instruction
from lionagi.session.ops.communicate import communicate

from .types import ActionContext, ChatContext, HandleValidation, ParseContext


async def operate(
    branch,
    instruction: JsonValue | Instruction,
    chat_ctx: ChatContext,
    action_ctx: ActionContext | None = None,
    parse_ctx: ParseContext | None = None,
    reason: bool = False,
    field_models: list[FieldModel] | None = None,
    handle_validation: HandleValidation = "raise",
    invoke_actions: bool = True,
    clear_messages=False,
):

    # 1. communicate chat context building to avoid changing parameters
    _cctx = ChatContext(**chat_ctx.to_dict())
    _pctx = ParseContext(**(parse_ctx.to_dict() if parse_ctx else {}))
    _pctx.handle_validation = "return_value"

    if tools := (action_ctx.tools or True) if action_ctx else None:
        _cctx.tool_schemas = branch.acts.get_tool_schema(tools=tools)

    t = type if isinstance(chat_ctx.response_format, type) else dict

    def normalize_field_model(fms):
        if not fms:
            return []
        if not isinstance(fms, list):
            return [fms]
        return fms

    fms = normalize_field_model(field_models)
    operative = None

    if t is type:
        from lionagi.protocols.operatives.step import Step

        operative = Step.request_operative(
            reason=reason,
            actions=bool(action_ctx is not None),
            base_type=chat_ctx.response_format,
            field_models=fms,
        )
        _cctx.response_format = operative.request_type
    elif field_models:
        dict_ = {}
        for fm in fms:
            if fm.name:
                dict_[fm.name] = str(fm.annotated())
        _cctx.response_format = dict_

    result = await communicate(
        branch,
        instruction,
        _cctx,
        _pctx,
        clear_messages,
    )

    if not isinstance(result, t):
        match handle_validation:
            case "return_value":
                return result
            case "return_none":
                return None
            case "raise":
                raise ValueError(
                    "Failed to parse the LLM response into the requested format."
                )
    if not invoke_actions:
        return result

    requests = (
        getattr(result, "action_requests", None)
        if t is type
        else result.get("action_requests", None)
    )

    action_response_models = None
    if action_ctx and requests is not None:
        from .act import act

        action_response_models = await act(
            branch,
            requests,
            strategy=action_ctx.strategy,
            suppress_errors=action_ctx.suppress_errors,
            verbose_action=action_ctx.verbose_action,
            call_params=action_ctx.action_call_params,
        )

    if not action_response_models:
        return result

    if t is dict:
        result.update({"action_responses": action_response_models})
        return result

    from lionagi.protocols.operatives.step import Step

    operative.response_model = result
    operative = Step.respond_operative(
        operative=operative,
        additional_data={"action_responses": action_response_models},
    )
    return operative.response_model
