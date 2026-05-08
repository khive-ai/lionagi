import pytest

from lionagi.providers.ag2.groupchat.endpoint import AG2GroupChatEndpoint
from lionagi.providers.ag2.groupchat.models import (
    AG2GroupChatRequest,
    GroupChatSpec,
    build_group_chat,
)
from lionagi.service.imodel import iModel


def _noop(*_, **__):
    return None


def test_ag2_groupchat_request_normalizes_legacy_and_rich_fields():
    request = AG2GroupChatRequest.model_validate(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "coordinate this"}],
                }
            ],
            "agent_configs": [
                {
                    "name": "Planner",
                    "role": "planning",
                    "handoff_conditions": [
                        {"target": "Writer", "condition": "plan is ready"}
                    ],
                },
                {"name": "Writer"},
            ],
            "max_rounds": 4,
            "context": {"ready": True},
            "pattern": "round_robin",
            "initial_agent": "Planner",
            "group_manager_args": {"silent": True},
            "safeguard_policy": {"rules": ["no secrets"]},
            "yield_on": ["text", "tool_result"],
        }
    )

    assert request.prompt == "coordinate this"
    assert request.max_round == 4
    assert request.context_variables == {"ready": True}
    assert request.agents[0].handoffs[0].target == "Writer"
    assert request.pattern == "round_robin"

    spec = request.to_group_chat_spec()
    assert spec.objective == "coordinate this"
    assert spec.initial_agent == "Planner"
    assert spec.group_manager_args == {"silent": True}


def test_ag2_groupchat_builds_sdk_pattern_with_handoffs_and_agent_knobs():
    def lookup(query: str) -> str:
        """Lookup helper."""
        return query

    spec = GroupChatSpec(
        name="research",
        objective="answer",
        context={"ready": True},
        initial_agent="Researcher",
        agents=[
            {
                "name": "Researcher",
                "role": "find facts",
                "description": "Finds supporting facts",
                "tools": ["lookup"],
                "handoffs": [
                    {
                        "target": "Writer",
                        "condition": "facts are gathered",
                        "llm_function_name": "handoff_to_writer",
                    }
                ],
            },
            {
                "name": "Writer",
                "human_input_mode": "NEVER",
                "after_work": "terminate",
            },
        ],
    )

    user, pattern, agents = build_group_chat(
        spec,
        llm_config=False,
        tool_registry={"lookup": lookup},
    )

    assert user.name == "User"
    assert type(pattern).__name__ == "DefaultPattern"
    assert pattern.initial_agent.name == "Researcher"
    assert list(agents) == ["Researcher", "Writer"]
    assert agents["Researcher"].description == "Finds supporting facts"
    assert agents["Researcher"].handoffs.llm_conditions[0].target.agent_name == "Writer"
    assert agents["Researcher"].handoffs.after_works[0].target.agent_name == "Writer"
    assert agents["Writer"].handoffs.after_works[0].target.display_name() == "Terminate"


def test_ag2_groupchat_runtime_config_and_handlers_stay_out_of_config_kwargs():
    model = iModel(
        provider="ag2",
        endpoint="groupchat",
        ag2_handlers={"on_text": _noop},
        agent_configs=[{"name": "DefaultAgent"}],
        llm_config=False,
    )

    call = model.create_api_calling(
        prompt="hello",
        agent_configs=[{"name": "RuntimeAgent"}],
        tool_registry={"noop": _noop},
        on_text=_noop,
    )

    assert "ag2_handlers" not in model.endpoint.config.kwargs
    assert call.payload["request"].prompt == "hello"
    assert call.call_kwargs["agent_configs"] == [{"name": "RuntimeAgent"}]
    assert call.call_kwargs["tool_registry"] == {"noop": _noop}
    assert call.call_kwargs["on_text"] is _noop


@pytest.mark.asyncio
async def test_ag2_endpoint_stream_passes_rich_request_to_runner(monkeypatch):
    import lionagi.providers.ag2.groupchat.models as group_models

    captured = {}

    def fake_build_group_chat(spec, llm_config, tool_registry, code_executor):
        captured["spec"] = spec
        captured["llm_config"] = llm_config
        captured["tool_registry"] = tool_registry
        captured["code_executor"] = code_executor

        class User:
            name = spec.user_name

        class Pattern:
            pass

        return User(), Pattern(), {agent.name: object() for agent in spec.agents}

    async def fake_stream_group_chat(**kwargs):
        captured["stream_kwargs"] = kwargs
        if False:
            yield None

    monkeypatch.setattr(group_models, "build_group_chat", fake_build_group_chat)
    monkeypatch.setattr(group_models, "stream_group_chat", fake_stream_group_chat)

    endpoint = AG2GroupChatEndpoint(
        llm_config={"default": True},
        tool_registry={"noop": _noop},
        code_executor="executor",
    )

    chunks = [
        chunk
        async for chunk in endpoint.stream(
            {
                "prompt": "coordinate",
                "agents": [{"name": "Planner"}],
                "pattern": "auto",
                "selection_message": "Pick the best agent from {agentlist}",
                "group_manager_args": {"llm_config": False},
                "safeguard_policy": {"rules": ["redact secrets"]},
                "yield_on": ["text"],
                "user_name": "Caller",
                "llm_config": {"request": True},
            },
            on_text=_noop,
        )
    ]

    assert captured["spec"].pattern == "auto"
    assert captured["spec"].selection_message == "Pick the best agent from {agentlist}"
    assert captured["spec"].group_manager_args == {"llm_config": False}
    assert captured["spec"].user_name == "Caller"
    assert captured["llm_config"] == {"request": True}
    assert captured["tool_registry"] == {"noop": _noop}
    assert captured["code_executor"] == "executor"
    assert captured["stream_kwargs"]["safeguard_policy"] == {
        "rules": ["redact secrets"]
    }
    assert captured["stream_kwargs"]["yield_on"] == ["text"]
    assert captured["stream_kwargs"]["on_text"] is _noop
    assert chunks[0].metadata["pattern"] == "auto"
    assert chunks[-1].type == "result"


@pytest.mark.asyncio
async def test_ag2_endpoint_default_no_llm_config_can_stream_default_reply():
    endpoint = AG2GroupChatEndpoint(
        agent_configs=[{"name": "Responder", "default_auto_reply": "done"}]
    )

    chunks = [
        chunk async for chunk in endpoint.stream({"prompt": "hello", "max_round": 2})
    ]

    assert endpoint._llm_config is False
    assert any(
        chunk.type == "text"
        and chunk.content == "done"
        and chunk.metadata.get("agent") == "Responder"
        for chunk in chunks
    )
    assert chunks[-1].type == "result"
