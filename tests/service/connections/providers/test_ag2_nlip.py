import pytest

from lionagi.providers.ag2.agent.endpoint import _resolve_model_config
from lionagi.providers.ag2.groupchat.models import AG2GroupChatRequest, build_group_chat
from lionagi.providers.ag2.nlip.endpoint import AG2NlipEndpoint
from lionagi.providers.ag2.sandbox import SandboxManager
from lionagi.service.imodel import iModel


def _noop(*_, **__):
    return None


def test_ag2_nlip_endpoint_exports_groupchat_agent_config():
    endpoint = AG2NlipEndpoint(
        url="https://remote.test",
        agent_name="RemoteExpert",
        timeout=12,
        max_retries=2,
        silent=True,
    )

    config = endpoint.as_agent_config(
        role="remote researcher",
        description="Remote specialist",
        context_variables={"topic": "nlip"},
        client_tools=[{"name": "client_tool"}],
    )
    spec = AG2GroupChatRequest(prompt="coordinate").to_group_chat_spec(
        agent_configs=[endpoint]
    )

    assert config == {
        "name": "RemoteExpert",
        "role": "remote researcher",
        "description": "Remote specialist",
        "nlip_url": "https://remote.test",
        "nlip_timeout": 12.0,
        "nlip_max_retries": 2,
        "nlip_silent": True,
        "context_variables": {"topic": "nlip"},
        "nlip_client_tools": [{"name": "client_tool"}],
    }
    assert spec.agents[0].name == "RemoteExpert"
    assert spec.agents[0].nlip_url == "https://remote.test"
    assert spec.agents[0].nlip_timeout == 12.0
    assert spec.agents[0].nlip_max_retries == 2


def test_ag2_groupchat_uses_shared_nlip_remote_agent_builder(monkeypatch):
    import lionagi.providers.ag2.nlip.models as nlip_models

    captured = {}

    def fake_build_nlip_remote_agent(spec):
        from autogen import ConversableAgent

        captured["spec"] = spec
        return ConversableAgent(
            name=spec.name,
            llm_config=False,
            human_input_mode="NEVER",
            default_auto_reply="remote reply",
        )

    monkeypatch.setattr(
        nlip_models, "build_nlip_remote_agent", fake_build_nlip_remote_agent
    )

    endpoint = AG2NlipEndpoint(
        url="https://remote.test",
        agent_name="RemoteExpert",
        timeout=10,
        max_retries=4,
    )
    spec = AG2GroupChatRequest(prompt="coordinate").to_group_chat_spec(
        agent_configs=[endpoint]
    )
    _, pattern, agents = build_group_chat(spec, llm_config=False)

    assert captured["spec"].url == "https://remote.test"
    assert captured["spec"].name == "RemoteExpert"
    assert captured["spec"].timeout == 10.0
    assert captured["spec"].max_retries == 4
    assert pattern.initial_agent.name == "RemoteExpert"
    assert list(agents) == ["RemoteExpert"]


def test_ag2_nlip_runtime_kwargs_survive_imodel_transport():
    model = iModel(
        provider="ag2",
        endpoint="nlip",
        url="https://default.test",
        agent_name="DefaultRemote",
    )

    call = model.create_api_calling(
        prompt="hello",
        url="https://runtime.test",
        agent_name="RuntimeRemote",
        timeout=2,
        max_retries=1,
    )

    assert model.endpoint.config.kwargs == {}
    assert call.payload["request"].prompt == "hello"
    assert call.payload["request"].url is None
    assert call.call_kwargs == {
        "url": "https://runtime.test",
        "agent_name": "RuntimeRemote",
        "timeout": 2,
        "max_retries": 1,
    }


def test_ag2_beta_runtime_kwargs_survive_imodel_transport():
    model = iModel(
        provider="ag2",
        endpoint="agent",
        agent_config={"name": "DefaultAgent"},
        llm_config={"api_type": "openai", "model": "default"},
    )

    call = model.create_api_calling(
        prompt="hello",
        agent_config={"name": "RuntimeAgent"},
        llm_config={"api_type": "openai", "model": "runtime"},
        tool_registry={"noop": _noop},
    )

    assert model.endpoint.config.kwargs == {}
    assert call.payload["request"].prompt == "hello"
    assert call.payload["request"].agent_config is None
    assert call.call_kwargs["agent_config"] == {"name": "RuntimeAgent"}
    assert call.call_kwargs["llm_config"] == {
        "api_type": "openai",
        "model": "runtime",
    }
    assert call.call_kwargs["tool_registry"] == {"noop": _noop}


def test_ag2_beta_model_config_requires_explicit_provider_and_model():
    with pytest.raises(ValueError, match="api_type"):
        _resolve_model_config({"model": "runtime"})

    with pytest.raises(ValueError, match="model"):
        _resolve_model_config({"api_type": "openai"})


@pytest.mark.asyncio
async def test_ag2_sandbox_requires_explicit_model_before_daytona_import():
    manager = SandboxManager(model=None)

    with pytest.raises(ValueError, match="explicit model"):
        await manager.create_agent_sandbox("RemoteExpert", "You are remote.")


@pytest.mark.asyncio
async def test_ag2_nlip_endpoint_stream_uses_shared_request_fields(monkeypatch):
    import lionagi.providers.ag2.nlip.models as nlip_models

    captured = {}

    async def fake_call_nlip_remote(**kwargs):
        captured.update(kwargs)
        return {
            "content": "remote reply",
            "context": {"status": "done"},
            "input_required": "more input",
        }

    monkeypatch.setattr(nlip_models, "call_nlip_remote", fake_call_nlip_remote)

    endpoint = AG2NlipEndpoint()
    chunks = [
        chunk
        async for chunk in endpoint.stream(
            {
                "prompt": "hello",
                "url": "https://remote.test",
                "agent_name": "RemoteExpert",
                "timeout": 2,
                "max_retries": 1,
                "context_variables": {"topic": "nlip"},
                "client_tools": [{"name": "client_tool"}],
            }
        )
    ]

    assert captured["url"] == "https://remote.test"
    assert captured["agent_name"] == "RemoteExpert"
    assert captured["timeout"] == 2.0
    assert captured["max_retries"] == 1
    assert captured["context"] == {"topic": "nlip"}
    assert captured["client_tools"] == [{"name": "client_tool"}]
    assert chunks[0].metadata["agent"] == "RemoteExpert"
    assert chunks[1].type == "text"
    assert chunks[1].content == "remote reply"
    assert chunks[2].metadata["event"] == "input_required"
    assert chunks[-1].type == "result"
