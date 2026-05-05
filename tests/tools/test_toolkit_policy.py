import pytest

from lionagi.beta.session.constraints import scope_in_resources
from lionagi.beta.session.context import RequestContext
from lionagi.beta.session.session import Branch, Session
from lionagi.tools import ToolKit, ToolKitConfig, tool_action


def _ctx(session: Session, branch: Branch, service: str, name: str) -> RequestContext:
    return RequestContext(
        name=name,
        session_id=session.id,
        branch=branch.name or branch.id,
        service=service,
        _bound_session=session,
        _bound_branch=branch,
    )


class SingleActionKit(ToolKit):
    @tool_action("read")
    async def read(self, args, ctx):
        return {"ok": True}


class MultiActionKit(ToolKit):
    @tool_action("read")
    async def read(self, args, ctx):
        return {"read": True}

    @tool_action("write")
    async def write(self, args, ctx):
        return {"write": True}


class SecureKit(ToolKit):
    @tool_action("read", requires={"fs.read:/workspace/*"})
    async def read(self, args, ctx):
        return {"ok": True}


@pytest.mark.asyncio
async def test_single_action_toolkit_accepts_bare_service_grant():
    session = Session()
    branch = session.create_branch(name="single_tool", resources={"files"})
    kit = SingleActionKit(config=ToolKitConfig(name="files", provider="local"))

    result = await kit.call("read", {}, _ctx(session, branch, "files", "files:read"))

    assert result.status == "success"
    assert kit.tool_schemas(fmt="dict", branch=branch)


@pytest.mark.asyncio
async def test_multi_action_toolkit_requires_scoped_service_grant():
    session = Session()
    branch = session.create_branch(name="multi_tool", resources={"files"})
    kit = MultiActionKit(config=ToolKitConfig(name="files", provider="local"))

    result = await kit.call("read", {}, _ctx(session, branch, "files", "files:read"))

    assert result.status == "error"
    assert "missing capability" in (result.error or "")
    assert kit.tool_schemas(fmt="dict", branch=branch) == []
    assert not scope_in_resources("files:read", {"files"})
    assert scope_in_resources("files:read", {"files:*"})


@pytest.mark.asyncio
async def test_tool_action_requires_extra_core_capability():
    session = Session()
    kit = SecureKit(config=ToolKitConfig(name="secure", provider="local"))
    denied = session.create_branch(name="secure_denied", resources={"secure:read"})
    allowed = session.create_branch(
        name="secure_allowed",
        resources={"secure:read"},
        capabilities={"fs.read:/workspace/*"},
    )

    denied_result = await kit.call(
        "read",
        {},
        _ctx(session, denied, "secure", "secure:read"),
    )
    allowed_result = await kit.call(
        "read",
        {},
        _ctx(session, allowed, "secure", "secure:read"),
    )

    assert denied_result.status == "error"
    assert "missing capabilities" in (denied_result.error or "")
    assert allowed_result.status == "success"
