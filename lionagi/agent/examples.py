# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Runnable examples for lionagi's coding agent infrastructure.

Each async function demonstrates one usage pattern. Run any example directly::

    import asyncio
    from lionagi.agent.examples import basic_coding_agent
    asyncio.run(basic_coding_agent())

Or run all examples::

    python -m lionagi.agent.examples

Note: examples use load_settings=False and cwd= a temp directory to avoid
side effects from the user's global ~/.lionagi/settings.yaml and any .mcp.json
files that may be present in parent directories.
"""

import asyncio
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Basic coding agent
# ---------------------------------------------------------------------------


async def basic_coding_agent() -> None:
    """Simplest case: create a coding agent and inspect its registered tools.

    AgentConfig.coding() is the standard preset.  It registers CodingToolkit
    (reader, editor, bash, search, context, subagent) and sets the coding
    system prompt.  Pass model= to wire a chat model for actual LLM calls.
    """
    import tempfile

    from lionagi.agent import AgentConfig, create_agent

    with tempfile.TemporaryDirectory() as tmpdir:
        # cwd= isolates MCP discovery so no parent-directory .mcp.json is loaded.
        config = AgentConfig.coding(cwd=tmpdir)
        branch = await create_agent(config, load_settings=False)

    tool_names = sorted(branch.acts.registry.keys())
    print("Registered tools:", tool_names)
    # -> ['bash', 'context', 'editor', 'reader', 'search', 'subagent']

    system = branch.msgs.system
    print("System prompt set:", system is not None)

    budget = branch.token_budget
    print(
        f"Token budget — used: {budget.used}, limit: {budget.limit}, "
        f"usage: {budget.usage_pct:.1%}"
    )

    # With a model configured the branch is ready for:
    #   result = await branch.operate(instruction="Fix the bug in utils.py")
    # Example with model (requires API key):
    #   config = AgentConfig.coding(model="openai/gpt-4.1", cwd="/path/to/project")
    print("Agent ready.")


# ---------------------------------------------------------------------------
# 2. ReAct loop
# ---------------------------------------------------------------------------


async def react_loop() -> None:
    """Show how branch.ReAct() drives a think-act-observe loop.

    branch.ReAct() calls the LLM, executes the tool it requests, feeds
    the result back, and repeats until the model stops requesting tools.
    This example shows the call signature; live execution requires an API key.

    Typical workflow for "find TODOs and fix them":
        1. search(action='grep', pattern='TODO')  — locate all markers
        2. reader(action='read', path=...)         — inspect each file
        3. editor(action='edit', ...)              — apply the fix
        4. bash(command='uv run pytest')           — verify nothing broke
    """
    import inspect
    import tempfile

    from lionagi.agent import AgentConfig, create_agent

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig.coding(cwd=tmpdir)
        branch = await create_agent(config, load_settings=False)

    sig = inspect.signature(branch.ReAct)
    # Show only the most relevant parameters to keep output readable.
    param_names = list(sig.parameters.keys())[:6]
    print("branch.ReAct key params:", param_names)

    print("Tools available:", sorted(branch.acts.registry.keys()))

    # Live call (requires API key + model on config):
    #   result = await branch.ReAct(
    #       instruct="Find all TODO comments and propose fixes.",
    #       tools=True,      # allow all registered tools
    #       max_extensions=10,  # stop after 10 think-act rounds
    #   )
    print("Call: await branch.ReAct(instruct=..., tools=True, max_extensions=10)")


# ---------------------------------------------------------------------------
# 3. Context management — evicting stale tool results
# ---------------------------------------------------------------------------


async def context_management() -> None:
    """Demonstrate context-window awareness and eviction of old tool outputs.

    In a long session the agent accumulates ActionResponse messages.  The
    context tool (action='evict_action_results') removes old ones from the
    active progression without deleting them from the conversation record.
    branch.progression respects evictions; branch.token_budget uses it.
    """
    import tempfile

    from lionagi.agent import AgentConfig, create_agent
    from lionagi.protocols.messages import ActionRequest, ActionResponse
    from lionagi.protocols.messages.action_request import ActionRequestContent
    from lionagi.protocols.messages.action_response import ActionResponseContent
    from lionagi.protocols.generic.progression import Progression

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig.coding(cwd=tmpdir)
        branch = await create_agent(config, load_settings=False)

    pile = branch.msgs.messages
    progression = branch.msgs.progression
    sender_id = branch.id

    # Simulate 8 bulky search results accumulating during a long session.
    for i in range(8):
        req = ActionRequest(
            content=ActionRequestContent(
                function="search",
                arguments={"action": "grep", "pattern": "TODO"},
            ),
            sender=sender_id,
            recipient=sender_id,
        )
        resp = ActionResponse(
            content=ActionResponseContent(
                function="search",
                arguments={"action": "grep", "pattern": "TODO"},
                output={"content": f"result_{i}: " + "match " * 400, "success": True},
                action_request_id=str(req.id),
            ),
            sender=sender_id,
            recipient=sender_id,
        )
        pile.include(req)
        pile.include(resp)
        progression.append(req.id)
        progression.append(resp.id)

    print(f"Active messages before eviction: {len(branch.progression)}")
    budget_before = branch.token_budget
    print(
        f"Token usage before: {budget_before.used} / {budget_before.limit} "
        f"({budget_before.usage_pct:.1%})"
    )

    # Set up current_progression — mirrors context tool's _ensure_current_progression.
    cp = Progression()
    for uid in branch.msgs.progression:
        cp.append(uid)
    branch.metadata["current_progression"] = cp

    # Evict all but the last 3 ActionResponse messages.
    keep = 3
    ar_uids = [
        uid for uid in cp if uid in pile and isinstance(pile[uid], ActionResponse)
    ]
    to_evict = ar_uids[:-keep] if keep > 0 else ar_uids
    cp.exclude(to_evict)

    print(f"Active messages after eviction: {len(branch.progression)}")
    print(f"Pile still has all {len(pile)} messages (eviction is non-destructive)")
    budget_after = branch.token_budget
    print(
        f"Token usage after: {budget_after.used} / {budget_after.limit} "
        f"({budget_after.usage_pct:.1%})"
    )


# ---------------------------------------------------------------------------
# 4. Permission-controlled worker
# ---------------------------------------------------------------------------


async def permission_controlled_worker() -> None:
    """Multi-agent pattern: orchestrator with full access, worker with restrictions.

    PermissionPolicy.safe() lets the worker read/edit/search freely but
    escalates all bash calls.  A denied call raises PermissionError before
    the tool runs.  An escalated call invokes on_escalate for runtime approval.
    """
    import tempfile

    from lionagi.agent import AgentConfig, PermissionPolicy, create_agent

    with tempfile.TemporaryDirectory() as tmpdir:
        # Orchestrator — unrestricted (allow_all mode).
        orch_config = AgentConfig.coding(cwd=tmpdir)
        orchestrator = await create_agent(orch_config, load_settings=False)
        print("Orchestrator tools:", sorted(orchestrator.acts.registry.keys()))

        # Worker — PermissionPolicy.safe(): read/edit/search OK; bash escalated.
        worker_config = AgentConfig.coding(cwd=tmpdir)
        policy = PermissionPolicy.safe()

        # Escalation handler: mimics the orchestrator deciding at runtime.
        approved_commands: list[str] = []

        async def escalate_to_orchestrator(decision, args: dict):
            cmd = args.get("command", "")
            print(f"  [escalate] worker wants: {cmd!r}")
            # Auto-approve 'git status', deny everything else.
            if cmd.startswith("git status"):
                approved_commands.append(cmd)
                print(f"  [orchestrator] approved: {cmd!r}")
                return True
            print(f"  [orchestrator] denied: {cmd!r}")
            return False

        policy.on_escalate = escalate_to_orchestrator
        worker_config.permissions = policy
        worker = await create_agent(worker_config, load_settings=False)

    # "rm -rf /" — matched by deny list, never escalated.
    decision = policy.check("bash", "", {"command": "rm -rf /"})
    print(f"rm -rf /: {decision.behavior} — {decision.reason}")

    # "docker build" — not in allow or deny, triggers escalate rule.
    decision = policy.check("bash", "", {"command": "docker build ."})
    print(f"docker build .: {decision.behavior} — {decision.reason}")

    # The pre-hook calls on_escalate; docker build denied by our handler.
    hook = policy.to_pre_hook()
    try:
        await hook("bash", "", {"command": "docker build ."})
    except PermissionError as e:
        print(f"docker build hook: PermissionError — {e}")

    # "git status" — escalated, then approved by the handler above.
    await hook("bash", "", {"command": "git status"})
    print(f"git status: approved. Approvals so far: {approved_commands}")


# ---------------------------------------------------------------------------
# 5. Hooks pipeline
# ---------------------------------------------------------------------------


async def hooks_pipeline() -> None:
    """Pre/post hook composition: guard, format, and log in one pipeline.

    Pre-hooks run before the tool executes.  Raising inside a pre-hook aborts
    the call entirely.  Post-hooks run after and receive the result dict.
    The wildcard key 'post:*' fires after every tool call.
    """
    import tempfile

    from lionagi.agent import AgentConfig, create_agent
    from lionagi.agent.hooks import auto_format_python, guard_destructive, log_tool_use

    config = AgentConfig.coding()

    # guard_destructive: blocks rm -rf, force-push, drop table, and similar.
    config.pre("bash", guard_destructive)

    # auto_format_python: runs ruff format on any .py file after an edit.
    config.post("editor", auto_format_python)

    # log_tool_use: emits a log.INFO line after every tool call.
    config.post("*", log_tool_use)

    with tempfile.TemporaryDirectory() as tmpdir:
        config.cwd = tmpdir
        branch = await create_agent(config, load_settings=False)

    print("Hook handlers registered:", sorted(config.hook_handlers.keys()))
    # -> ['post:*', 'post:editor', 'pre:bash']

    # Simulate guard_destructive blocking a dangerous command.
    try:
        await guard_destructive("bash", "", {"command": "rm -rf /tmp/project"})
    except PermissionError as e:
        print(f"Blocked: {e}")

    # Safe command passes through (returns None = no args override).
    result = await guard_destructive("bash", "", {"command": "uv run pytest"})
    print(f"Safe command passed through: result={result}")

    # log_tool_use fires as a post-hook.
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    await log_tool_use(
        "bash", "run", {"command": "uv run pytest"}, {"return_code": 0}
    )
    print("log_tool_use emitted a log.INFO line.")


# ---------------------------------------------------------------------------
# 6. Settings YAML config
# ---------------------------------------------------------------------------


async def settings_yaml_config() -> None:
    """Load hooks from a settings.yaml and apply them to a config.

    apply_hooks_from_settings() resolves 'python:' import specs from the
    trusted_hook_modules allowlist and wires them into the config as callables.
    Shell command hooks ('command:' argv lists) are also supported.
    """
    import yaml

    from lionagi.agent import AgentConfig, apply_hooks_from_settings

    settings_content = """\
hooks:
  pre:
    bash:
      - python: "lionagi.agent.hooks:guard_destructive"
  post:
    "*":
      - python: "lionagi.agent.hooks:log_tool_use"
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        settings_path = Path(tmpdir) / "settings.yaml"
        settings_path.write_text(settings_content)

        raw = yaml.safe_load(settings_path.read_text())
        print("Settings hook phases:", list(raw.get("hooks", {}).keys()))

        config = AgentConfig.coding(cwd=tmpdir)
        apply_hooks_from_settings(
            config,
            raw,
            trusted_hook_modules={"lionagi.agent.hooks"},
        )

        print("Hook handlers after apply:", sorted(config.hook_handlers.keys()))
        # -> ['post:*', 'pre:bash']
        print("pre:bash hooks:", len(config.hook_handlers.get("pre:bash", [])))
        print("post:* hooks:", len(config.hook_handlers.get("post:*", [])))


# ---------------------------------------------------------------------------
# 7. Custom toolkit — domain-specific tools alongside coding tools
# ---------------------------------------------------------------------------


async def custom_toolkit() -> None:
    """Extend a coding agent with domain-specific tools.

    Sage-style agents mix CodingToolkit tools with domain tools (notifications,
    student records, calendar).  Wrap plain async functions as Tool objects
    and call branch.register_tools() after create_agent().
    """
    from lionagi.agent import AgentConfig, create_agent
    from lionagi.protocols.action.tool import Tool

    async def send_notification(
        recipient: str,
        subject: str,
        body: str,
    ) -> dict:
        """Send a notification to a user or team channel.

        Args:
            recipient: Email address or Slack handle.
            subject: Notification subject line.
            body: Notification body text.
        """
        print(f"  [notify] to={recipient!r} subject={subject!r}")
        return {"success": True, "recipient": recipient}

    async def lookup_student(student_id: str) -> dict:
        """Look up a student record by ID.

        Args:
            student_id: The institution's student identifier.
        """
        print(f"  [lookup] student_id={student_id!r}")
        return {"success": True, "student_id": student_id, "name": "Demo Student"}

    notification_tool = Tool(func_callable=send_notification)
    student_tool = Tool(func_callable=lookup_student)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig(
            name="sage-agent",
            tools=["coding"],   # registers CodingToolkit
            system_prompt=(
                "You are an academic support agent. "
                "You can read/edit code AND look up students and send notifications."
            ),
            cwd=tmpdir,
        )
        branch = await create_agent(config, load_settings=False)

    branch.register_tools([notification_tool, student_tool])

    all_tools = sorted(branch.acts.registry.keys())
    print("All registered tools:", all_tools)
    # -> ['bash', 'context', 'editor', 'lookup_student', 'reader',
    #     'search', 'send_notification', 'subagent']


# ---------------------------------------------------------------------------
# 8. Token budget awareness
# ---------------------------------------------------------------------------


async def token_budget_awareness() -> None:
    """Agent monitors its own context usage via branch.token_budget.

    TokenBudget fields: used, limit, remaining, usage_pct, is_warning (>=70%),
    is_critical (>=90%).  CodingToolkit appends a context summary to every
    tool result when notify=True (the default) so the LLM always knows its
    headroom without an explicit context check.
    """
    from lionagi.agent import AgentConfig, create_agent
    from lionagi.protocols.messages import ActionRequest, ActionResponse
    from lionagi.protocols.messages.action_request import ActionRequestContent
    from lionagi.protocols.messages.action_response import ActionResponseContent

    with tempfile.TemporaryDirectory() as tmpdir:
        config = AgentConfig.coding(cwd=tmpdir)
        branch = await create_agent(config, load_settings=False)

    sender_id = branch.id

    print("=== Initial state ===")
    b = branch.token_budget
    print(f"  used={b.used}  limit={b.limit}  remaining={b.remaining}")
    print(
        f"  usage_pct={b.usage_pct:.1%}  "
        f"is_warning={b.is_warning}  is_critical={b.is_critical}"
    )

    # Simulate 5 bulky search results accumulating in a long session.
    pile = branch.msgs.messages
    progression = branch.msgs.progression

    for i in range(5):
        req = ActionRequest(
            content=ActionRequestContent(
                function="search",
                arguments={"action": "grep", "pattern": "TODO"},
            ),
            sender=sender_id,
            recipient=sender_id,
        )
        resp = ActionResponse(
            content=ActionResponseContent(
                function="search",
                arguments={"action": "grep", "pattern": "TODO"},
                output={"content": f"result_{i}: " + "match " * 500, "success": True},
                action_request_id=str(req.id),
            ),
            sender=sender_id,
            recipient=sender_id,
        )
        pile.include(req)
        pile.include(resp)
        progression.append(req.id)
        progression.append(resp.id)

    print("\n=== After 5 large search results ===")
    b2 = branch.token_budget
    print(f"  used={b2.used}  remaining={b2.remaining}  usage_pct={b2.usage_pct:.1%}")
    if b2.is_critical:
        print("  CRITICAL: context nearly full — evict immediately.")
    elif b2.is_warning:
        print("  WARNING: context filling up — evict old action results.")
    else:
        print(f"  Context healthy at {b2.usage_pct:.1%}.")

    # Threshold guard — mirrors what CodingToolkit's _system_status() emits.
    if b2.usage_pct >= 0.7:
        print(
            "  Auto-evict trigger would fire: "
            "context(action='evict_action_results', keep_last=3)"
        )


# ---------------------------------------------------------------------------
# 9. MCP integration
# ---------------------------------------------------------------------------


async def mcp_integration() -> None:
    """Coding tools + MCP tools from a .mcp.json config file.

    When config.mcp_servers is set, create_agent() calls
    branch.acts.load_mcp_config() after registering CodingToolkit.  Both
    coding tools and MCP tools then live in branch.acts.registry together.

    This example writes a minimal .mcp.json and shows the wiring pattern.
    The MCP server binary is not launched (demo only).
    """
    import json

    from lionagi.agent import AgentConfig

    mcp_config = {
        "mcpServers": {
            "khive": {
                "command": "khived",
                "args": ["serve"],
                "env": {},
            }
        }
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        mcp_path = Path(tmpdir) / ".mcp.json"
        mcp_path.write_text(json.dumps(mcp_config))

        config = AgentConfig.coding(
            mcp_servers=["khive"],          # only connect the 'khive' server
            mcp_config_path=str(mcp_path),  # explicit path overrides discovery
            cwd=tmpdir,
        )

        print("config.mcp_servers:", config.mcp_servers)
        print("config.mcp_config_path:", config.mcp_config_path)
        print(
            "On create_agent(), _load_mcp() calls:\n"
            "  branch.acts.load_mcp_config(mcp_path, server_names=['khive'])\n"
            "After loading, branch.acts.registry contains CodingToolkit tools\n"
            "AND all tools exposed by the MCP server."
        )

        # In production (requires khived binary on PATH):
        #   branch = await create_agent(config, load_settings=False)
        #   coding = {'bash', 'context', 'editor', 'reader', 'search', 'subagent'}
        #   mcp_tools = [k for k in branch.acts.registry if k not in coding]
        #   print("MCP tools:", mcp_tools)


# ---------------------------------------------------------------------------
# 10. Image and document reading
# ---------------------------------------------------------------------------


async def image_and_document_reading() -> None:
    """Multimodal file access: text with line numbers, images as base64, docs via docling.

    CodingToolkit's reader auto-detects images by extension and returns a
    base64 data URL so vision-capable LLMs can see them directly.  ReaderTool
    (standalone) adds action='open' for docling conversion of PDF/PPTX/DOCX/HTML
    with an in-process LRU cache keyed by path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use .resolve() to expand the macOS /private/var symlink so that
        # _resolve_workspace_path's containment check succeeds.
        root = Path(tmpdir).resolve()

        # --- Text file ---
        txt = root / "hello.py"
        txt.write_text("def greet():\n    return 'hello'\n")

        # --- Minimal PNG (1x1 white pixel) ---
        import base64

        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YGD4DwABBAEAfbLI3wAAAABJRU5ErkJggg=="
        )
        img = root / "screenshot.png"
        img.write_bytes(png_bytes)

        # _read_file_sync is the same code CodingToolkit's reader wraps.
        from lionagi.tools.coding import _read_file_sync

        txt_result = _read_file_sync(str(txt), 0, 2000, root)
        print("Text read — success:", txt_result["success"])
        print("Content:\n", txt_result.get("content", txt_result.get("error")))

        img_result = _read_file_sync(str(img), 0, 2000, root)
        print("Image read — success:", img_result["success"])
        print("  type:", img_result.get("type"))
        print("  media_type:", img_result.get("media_type"))
        print("  size_bytes:", img_result.get("size_bytes"))
        print("  content prefix:", (img_result.get("content") or "")[:40], "...")

        # --- ReaderTool standalone (adds docling 'open' + caching) ---
        from lionagi.tools.file.reader import ReaderAction, ReaderRequest, ReaderTool

        reader = ReaderTool(workspace_root=root)
        req = ReaderRequest(action=ReaderAction.read, path=str(txt))
        resp = await reader.handle_request(req)
        print("\nReaderTool.read — success:", resp.success)

        # ReaderTool.open converts documents via docling (requires lionagi[reader]).
        # Converted text is cached; subsequent 'read' calls slice the cache.
        print(
            "\nReaderTool.open (docling) pattern:\n"
            "  1. ReaderRequest(action=ReaderAction.open, path='report.pdf')\n"
            "     -> converts PDF, caches plain text at 'report.pdf'\n"
            "  2. ReaderRequest(action=ReaderAction.read, path='report.pdf',\n"
            "                   offset=0, limit=100)\n"
            "     -> slices cached text — no re-conversion\n"
            "Install: pip install 'lionagi[reader]'"
        )


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

_EXAMPLES = [
    ("basic_coding_agent", basic_coding_agent),
    ("react_loop", react_loop),
    ("context_management", context_management),
    ("permission_controlled_worker", permission_controlled_worker),
    ("hooks_pipeline", hooks_pipeline),
    ("settings_yaml_config", settings_yaml_config),
    ("custom_toolkit", custom_toolkit),
    ("token_budget_awareness", token_budget_awareness),
    ("mcp_integration", mcp_integration),
    ("image_and_document_reading", image_and_document_reading),
]


if __name__ == "__main__":
    import sys

    names = sys.argv[1:]  # optional: run specific examples by name

    async def _run_all():
        for name, fn in _EXAMPLES:
            if names and name not in names:
                continue
            print(f"\n{'=' * 60}")
            print(f"  {name}")
            print("=" * 60)
            try:
                await fn()
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")

    asyncio.run(_run_all())
