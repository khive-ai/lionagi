# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Named-role wrapper around Session with shared context accumulation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from lionagi.beta.core.base.element import Element
from lionagi.beta.rules import Validator
from lionagi.ln.types import HashableModel, Operable

from .session import Branch, Session, SessionConfig


def _make_validator():
    return Validator()

__all__ = (
    "Conversation",
    "RoleConfig",
    "create_conversation",
)


class RoleConfig(HashableModel):
    """Model and prompt configuration for a named Conversation role."""

    gen_model: str
    parse_model: str | None = None
    system: str | None = None
    tools: list[str] | None = None
    resources_extra: set[str] | None = None

    @property
    def effective_parse_model(self) -> str:
        return self.parse_model or self.gen_model


class Conversation(Element):
    """Multi-role agent session with shared context accumulation across tasks."""

    session: Session = Field(default_factory=Session, exclude=True)
    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    shared_context: dict[str, Any] = Field(default_factory=dict)
    validator: Validator | None = Field(default=None, exclude=True)

    _role_branches: dict[str, str] = PrivateAttr(default_factory=dict)
    _role_context_seen: dict[str, set[str]] = PrivateAttr(default_factory=dict)
    _model_aliases: dict[str, str] = PrivateAttr(default_factory=dict)
    _validator: Validator = PrivateAttr(default_factory=_make_validator)

    @model_validator(mode="after")
    def _setup_roles(self) -> Conversation:
        """Resource grants are EXPLICIT: branches start with model-name resources only; toolkit access must be declared via shared_resources or resources_extra so denials surface as signal."""
        if self.validator is not None:
            self._validator = self.validator

        for role_name, config in self.roles.items():
            if role_name not in self._role_branches:
                caps = set()
                if config.tools:
                    caps.add("action")
                # Resources: model names only — NOT auto-extended with tools.
                # Toolkit access is granted explicitly via session.shared_resources
                # (applies to all roles) or per-role config.resources_extra
                # (e.g. {"lore:*"}, {"lore:suggest"}).
                resources = set(self._collect_model_names(config))
                if shared := getattr(self.session.config, "shared_resources", None):
                    resources.update(shared)
                if getattr(config, "resources_extra", None):
                    resources.update(config.resources_extra)
                branch = self.session.create_branch(
                    name=role_name,
                    capabilities=caps,
                    resources=resources,
                )
                self._role_branches[role_name] = role_name
                self._role_context_seen[role_name] = set()

                if config.system:
                    from lionagi.beta.core.message import System
                    from lionagi.protocols.messages import Message

                    sys_msg = Message(content=System(system_message=config.system))
                    self.session.add_message(sys_msg, branches=branch)

        return self

    def _collect_model_names(self, config: RoleConfig) -> list[str]:
        names = []
        gen = self._resolve_model_alias(config.gen_model)
        names.append(gen)
        parse = self._resolve_model_alias(config.effective_parse_model)
        if parse != gen:
            names.append(parse)
        return names

    def _resolve_model_alias(self, name: str) -> str:
        return self._model_aliases.get(name, name)

    def _get_branch(self, role: str) -> Branch:
        if role not in self._role_branches:
            raise KeyError(f"Unknown role '{role}'. Available: {list(self.roles.keys())}")
        return self.session.get_branch(self._role_branches[role])

    def _inject_context(self, role: str, required_keys: set[str]) -> dict[str, Any]:
        seen = self._role_context_seen[role]
        to_inject = {}
        for key in required_keys:
            if key not in seen and key in self.shared_context:
                to_inject[key] = self.shared_context[key]
        return to_inject

    def _mark_seen(self, role: str, keys: set[str]) -> None:
        self._role_context_seen[role].update(keys)

    def _store_outputs(
        self, role: str, output: Any, output_fields: list[str] | None = None
    ) -> None:
        """When output_fields are given and a name has no matching attribute, the whole output is stored under that name (assignment DSL "topic -> analysis" stores entire result as "analysis")."""
        if output is None:
            return

        if output_fields:
            for field_name in output_fields:
                if hasattr(output, field_name):
                    self.shared_context[field_name] = getattr(output, field_name)
                elif isinstance(output, dict) and field_name in output:
                    self.shared_context[field_name] = output[field_name]
                else:
                    # Store entire output under the field name
                    if isinstance(output, BaseModel):
                        self.shared_context[field_name] = output.model_dump()
                    else:
                        self.shared_context[field_name] = output
            self._role_context_seen[role].update(output_fields)
        elif isinstance(output, dict):
            self.shared_context.update(output)
            self._role_context_seen[role].update(output.keys())
        elif isinstance(output, BaseModel):
            data = output.model_dump()
            self.shared_context.update(data)
            self._role_context_seen[role].update(data.keys())

    async def op(
        self,
        role: str,
        instruction: str,
        *,
        context: Any | None = None,
        guidance: str | None = None,
        response_format: type[BaseModel] | None = None,
        tools: list[str] | None = None,
        reason: bool = False,
        structure_format: Literal["json", "lndl"] = "json",
        mode: Literal["auto", "generate", "operate", "react"] = "auto",
        max_rounds: int = 3,
        persist: bool = True,
        assignment: str | None = None,
        imodel_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if role not in self.roles:
            raise KeyError(f"Unknown role '{role}'. Available: {list(self.roles.keys())}")

        role_config = self.roles[role]
        branch = self._get_branch(role)

        input_fields: list[str] = []
        output_fields: list[str] = []
        if assignment:
            from lionagi.beta.work.form import parse_assignment

            input_fields, output_fields = parse_assignment(assignment)

        merged_context = {}
        if input_fields:
            injected = self._inject_context(role, set(input_fields))
            merged_context.update(injected)
        if context is not None:
            if isinstance(context, dict):
                merged_context.update(context)
            else:
                merged_context["context"] = context

        effective_mode = mode
        if effective_mode == "auto":
            if response_format or tools or role_config.tools or reason:
                effective_mode = "operate"
            else:
                effective_mode = "generate"

        gen_model = self._resolve_model_alias(role_config.gen_model)
        parse_model = self._resolve_model_alias(role_config.effective_parse_model)

        extra_imodel_kwargs = imodel_kwargs or {}

        if effective_mode == "generate":
            result = await self._do_generate(
                branch=branch,
                instruction=instruction,
                context=merged_context or None,
                guidance=guidance,
                gen_model=gen_model,
                persist=persist,
                imodel_kwargs=extra_imodel_kwargs,
            )
        elif effective_mode == "operate":
            result = await self._do_operate(
                branch=branch,
                instruction=instruction,
                context=merged_context or None,
                guidance=guidance,
                gen_model=gen_model,
                parse_model=parse_model,
                response_format=response_format,
                tools=tools or role_config.tools,
                reason=reason,
                structure_format=structure_format,
                persist=persist,
                imodel_kwargs=extra_imodel_kwargs,
                **kwargs,
            )
        elif effective_mode == "react":
            result = await self._do_react(
                branch=branch,
                instruction=instruction,
                context=merged_context or None,
                guidance=guidance,
                gen_model=gen_model,
                parse_model=parse_model,
                response_format=response_format,
                tools=tools or role_config.tools,
                max_rounds=max_rounds,
                persist=persist,
                imodel_kwargs=extra_imodel_kwargs,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown mode: {effective_mode}")

        if input_fields:
            self._mark_seen(role, set(input_fields))
        if output_fields:
            self._store_outputs(role, result, output_fields)
        elif assignment is None and response_format and isinstance(result, BaseModel):
            # Auto-store all fields if no explicit assignment
            self._store_outputs(role, result)

        return result

    async def _do_generate(
        self,
        branch: Branch,
        instruction: str,
        context: Any | None,
        guidance: str | None,
        gen_model: str,
        persist: bool,
        imodel_kwargs: dict[str, Any],
    ) -> str:
        from lionagi.beta.operations.generate import GenerateParams
        from lionagi.beta.operations.utils import ReturnAs

        full_instruction = instruction
        if guidance:
            full_instruction = f"{instruction}\n\nGuidance: {guidance}"

        params = GenerateParams(
            primary=full_instruction,
            context=context,
            imodel=gen_model,
            return_as=ReturnAs.TEXT,
            imodel_kwargs=imodel_kwargs,
        )

        op = await self.session.conduct("generate", branch=branch, params=params)
        if op.execution.error is not None:
            raise op.execution.error
        result = op.response

        if persist and isinstance(result, str):
            from lionagi.beta.core.message import Instruction
            from lionagi.protocols.messages import Message

            ins_msg = Message(content=Instruction.create(primary=full_instruction, context=context))
            self.session.add_message(ins_msg, branches=branch)

            from lionagi.beta.core.message import Assistant

            res_msg = Message(content=Assistant(response=result))
            self.session.add_message(res_msg, branches=branch)

        return result

    async def _do_operate(
        self,
        branch: Branch,
        instruction: str,
        context: Any | None,
        guidance: str | None,
        gen_model: str,
        parse_model: str,
        response_format: type[BaseModel] | None,
        tools: list[str] | None,
        reason: bool,
        structure_format: Literal["json", "lndl"] = "json",
        persist: bool = True,
        imodel_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        from lionagi.beta.operations.generate import GenerateParams
        from lionagi.beta.operations.operate import OperateParams
        from lionagi.ln.types import Spec

        if imodel_kwargs is None:
            imodel_kwargs = {}

        full_instruction = instruction
        if guidance:
            full_instruction = f"{instruction}\n\nGuidance: {guidance}"

        if response_format:
            if structure_format == "lndl":
                spec_name = response_format.__name__[0].lower() + response_format.__name__[1:]
                operable = Operable([Spec(response_format, name=spec_name)])
            else:
                operable = Operable.from_structure(response_format)
        else:
            from lionagi.beta.operations.react import Analysis

            operable = Operable.from_structure(Analysis)
            response_format = Analysis

        if reason:
            from lionagi.ln.types import Spec

            reason_spec = Spec(str, name="reasoning", description="Chain-of-thought reasoning")
            operable = operable.extend([reason_spec])

        # Auto-render filtered schemas from global toolkits when caller didn't
        # pre-render their own. Filtering uses branch.resources so the LLM only
        # sees actions it's actually permitted to invoke (defense in depth +
        # tighter prompts). Callers that pass an explicit `tools=[...]` list
        # keep full control (no auto-filter applied — they already shaped it).
        rendered_tools = tools
        if rendered_tools is None:
            from lionagi.tools import render_toolkit_schemas

            rendered_tools = render_toolkit_schemas(branch=branch)
            if not rendered_tools:
                rendered_tools = None  # nothing to show — keep field unset

        gen_params = GenerateParams(
            primary=full_instruction,
            context=context,
            imodel=gen_model,
            tool_schemas=rendered_tools,
            structure_format=structure_format,
            imodel_kwargs=imodel_kwargs,
        )

        operate_params = OperateParams(
            operable=operable,
            validator=self._validator,
            generate_params=gen_params,
            invoke_actions=bool(rendered_tools),
            persist=persist,
            parse_imodel=parse_model if parse_model != gen_model else gen_model,
            **kwargs,
        )

        op = await self.session.conduct("operate", branch=branch, params=operate_params)
        if op.execution.error is not None:
            raise op.execution.error
        result = op.response

        if structure_format == "lndl" and result is not None and response_format is not None:
            spec_name = response_format.__name__[0].lower() + response_format.__name__[1:]
            if hasattr(result, spec_name):
                return getattr(result, spec_name)

        return result

    async def _do_react(
        self,
        branch: Branch,
        instruction: str,
        context: Any | None,
        guidance: str | None,
        gen_model: str,
        parse_model: str,
        response_format: type[BaseModel] | None,
        tools: list[str] | None,
        max_rounds: int,
        persist: bool,
        imodel_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> Any:
        from lionagi.beta.operations.generate import GenerateParams
        from lionagi.beta.operations.react import ReActParams

        full_instruction = instruction
        if guidance:
            full_instruction = f"{instruction}\n\nGuidance: {guidance}"

        if response_format is None:
            from lionagi.beta.operations.react import Analysis

            response_format = Analysis
        operable = Operable.from_structure(response_format)

        # Auto-filter global toolkit schemas by branch.resources when caller didn't pre-render.
        rendered_tools = tools
        if rendered_tools is None:
            from lionagi.tools import render_toolkit_schemas

            rendered_tools = render_toolkit_schemas(branch=branch)
            if not rendered_tools:
                rendered_tools = None

        gen_params = GenerateParams(
            primary=full_instruction,
            context=context,
            imodel=gen_model,
            tool_schemas=rendered_tools,
            imodel_kwargs=imodel_kwargs,
        )

        react_params = ReActParams(
            instruction=full_instruction,
            operable=operable,
            validator=self._validator,
            generate_params=gen_params,
            max_rounds=max_rounds,
            request_model=response_format,
            invoke_actions=bool(rendered_tools),
            persist=persist,
            parse_imodel=parse_model if parse_model != gen_model else gen_model,
        )

        op = await self.session.conduct("react", branch=branch, params=react_params)
        if op.execution.error is not None:
            raise op.execution.error
        return op.response

    def add_role(
        self,
        name: str,
        gen_model: str,
        parse_model: str | None = None,
        system: str | None = None,
        tools: list[str] | None = None,
    ) -> None:
        config = RoleConfig(
            gen_model=gen_model,
            parse_model=parse_model,
            system=system,
            tools=tools,
        )
        self.roles[name] = config

        caps = set()
        if tools:
            caps.add("action")
        branch = self.session.create_branch(
            name=name,
            capabilities=caps,
            resources=set(self._collect_model_names(config)),
        )
        self._role_branches[name] = name
        self._role_context_seen[name] = set()

        if system:
            from lionagi.beta.core.message import System
            from lionagi.protocols.messages import Message

            sys_msg = Message(content=System(system_message=system))
            self.session.add_message(sys_msg, branches=branch)

    def register_model(
        self,
        name: str,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        alias: str | None = None,
        **kwargs: Any,
    ) -> None:
        from lionagi.beta.resource import iModel

        model_id = model or name
        spec = f"{provider}/{model_id}" if provider else model_id
        imodel = iModel.from_provider(
            spec,
            api_key=api_key,
            name=name,
            **kwargs,
        )
        self.session.resources.register(imodel, update=True)

        if alias:
            self._model_aliases[alias] = name

    async def run_task(self, task_name: str, instruction: str, **kwargs: Any) -> Any:
        tasks = self.metadata.get("tasks", {})
        if task_name not in tasks:
            raise KeyError(f"Unknown task '{task_name}'. Available: {list(tasks.keys())}")

        task_def = tasks[task_name]
        role = task_def.get("role")
        if role is None:
            raise ValueError(f"Task '{task_name}' missing required 'role' field")
        assignment = task_def.get("assignment")
        mode = task_def.get("mode", "auto")
        max_rounds = task_def.get("max_rounds", 3)

        if "response_format" not in kwargs and "response_format" in task_def:
            kwargs["response_format"] = _resolve_type(task_def["response_format"])

        return await self.op(
            role,
            instruction,
            assignment=assignment,
            mode=mode,
            max_rounds=max_rounds,
            **kwargs,
        )

    async def run_tasks(
        self,
        instructions: dict[str, str],
        **kwargs: Any,
    ) -> dict[str, Any]:
        results = {}
        for task_name, instruction in instructions.items():
            results[task_name] = await self.run_task(task_name, instruction, **kwargs)
        return results

    @property
    def context(self) -> dict[str, Any]:
        """Read-only view of shared context."""
        return dict(self.shared_context)

    def __repr__(self) -> str:
        return (
            f"Conversation(roles={list(self.roles.keys())}, "
            f"context_keys={list(self.shared_context.keys())})"
        )


def _resolve_type(type_ref: str | type) -> type:
    if isinstance(type_ref, type):
        return type_ref

    if "." not in type_ref:
        raise ValueError(
            f"response_format '{type_ref}' must be a dotted path (e.g. 'mymodels.TechAnalysis')"
        )

    module_path, class_name = type_ref.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(f"'{type_ref}' resolved to {type(cls)}, expected a class")

    return cls


def create_conversation(
    *,
    config: SessionConfig | None = None,
    models: dict[str, dict[str, Any]] | None = None,
    **role_configs: dict[str, Any],
) -> Conversation:
    """Factory for Conversation; kwargs are role names mapped to RoleConfig dicts."""
    session_config = config or SessionConfig(auto_create_default_branch=False)

    if session_config.log_persist_dir is None:
        from pathlib import Path as _Path

        default_log_dir = _Path.home() / ".lionagi" / "logs"
        session_config = SessionConfig(
            **{
                **session_config.model_dump(),
                "log_persist_dir": str(default_log_dir),
            }
        )

    # Roles create their own branches; suppress the default "main" branch.
    if session_config.auto_create_default_branch:
        session_config = SessionConfig(
            **{
                **session_config.model_dump(),
                "auto_create_default_branch": False,
            }
        )

    session = Session(config=session_config)

    roles = {}
    for name, cfg in role_configs.items():
        if isinstance(cfg, dict):
            roles[name] = RoleConfig(**cfg)
        elif isinstance(cfg, RoleConfig):
            roles[name] = cfg
        else:
            raise TypeError(f"Role config for '{name}' must be dict or RoleConfig, got {type(cfg)}")

    convo = Conversation(session=session, roles=roles)

    if models:
        for model_name, model_cfg in models.items():
            model_cfg = dict(model_cfg)
            provider = model_cfg.pop("provider", "openai")
            model_id = model_cfg.pop("model", model_name)
            api_key = model_cfg.pop("api_key", None)
            convo.register_model(
                name=model_name,
                provider=provider,
                model=model_id,
                api_key=api_key,
                alias=model_name,
                **model_cfg,
            )

    return convo


def load_conversation(path: str) -> Conversation:
    """Load a Conversation from a YAML or TOML config file (use/roles/tasks sections)."""
    from pathlib import Path as _Path

    config_path = _Path(path)
    suffix = config_path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as err:
            raise ImportError("PyYAML required for YAML config: uv add pyyaml") from err

        with open(config_path) as f:
            data = yaml.safe_load(f)
    elif suffix == ".toml":
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .yaml, .yml, or .toml")

    if not isinstance(data, dict):
        raise ValueError(f"Invalid config file: expected YAML mapping, got {type(data).__name__}")

    use_section = data.get("use", {})
    models_config = use_section.get("models", {})
    roles_config = data.get("roles", {})
    tasks_config = data.get("tasks", {})

    log_dir = use_section.get("log_dir")
    session_config = None
    if log_dir:
        session_config = SessionConfig(
            auto_create_default_branch=False,
            log_persist_dir=log_dir,
        )

    convo = create_conversation(
        config=session_config,
        models=models_config,
        **roles_config,
    )

    convo.metadata["tasks"] = tasks_config

    return convo
