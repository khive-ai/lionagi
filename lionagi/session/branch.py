# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Literal

import pandas as pd
from jinja2 import Template
from pydantic import BaseModel, Field, JsonValue, PrivateAttr

from lionagi.config import settings
from lionagi.fields import Instruct
from lionagi.libs.schema.as_readable import as_readable
from lionagi.models.field_model import FieldModel
from lionagi.protocols.action.tool import FuncTool, Tool, ToolRef
from lionagi.protocols.types import (
    ID,
    MESSAGE_FIELDS,
    ActionManager,
    ActionRequest,
    ActionResponse,
    AssistantResponse,
    Communicatable,
    Element,
    IDType,
    Instruction,
    Log,
    LogManagerConfig,
    Mail,
    Mailbox,
    MessageManager,
    MessageRole,
    Operative,
    Package,
    PackageCategory,
    Pile,
    Progression,
    Relational,
    RoledMessage,
    SenderRecipient,
    System,
)
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.types import iModel, iModelConfig, iModelManager
from lionagi.tools.base import LionTool
from lionagi.utils import UNDEFINED, alcall, bcall, copy, to_list

from .prompts import LION_SYSTEM_MESSAGE

__all__ = ("Branch",)


class Branch(Element, Communicatable, Relational):
    """
    Manages a conversation 'branch' with messages, tools, and iModels.

    The `Branch` class serves as a high-level interface or orchestrator that:
        - Handles message management (`MessageManager`).
        - Registers and invokes tools/actions (`ActionManager`).
        - Manages model instances (`iModelManager`).
        - Logs activity (`LogManager`).
        - Communicates via mailboxes (`Mailbox`).

    **Key responsibilities**:
        - Storing and organizing messages, including system instructions, user instructions, and model responses.
        - Handling asynchronous or synchronous execution of LLM calls and tool invocations.
        - Providing a consistent interface for “operate,” “chat,” “communicate,” “parse,” etc.

    Attributes:
        user (SenderRecipient | None):
            The user or "owner" of this branch (often tied to a session).
        name (str | None):
            A human-readable name for this branch.
        mailbox (Mailbox):
            A mailbox for sending and receiving `Package` objects to/from other branches.

    Note:
        Actual implementations for chat, parse, operate, etc., are referenced
        via lazy loading or modular imports. You typically won't need to
        subclass `Branch`, but you can instantiate it and call the
        associated methods for complex orchestrations.
    """

    user: SenderRecipient | None = Field(
        None,
        description=(
            "The user or sender of the branch, often a session object or "
            "an external user identifier. Not to be confused with the "
            "LLM API's user parameter."
        ),
    )

    name: str | None = Field(
        None,
        description="A human-readable name of the branch (optional).",
    )

    mailbox: Mailbox = Field(
        default_factory=Mailbox,
        exclude=True,
        description="Mailbox for cross-branch or external communication.",
    )

    _message_manager: MessageManager | None = PrivateAttr(None)
    _action_manager: ActionManager | None = PrivateAttr(None)
    _imodel_manager: iModelManager | None = PrivateAttr(None)

    def __init__(
        self,
        *,
        user: SenderRecipient = None,
        name: str | None = None,
        messages: Pile[RoledMessage] = None,  # message manager kwargs
        system: System | JsonValue = None,
        system_sender: SenderRecipient = None,
        tools: FuncTool | list[FuncTool] = None,  # ActionManager kwargs
        system_datetime: bool | str = None,
        use_lion_system_message: bool = False,
        log_configs: dict[Literal["api", "hook", "action"], dict] = None,
        imodels: iModelConfig = None,
        **kwargs,
    ):
        super().__init__(user=user, name=name, **kwargs)

        # --- MessageManager ---
        self._message_manager = MessageManager(messages=messages)

        if any(
            bool(x)
            for x in [
                system,
                system_datetime,
                use_lion_system_message,
            ]
        ):
            if ID.is_id(system):
                if not (id_ := ID.get_id(system)) in self._message_manager.messages:
                    raise ValueError(
                        f"System message with ID '{system}' not found in messages."
                    )
                system = self._message_manager.messages[id_]

            sy_ = self._message_manager.add_message(
                system=system or "",
                system_datetime=system_datetime,
                recipient=self.id,
                sender=system_sender or self.user or MessageRole.SYSTEM,
            )
            if use_lion_system_message:
                dev_sys = f"Developer Prompt: {str(system)}" if system else ""
                sy_.content["system_message"] = f"{LION_SYSTEM_MESSAGE}\n\n{dev_sys}".strip()

        imodels = imodels or {}
        if "chat" not in imodels:
            imodels["chat"] = iModel(
                provider=settings.LIONAGI_CHAT_PROVIDER,
                model=settings.LIONAGI_CHAT_MODEL,
            )
        if "parse" not in imodels:
            imodels["parse"] = imodels["chat"]

        self._imodel_manager = iModelManager(
            api_log_config=log_configs.get("api"),
            hook_log_config=log_configs.get("hook"),
            kw=imodels,
        )
        self._action_manager = ActionManager(
            action_log_config=log_configs.get("action"),
            args=tools if not isinstance(tools, dict) else None,
            kw=tools if isinstance(tools, dict) else None,
        )

    @property
    def system(self) -> System | None:
        """The system message/configuration, if any."""
        return self._message_manager.system

    @property
    def msgs(self) -> MessageManager:
        """Returns the associated MessageManager."""
        return self._message_manager

    @property
    def acts(self) -> ActionManager:
        """Returns the associated ActionManager for tool management."""
        return self._action_manager

    @property
    def mdls(self) -> iModelManager:
        """Returns the associated iModelManager."""
        return self._imodel_manager

    @property
    def messages(self) -> Pile[RoledMessage]:
        """Convenience property to retrieve all messages from MessageManager."""
        return self._message_manager.messages

    @property
    def tools(self) -> dict[str, Tool]:
        """
        All registered tools (actions) in the ActionManager,
        keyed by their tool names or IDs.
        """
        return self._action_manager.registry

    # -------------------------------------------------------------------------
    # Cloning
    # -------------------------------------------------------------------------
    async def aclone(self, sender: ID.Ref = None) -> "Branch":
        """
        Asynchronously clones this `Branch` with optional new sender ID.

        Args:
            sender (ID.Ref, optional):
                If provided, this ID is set as the sender for all cloned messages.

        Returns:
            Branch: A new branch instance, containing cloned state.
        """
        async with self.msgs.messages:
            return self.clone(sender)

    def clone(self, sender: ID.Ref = None) -> "Branch":
        """
        Clones this `Branch` synchronously, optionally updating the sender ID.

        Args:
            sender (ID.Ref, optional):
                If provided, all messages in the clone will have this sender ID.
                Otherwise, uses the current branch's ID.

        Raises:
            ValueError: If `sender` is not a valid ID.Ref.

        Returns:
            Branch: A new branch object with a copy of the messages, system info, etc.
        """
        if sender is not None:
            if not ID.is_id(sender):
                raise ValueError(
                    f"Cannot clone Branch: '{sender}' is not a valid sender ID."
                )
            sender = ID.get_id(sender)

        system = self.msgs.system.clone() if self.msgs.system else None
        tools = (
            list(self._action_manager.registry.values())
            if self._action_manager.registry
            else None
        )
        branch_clone = Branch(
            system=system,
            user=self.user,
            messages=[msg.clone() for msg in self.msgs.messages],
            tools=tools,
            metadata={"clone_from": self},
        )
        for message in branch_clone.msgs.messages:
            message.sender = sender or self.id
            message.recipient = branch_clone.id

        return branch_clone

    def _register_tool(self, tools: FuncTool | LionTool, update: bool = False):
        if isinstance(tools, type) and issubclass(tools, LionTool):
            tools = tools()
        if isinstance(tools, LionTool):
            tools = tools.to_tool()
        self._action_manager.register_tool(tools, update=update)

    def register_tools(
        self, tools: FuncTool | list[FuncTool] | LionTool, update: bool = False
    ):
        """
        Registers one or more tools in the ActionManager.

        Args:
            tools (FuncTool | list[FuncTool] | LionTool):
                A single tool or a list of tools to register.
            update (bool, optional):
                If `True`, updates existing tools with the same name.
        """
        tools = [tools] if not isinstance(tools, list) else tools
        for tool in tools:
            self._register_tool(tool, update=update)

    # -------------------------------------------------------------------------
    # Conversion / Serialization
    # -------------------------------------------------------------------------
    def to_df(self, *, progression: Progression = None) -> pd.DataFrame:
        """
        Convert branch messages into a `pandas.DataFrame`.

        Args:
            progression (Progression, optional):
                A custom message ordering. If `None`, uses the stored progression.

        Returns:
            pd.DataFrame: Each row represents a message, with columns defined by MESSAGE_FIELDS.
        """
        if progression is None:
            progression = self.msgs.progression

        msgs = [
            self.msgs.messages[i]
            for i in progression
            if i in self.msgs.messages
        ]
        p = Pile(collections=msgs)
        return p.to_df(columns=MESSAGE_FIELDS)

    # -------------------------------------------------------------------------
    # Mailbox Send / Receive
    # -------------------------------------------------------------------------
    def send(
        self,
        recipient: IDType,
        category: PackageCategory | None,
        item: Any,
        request_source: IDType | None = None,
    ) -> None:
        """
        Sends a `Package` (wrapped in a `Mail` object) to a specified recipient.

        Args:
            recipient (IDType):
                ID of the recipient branch or component.
            category (PackageCategory | None):
                The category/type of the package (e.g., 'message', 'tool', 'imodel').
            item (Any):
                The payload to send (e.g., a message, tool reference, model, etc.).
            request_source (IDType | None):
                The ID that prompted or requested this send operation (optional).
        """
        package = Package(
            category=category,
            item=item,
            request_source=request_source,
        )

        mail = Mail(
            sender=self.id,
            recipient=recipient,
            package=package,
        )
        self.mailbox.append_out(mail)

    def receive(
        self,
        sender: IDType,
        message: bool = False,
        tool: bool = False,
        imodel: bool = False,
    ) -> None:
        """
        Retrieves and processes mail from a given sender according to the specified flags.

        Args:
            sender (IDType):
                The ID of the mail sender.
            message (bool):
                If `True`, process packages categorized as "message".
            tool (bool):
                If `True`, process packages categorized as "tool".
            imodel (bool):
                If `True`, process packages categorized as "imodel".

        Raises:
            ValueError: If no mail exists from the specified sender,
                        or if a package is invalid for the chosen category.
        """
        sender = ID.get_id(sender)
        if sender not in self.mailbox.pending_ins.keys():
            raise ValueError(f"No mail or package found from sender: {sender}")

        skipped_requests = Progression()
        while self.mailbox.pending_ins[sender]:
            mail_id = self.mailbox.pending_ins[sender].popleft()
            mail: Mail = self.mailbox.pile_[mail_id]

            if mail.category == "message" and message:
                if not isinstance(mail.package.item, RoledMessage):
                    raise ValueError(
                        "Invalid message package: The item must be a `RoledMessage`."
                    )
                new_message = mail.package.item.clone()
                new_message.sender = mail.sender
                new_message.recipient = self.id
                self.msgs.messages.include(new_message)
                self.mailbox.pile_.pop(mail_id)

            elif mail.category == "tool" and tool:
                if not isinstance(mail.package.item, Tool):
                    raise ValueError(
                        "Invalid tool package: The item must be a `Tool` instance."
                    )
                self._action_manager.register_tools(mail.package.item)
                self.mailbox.pile_.pop(mail_id)

            elif mail.category == "imodel" and imodel:
                if not isinstance(mail.package.item, iModel):
                    raise ValueError(
                        "Invalid iModel package: The item must be an `iModel` instance."
                    )
                self._imodel_manager.register_imodel(
                    mail.package.item.name or "chat", mail.package.item
                )
                self.mailbox.pile_.pop(mail_id)

            else:
                # If the category doesn't match the flags or is unhandled
                skipped_requests.append(mail)

        # Requeue any skipped mail
        self.mailbox.pending_ins[sender] = skipped_requests
        if len(self.mailbox.pending_ins[sender]) == 0:
            self.mailbox.pending_ins.pop(sender)

    async def asend(
        self,
        recipient: IDType,
        category: PackageCategory | None,
        package: Any,
        request_source: IDType | None = None,
    ):
        """
        Async version of `send()`.

        Args:
            recipient (IDType):
                ID of the recipient branch or component.
            category (PackageCategory | None):
                The category/type of the package.
            package (Any):
                The item(s) to send (message/tool/model).
            request_source (IDType | None):
                The origin request ID (if any).
        """
        async with self.mailbox.pile_:
            self.send(recipient, category, package, request_source)

    async def areceive(
        self,
        sender: IDType,
        message: bool = False,
        tool: bool = False,
        imodel: bool = False,
    ) -> None:
        """
        Async version of `receive()`.

        Args:
            sender (IDType):
                The ID of the mail sender.
            message (bool):
                If `True`, process packages categorized as "message".
            tool (bool):
                If `True`, process packages categorized as "tool".
            imodel (bool):
                If `True`, process packages categorized as "imodel".
        """
        async with self.mailbox.pile_:
            self.receive(sender, message, tool, imodel)

    def receive_all(self) -> None:
        """
        Receives mail from all known senders without filtering.

        (Duplicate method included in your snippet; you may unify or remove.)
        """
        for key in self.mailbox.pending_ins:
            self.receive(key)

    def connect(
        self,
        provider: str = None,
        base_url: str = None,
        endpoint: str | Endpoint = None,
        endpoint_params: list[str] | None = None,
        api_key: str = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        imodel: iModel = None,
        name: str = None,
        request_options: type[BaseModel] = None,
        description: str = None,
        update: bool = False,
        **kwargs,
    ):
        if name in 
        
        if not imodel:
            imodel = iModel(
                provider=provider,
                base_url=base_url,
                endpoint=endpoint,
                endpoint_params=endpoint_params,
                api_key=api_key,
                queue_capacity=queue_capacity,
                capacity_refresh_time=capacity_refresh_time,
                interval=interval,
                limit_requests=limit_requests,
                limit_tokens=limit_tokens,
                **kwargs,
            )

        if not update and name in self.tools:
            raise ValueError(f"Tool with name '{name}' already exists.")

        async def _connect(**kwargs):
            """connect to an api endpoint"""
            if imodel.endpoint.config.name in self.mdls.registry:
                imodel = self.mdls.registry[imodel.endpoint.config.name]
            
            api_call = await imodel.invoke(**kwargs)
            
            
            self._log_manager.log(Log.create(api_call))
            return api_call.response

        _connect.__name__ = name or imodel.endpoint.name
        if description:
            _connect.__doc__ = description

        tool = Tool(
            func_callable=_connect,
            request_options=request_options or imodel.request_options,
        )
        self._action_manager.register_tools(tool, update=update)

    # -------------------------------------------------------------------------
    # Dictionary Conversion
    # -------------------------------------------------------------------------
    def to_dict(self):
        """
        Serializes the branch to a Python dictionary, including:
            - Messages
            - Logs
            - Chat/Parse models
            - System message
            - LogManager config
            - Metadata

        Returns:
            dict: A dictionary representing the branch's internal state.
        """
        meta = {}
        if ( cf := self.metadata.get("clone_from") ) is not None:
            # Provide some reference info about the source from which we cloned
            meta["clone_from"] = {
                "id": str(cf.id),
                "user": str(cf.user),
                "created_at": cf.created_at,
                "progression": [
                    str(i)
                    for i in cf.msgs.progression
                ],
            }
        meta.update(
            copy({k: v for k, v in self.metadata.items() if k != "clone_from"})
        )

        dict_ = super().to_dict()
        dict_["messages"] = self.messages.to_dict()
        dict_["imodels"] = {k: v.to_dict() for k, v in self.mdls.registry.items()}
        if self.system:
            dict_["system"] = str(self.system.id)
        dict_["metadata"] = meta
        return dict_

    @classmethod
    def from_dict(cls, data: dict):
        """
        Creates a `Branch` instance from a serialized dictionary.

        Args:
            data (dict):
                Must include (or optionally include) `messages`, `logs`,
                `chat_model`, `parse_model`, `system`, and `log_config`.

        Returns:
            Branch: A new `Branch` instance based on the deserialized data.
        """
        dict_ = {
            "messages": data.pop("messages", UNDEFINED),
            "logs": data.pop("logs", UNDEFINED),
            "imodels": data.pop("imodels", UNDEFINED),
            "system": data.pop("system", UNDEFINED),
            "log_configs": data.pop("log_configs", UNDEFINED),
        }
        params = {}

        # Merge in the rest of the data
        for k, v in data.items():
            # If the item is a dict with an 'id', we expand it
            if isinstance(v, dict) and "id" in v:
                params.update(v)
            else:
                params[k] = v

        params.update(dict_)
        # Remove placeholders (UNDEFINED) so we don't incorrectly assign them
        return cls(**{k: v for k, v in params.items() if v is not UNDEFINED})

    def dump_logs(self, clear: bool = True, persist_path=None):
        """
        Dumps the log to a file or clears it.

        Args:
            clear (bool, optional):
                If `True`, clears the log after dumping.
            persist_path (str, optional):
                The file path to save the log to.
        """
        self._log_manager.dump(clear=clear, persist_path=persist_path)

    async def adump_logs(self, clear: bool = True, persist_path=None):
        """
        Asynchronously dumps the log to a file or clears it.
        """
        await self._log_manager.adump(clear=clear, persist_path=persist_path)

    # -------------------------------------------------------------------------
    # Asynchronous Operations (chat, parse, operate, etc.)
    # -------------------------------------------------------------------------
    async def chat(
        self,
        instruction: Instruction | JsonValue = None,
        guidance: JsonValue = None,
        context: JsonValue = None,
        sender: ID.Ref = None,
        recipient: ID.Ref = None,
        request_fields: list[str] | dict[str, JsonValue] = None,
        response_format: type[BaseModel] | BaseModel = None,
        progression: Progression | list[ID[RoledMessage].ID] = None,
        imodel: iModel = None,
        tool_schemas: list[dict] = None,
        images: list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        plain_content: str = None,
        return_ins_res_message: bool = False,
        **kwargs,
    ) -> tuple[Instruction, AssistantResponse]:
        from lionagi.operations.chat.chat import chat

        return await chat(
            self,
            instruction=instruction,
            guidance=guidance,
            context=context,
            sender=sender,
            recipient=recipient,
            request_fields=request_fields,
            response_format=response_format,
            progression=progression,
            imodel=imodel or kwargs.pop("chat_model", None) or self.chat_model,
            tool_schemas=tool_schemas,
            images=images,
            image_detail=image_detail,
            plain_content=plain_content,
            return_ins_res_message=return_ins_res_message,
            **kwargs,
        )

    async def operate(
        self,
        *,
        instruct: Instruct = None,
        instruction: Instruction | JsonValue = None,
        guidance: JsonValue = None,
        context: JsonValue = None,
        sender: SenderRecipient = None,
        recipient: SenderRecipient = None,
        progression: Progression = None,
        chat_model: iModel = None,
        invoke_actions: bool = True,
        tool_schemas: list[dict] = None,
        images: list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        parse_model: iModel = None,
        skip_validation: bool = False,
        tools: ToolRef = None,
        operative: Operative = None,
        response_format: type[
            BaseModel
        ] = None,  # alias of operative.request_type
        actions: bool = False,
        reason: bool = False,
        action_kwargs: dict = None,
        action_strategy: Literal["sequential", "concurrent"] = "concurrent",
        verbose_action: bool = False,
        field_models: list[FieldModel] = None,
        exclude_fields: list | dict | None = None,
        handle_validation: Literal[
            "raise", "return_value", "return_none"
        ] = "return_value",
        include_token_usage_to_model: bool = False,
        **kwargs,
    ) -> list | BaseModel | None | dict | str:
        from lionagi.operations.operate.operate import operate

        return await operate(
            self,
            instruct=instruct,
            instruction=instruction,
            guidance=guidance,
            context=context,
            sender=sender,
            recipient=recipient,
            progression=progression,
            chat_model=chat_model,
            invoke_actions=invoke_actions,
            tool_schemas=tool_schemas,
            images=images,
            image_detail=image_detail,
            parse_model=parse_model,
            skip_validation=skip_validation,
            tools=tools,
            operative=operative,
            response_format=response_format,
            actions=actions,
            reason=reason,
            action_kwargs=action_kwargs,
            action_strategy=action_strategy,
            verbose_action=verbose_action,
            field_models=field_models,
            exclude_fields=exclude_fields,
            handle_validation=handle_validation,
            include_token_usage_to_model=include_token_usage_to_model,
            **kwargs,
        )

    async def communicate(
        self,
        instruction: Instruction | JsonValue = None,
        *,
        guidance: JsonValue = None,
        context: JsonValue = None,
        plain_content: str = None,
        sender: SenderRecipient = None,
        recipient: SenderRecipient = None,
        progression: ID.IDSeq = None,
        response_format: type[BaseModel] = None,
        request_fields: dict | list[str] = None,
        chat_model: iModel = None,
        parse_model: iModel = None,
        skip_validation: bool = False,
        images: list = None,
        image_detail: Literal["low", "high", "auto"] = None,
        num_parse_retries: int = 3,
        clear_messages: bool = False,
        include_token_usage_to_model: bool = False,
        **kwargs,
    ):
        from lionagi.operations.communicate.communicate import communicate

        return await communicate(
            self,
            instruction=instruction,
            guidance=guidance,
            context=context,
            plain_content=plain_content,
            sender=sender,
            recipient=recipient,
            progression=progression,
            response_format=response_format,
            request_fields=request_fields,
            chat_model=chat_model,
            parse_model=parse_model,
            skip_validation=skip_validation,
            images=images,
            image_detail=image_detail,
            num_parse_retries=num_parse_retries,
            clear_messages=clear_messages,
            include_token_usage_to_model=include_token_usage_to_model,
            **kwargs,
        )

    async def instruct(
        self,
        instruct: Instruct,
        /,
        **kwargs,
    ):
        from lionagi.operations.instruct.instruct import instruct as _ins

        return await _ins(self, instruct, **kwargs)

    async def ReAct(
        self,
        instruct: Instruct | dict[str, Any],
        interpret: bool = False,
        interpret_domain: str | None = None,
        interpret_style: str | None = None,
        interpret_sample: str | None = None,
        interpret_model: str | None = None,
        interpret_kwargs: dict | None = None,
        tools: Any = None,
        tool_schemas: Any = None,
        response_format: type[BaseModel] | BaseModel = None,
        intermediate_response_options: list[BaseModel] | BaseModel = None,
        intermediate_listable: bool = False,
        reasoning_effort: Literal["low", "medium", "high"] = None,
        extension_allowed: bool = True,
        max_extensions: int | None = 3,
        response_kwargs: dict | None = None,
        display_as: Literal["json", "yaml"] = "yaml",
        return_analysis: bool = False,
        analysis_model: iModel | None = None,
        verbose: bool = False,
        verbose_length: int = None,
        include_token_usage_to_model: bool = True,
        **kwargs,
    ):
        from lionagi.operations.ReAct.ReAct import ReAct

        return await ReAct(
            self,
            instruct,
            interpret=interpret,
            interpret_domain=interpret_domain,
            interpret_style=interpret_style,
            interpret_sample=interpret_sample,
            interpret_kwargs=interpret_kwargs,
            tools=tools,
            tool_schemas=tool_schemas,
            response_format=response_format,
            extension_allowed=extension_allowed,
            max_extensions=max_extensions,
            response_kwargs=response_kwargs,
            return_analysis=return_analysis,
            analysis_model=analysis_model,
            verbose_action=verbose,
            verbose_analysis=verbose,
            verbose_length=verbose_length,
            interpret_model=interpret_model,
            intermediate_response_options=intermediate_response_options,
            intermediate_listable=intermediate_listable,
            reasoning_effort=reasoning_effort,
            display_as=display_as,
            include_token_usage_to_model=include_token_usage_to_model,
            **kwargs,
        )

    async def ReActStream(
        self,
        instruct: Instruct | dict[str, Any],
        interpret: bool = False,
        interpret_domain: str | None = None,
        interpret_style: str | None = None,
        interpret_sample: str | None = None,
        interpret_model: str | None = None,
        interpret_kwargs: dict | None = None,
        tools: Any = None,
        tool_schemas: Any = None,
        response_format: type[BaseModel] | BaseModel = None,
        intermediate_response_options: list[BaseModel] | BaseModel = None,
        intermediate_listable: bool = False,
        reasoning_effort: Literal["low", "medium", "high"] = None,
        extension_allowed: bool = True,
        max_extensions: int | None = 3,
        response_kwargs: dict | None = None,
        analysis_model: iModel | None = None,
        verbose: bool = False,
        display_as: Literal["json", "yaml"] = "yaml",
        verbose_length: int = None,
        include_token_usage_to_model: bool = True,
        **kwargs,
    ) -> AsyncGenerator:
        from lionagi.operations.ReAct.ReAct import ReActStream

        async for result in ReActStream(
            self,
            instruct,
            interpret=interpret,
            interpret_domain=interpret_domain,
            interpret_style=interpret_style,
            interpret_sample=interpret_sample,
            interpret_model=interpret_model,
            interpret_kwargs=interpret_kwargs,
            tools=tools,
            tool_schemas=tool_schemas,
            response_format=response_format,
            intermediate_response_options=intermediate_response_options,
            intermediate_listable=intermediate_listable,
            reasoning_effort=reasoning_effort,
            extension_allowed=extension_allowed,
            max_extensions=max_extensions,
            response_kwargs=response_kwargs,
            analysis_model=analysis_model,
            verbose_analysis=True,
            display_as=display_as,
            verbose_length=verbose_length,
            include_token_usage_to_model=include_token_usage_to_model,
            **kwargs,
        ):
            analysis, str_ = result
            if verbose:
                str_ += "\n---------\n"
                as_readable(str_, md=True, display_str=True)
            yield analysis


# File: lionagi/session/branch.py
