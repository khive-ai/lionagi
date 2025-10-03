# Copyright (c) 2023-2025, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncGenerator, Callable
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, JsonValue, PrivateAttr, field_serializer
from typing_extensions import deprecated

from lionagi.config import settings
from lionagi.ln.async_call import AlcallParams
from lionagi.ln.fuzzy import FuzzyMatchKeysParams
from lionagi.ln.types import Unset
from lionagi.models.field_model import FieldModel
from lionagi.operations.manager import OperationManager
from lionagi.protocols.action.manager import ActionManager
from lionagi.protocols.action.tool import FuncTool, Tool
from lionagi.protocols.types import (
    ID,
    MESSAGE_FIELDS,
    ActionRequest,
    ActionResponse,
    AssistantResponse,
    Communicatable,
    Element,
    IDType,
    Instruction,
    Log,
    LogManager,
    LogManagerConfig,
    Mail,
    Mailbox,
    MessageManager,
    MessageRole,
    PackageCategory,
    Pile,
    Progression,
    Relational,
    RoledMessage,
    SenderRecipient,
    System,
)
from lionagi.service.connections.endpoint import Endpoint
from lionagi.service.types import iModel, iModelManager
from lionagi.tools.base import LionTool
from lionagi.utils import copy

from .ops.types import (
    ActionContext,
    ChatContext,
    HandleValidation,
    InterpretContext,
    ParseContext,
)
from .prompts import LION_SYSTEM_MESSAGE

__all__ = ("Branch",)


_DEFAULT_ALCALL_PARAMS = None


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
    _log_manager: LogManager | None = PrivateAttr(None)
    _operation_manager: OperationManager | None = PrivateAttr(None)

    def __init__(
        self,
        *,
        user: "SenderRecipient" = None,
        name: str | None = None,
        messages: Pile[RoledMessage] = None,  # message manager kwargs
        system: System | JsonValue = None,
        system_sender: "SenderRecipient" = None,
        chat_model: iModel | dict = None,  # iModelManager kwargs
        parse_model: iModel | dict = None,
        imodel: iModel = None,  # deprecated, alias of chat_model
        tools: FuncTool | list[FuncTool] = None,  # ActionManager kwargs
        log_config: LogManagerConfig | dict = None,  # LogManager kwargs
        system_datetime: bool | str = None,
        system_template=None,
        system_template_context: dict = None,
        logs: Pile[Log] = None,
        use_lion_system_message: bool = False,
        **kwargs,
    ):
        """
        Initializes a `Branch` with references to managers and an optional mailbox.

        Args:
            user (SenderRecipient, optional):
                The user or sender context for this branch.
            name (str | None, optional):
                A human-readable name for this branch.
            messages (Pile[RoledMessage], optional):
                Initial messages for seeding the MessageManager.
            system (System | JsonValue, optional):
                Optional system-level configuration or message for the LLM.
            system_sender (SenderRecipient, optional):
                Sender to attribute to the system message if it is added.
            chat_model (iModel, optional):
                The primary "chat" iModel for conversation. If not provided,
                uses default provider and model from settings.
            parse_model (iModel, optional):
                The "parse" iModel for structured data parsing.
                Defaults to chat_model if not provided.
            imodel (iModel, optional):
                Deprecated. Alias for `chat_model`.
            tools (FuncTool | list[FuncTool], optional):
                Tools or a list of tools for the ActionManager.
            log_config (LogManagerConfig | dict, optional):
                Configuration dict or object for the LogManager.
            system_datetime (bool | str, optional):
                Whether to include timestamps in system messages (True/False)
                or a string format for datetime.
            system_template (jinja2.Template | str, optional):
                Optional Jinja2 template for system messages.
            system_template_context (dict, optional):
                Context for rendering the system template.
            logs (Pile[Log], optional):
                Existing logs to seed the LogManager.
            use_lion_system_message (bool, optional):
                If `True`, uses the Lion system message for the branch.
            **kwargs:
                Additional parameters passed to `Element` parent init.
        """
        super().__init__(user=user, name=name, **kwargs)

        # --- MessageManager ---
        from lionagi.protocols.messages.manager import MessageManager

        self._message_manager = MessageManager(messages=messages)

        if any(
            bool(x)
            for x in [
                system,
                system_datetime,
                system_template,
                system_template_context,
                use_lion_system_message,
            ]
        ):
            if use_lion_system_message:
                system = f"Developer Prompt: {str(system)}" if system else ""
                system = (LION_SYSTEM_MESSAGE + "\n\n" + system).strip()

            self._message_manager.add_message(
                system=system,
                system_datetime=system_datetime,
                template=system_template,
                template_context=system_template_context,
                recipient=self.id,
                sender=system_sender or self.user or MessageRole.SYSTEM,
            )

        chat_model = chat_model or imodel
        if not chat_model:
            chat_model = iModel(
                provider=settings.LIONAGI_CHAT_PROVIDER,
                model=settings.LIONAGI_CHAT_MODEL,
            )
        if not parse_model:
            parse_model = chat_model

        if isinstance(chat_model, dict):
            chat_model = iModel.from_dict(chat_model)
        if isinstance(parse_model, dict):
            parse_model = iModel.from_dict(parse_model)

        self._imodel_manager = iModelManager(
            chat=chat_model, parse=parse_model
        )

        # --- ActionManager ---
        self._action_manager = ActionManager()
        if tools:
            self.register_tools(tools)

        # --- LogManager ---
        if log_config:
            if isinstance(log_config, dict):
                log_config = LogManagerConfig(**log_config)
            self._log_manager = LogManager.from_config(log_config, logs=logs)
        else:
            self._log_manager = LogManager(**settings.LOG_CONFIG, logs=logs)

        self._operation_manager = OperationManager()

    # -------------------------------------------------------------------------
    # Properties to expose managers and core data
    # -------------------------------------------------------------------------
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
    def logs(self) -> Pile[Log]:
        """Convenience property to retrieve all logs from the LogManager."""
        return self._log_manager.logs

    @property
    def chat_model(self) -> iModel:
        """
        The primary "chat" model (`iModel`) used for conversational LLM calls.
        """
        return self._imodel_manager.chat

    @chat_model.setter
    def chat_model(self, value: iModel) -> None:
        """
        Sets the primary "chat" model in the iModelManager.

        Args:
            value (iModel): The new chat model to register.
        """
        self._imodel_manager.register_imodel("chat", value)

    @property
    def parse_model(self) -> iModel:
        """The "parse" model (`iModel`) used for structured data parsing."""
        return self._imodel_manager.parse

    @parse_model.setter
    def parse_model(self, value: iModel) -> None:
        """
        Sets the "parse" model in the iModelManager.

        Args:
            value (iModel): The new parse model to register.
        """
        self._imodel_manager.register_imodel("parse", value)

    @property
    def tools(self) -> dict[str, Tool]:
        """
        All registered tools (actions) in the ActionManager,
        keyed by their tool names or IDs.
        """
        return self._action_manager.registry

    def get_operation(self, operation: str) -> Callable | None:
        if hasattr(self, operation):
            return getattr(self, operation)
        return self._operation_manager.registry.get(operation)

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

    @field_serializer("user")
    def _serialize_user(self, v):
        return str(v) if v else None

    # -------------------------------------------------------------------------
    # Conversion / Serialization
    # -------------------------------------------------------------------------
    def to_df(self, *, progression: Progression = None):
        """
        Convert branch messages into a `pandas.DataFrame`.

        Args:
            progression (Progression, optional):
                A custom message ordering. If `None`, uses the stored progression.

        Returns:
            pd.DataFrame: Each row represents a message, with columns defined by MESSAGE_FIELDS.
        """
        from lionagi.protocols.generic.pile import Pile

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
        category: Optional["PackageCategory"],
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
        from lionagi.protocols.mail.package import Package

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
        endpoint: str | Endpoint = "chat",
        endpoint_params: list[str] | None = None,
        api_key: str = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        interval: float | None = None,
        limit_requests: int = None,
        limit_tokens: int = None,
        invoke_with_endpoint: bool = False,
        imodel: iModel = None,
        name: str = None,
        request_options: type[BaseModel] = None,
        description: str = None,
        update: bool = False,
        **kwargs,
    ):
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
                invoke_with_endpoint=invoke_with_endpoint,
                **kwargs,
            )

        if not update and name in self.tools:
            raise ValueError(f"Tool with name '{name}' already exists.")

        async def _connect(**kwargs):
            """connect to an api endpoint"""
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
        if "clone_from" in self.metadata:
            # Provide some reference info about the source from which we cloned
            meta["clone_from"] = {
                "id": str(self.metadata["clone_from"].id),
                "user": str(self.metadata["clone_from"].user),
                "created_at": self.metadata["clone_from"].created_at,
                "progression": [
                    str(i)
                    for i in self.metadata["clone_from"].msgs.progression
                ],
            }
        meta.update(
            copy({k: v for k, v in self.metadata.items() if k != "clone_from"})
        )

        dict_ = super().to_dict()
        dict_["messages"] = self.messages.to_dict()
        dict_["logs"] = self.logs.to_dict()
        dict_["chat_model"] = self.chat_model.to_dict()
        dict_["parse_model"] = self.parse_model.to_dict()
        if self.system:
            dict_["system"] = self.system.to_dict()
        dict_["log_config"] = self._log_manager._config.model_dump()
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
            "messages": data.pop("messages", Unset),
            "logs": data.pop("logs", Unset),
            "chat_model": data.pop("chat_model", Unset),
            "parse_model": data.pop("parse_model", Unset),
            "system": data.pop("system", Unset),
            "log_config": data.pop("log_config", Unset),
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
        # Remove placeholders (Unset) so we don't incorrectly assign them
        return cls(**{k: v for k, v in params.items() if v is not Unset})

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

    async def chat(
        self,
        instruction: JsonValue | Instruction,
        chat_ctx: ChatContext,
        return_ins_res_message: bool = False,
    ) -> str | tuple[Instruction, AssistantResponse]:
        """
        Execute a chat operation with the given instruction and context.

        Args:
            instruction: The instruction to send
            chat_ctx: Chat context with model and configuration
            return_ins_res_message: If True, return (Instruction, AssistantResponse) tuple

        Returns:
            str or tuple[Instruction, AssistantResponse]
        """
        from .ops.chat import chat

        return await chat(
            self,
            instruction,
            chat_ctx,
            return_ins_res_message,
        )

    async def parse(
        self,
        text: str,
        response_format: type[BaseModel] | dict,
        fuzzy_match_params: FuzzyMatchKeysParams | dict = None,
        handle_validation: HandleValidation = "raise",
        alcall_params: AlcallParams | dict | None = None,
        parse_model: "iModel" = None,
        return_res_message: bool = False,
    ):
        """
        Parse text into structured format.

        Args:
            text: Text to parse
            response_format: Target format (BaseModel or dict)
            fuzzy_match_params: Fuzzy matching parameters
            handle_validation: How to handle validation errors
            alcall_params: Async call parameters
            parse_model: Model to use for parsing
            return_res_message: If True, return (result, response_message) tuple

        Returns:
            Parsed result or tuple if return_res_message=True
        """
        from .ops.parse import parse

        return await parse(
            self,
            text,
            response_format=response_format,
            fuzzy_match_params=fuzzy_match_params,
            handle_validation=handle_validation,
            alcall_params=alcall_params,
            parse_model=parse_model or self.parse_model,
            return_res_message=return_res_message,
        )

    async def operate(
        self,
        instruction: JsonValue | Instruction | dict,
        chat_ctx: ChatContext,
        action_ctx: ActionContext | None = None,
        parse_ctx: ParseContext | None = None,
        reason: bool = False,
        field_models: list[FieldModel] | None = None,
        handle_validation: HandleValidation = "raise",
        invoke_actions: bool = True,
        clear_messages: bool = False,
    ):
        """
        Execute a structured operation with optional actions.

        Args:
            instruction: The instruction to execute
            chat_ctx: Chat context with model and configuration
            action_ctx: Action context for tool invocation
            parse_ctx: Parse context for response parsing
            reason: Add reasoning field
            field_models: Additional field models
            handle_validation: How to handle validation errors
            invoke_actions: Whether to invoke actions
            clear_messages: Whether to clear messages before operation

        Returns:
            Parsed operation result
        """
        from .ops.operate import operate

        return await operate(
            self,
            instruction=instruction,
            chat_ctx=chat_ctx,
            action_ctx=action_ctx,
            parse_ctx=parse_ctx,
            reason=reason,
            field_models=field_models,
            handle_validation=handle_validation,
            invoke_actions=invoke_actions,
            clear_messages=clear_messages,
        )

    async def communicate(
        self,
        instruction: JsonValue | Instruction,
        chat_ctx: ChatContext,
        parse_ctx: ParseContext | None = None,
        clear_messages: bool = False,
    ):
        """
        Chat and optionally parse the response.

        Args:
            instruction: The instruction to send
            chat_ctx: Chat context with model and configuration
            parse_ctx: Parse context for response parsing
            clear_messages: Whether to clear messages before operation

        Returns:
            Response string or parsed result
        """
        from .ops.communicate import communicate

        return await communicate(
            self,
            instruction,
            chat_ctx,
            parse_ctx,
            clear_messages,
        )

    async def _act(
        self,
        action_request: BaseModel | dict,
        suppress_errors: bool = False,
        verbose_action: bool = False,
    ):
        from .ops.act import _act

        return await _act(
            self,
            action_request,
            suppress_errors=suppress_errors,
            verbose_action=verbose_action,
        )

    async def act(
        self,
        action_request: list | ActionRequest | BaseModel | dict,
        *,
        strategy: Literal["concurrent", "sequential"] = "concurrent",
        verbose_action: bool = False,
        suppress_errors: bool = True,
        call_params: AlcallParams = None,
    ) -> list[ActionResponse]:
        from .ops.act import act

        return await act(
            self,
            action_request,
            strategy=strategy,
            verbose_action=verbose_action,
            suppress_errors=suppress_errors,
            call_params=call_params or _DEFAULT_ALCALL_PARAMS,
        )

    async def ReAct(
        self,
        instruction: JsonValue | Instruction,
        chat_ctx: ChatContext,
        action_ctx: ActionContext | None = None,
        parse_ctx: ParseContext | None = None,
        intp_ctx: InterpretContext | bool = None,
        resp_ctx: dict | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        reason: bool = False,
        field_models: list[FieldModel] | None = None,
        handle_validation: HandleValidation = "raise",
        invoke_actions: bool = True,
        clear_messages=False,
        intermediate_response_options: (
            type[BaseModel] | list[type[BaseModel]]
        ) = None,
        intermediate_listable: bool = False,
        intermediate_nullable: bool = False,
        max_extensions: int | None = 0,
        extension_allowed: bool = True,
        verbose_analysis: bool = False,
        display_as: Literal["yaml", "json"] = "yaml",
        verbose_length: int = None,
        continue_after_failed_response: bool = False,
        return_analysis: bool = False,
        stream: bool = False,
    ):
        """
        Execute ReAct (Reasoning-Acting) operation.

        Args:
            instruction: The instruction to execute
            chat_ctx: Chat context with model and configuration
            action_ctx: Action context for tool invocation
            parse_ctx: Parse context for response parsing
            intp_ctx: Interpretation context or bool to enable interpretation
            resp_ctx: Response context for final answer
            reasoning_effort: Reasoning effort level (low/medium/high)
            reason: Add reasoning field
            field_models: Additional field models
            handle_validation: How to handle validation errors
            invoke_actions: Whether to invoke actions
            clear_messages: Whether to clear messages before operation
            intermediate_response_options: Options for intermediate responses
            intermediate_listable: Make intermediate responses listable
            intermediate_nullable: Make intermediate responses nullable
            max_extensions: Maximum number of extensions allowed
            extension_allowed: Whether extensions are allowed
            verbose_analysis: Show verbose analysis output
            display_as: Display format (yaml/json)
            verbose_length: Length limit for verbose output
            continue_after_failed_response: Continue after failed responses
            return_analysis: Return full analysis list
            stream: If True, return async generator streaming results

        Returns:
            Final result, list of analyses if return_analysis=True, or AsyncGenerator if stream=True
        """
        params = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "stream"] and v is not None
        }

        if stream:
            from .ops.ReAct import ReActStream

            async for result in ReActStream(self, **params):
                yield result
        else:
            from .ops.ReAct import ReAct

            return await ReAct(self, **params)

    @deprecated(
        "Use ReAct(stream=True) instead. This method will be removed in a future version."
    )
    async def ReActStream(
        self,
        instruction: JsonValue | Instruction,
        chat_ctx: ChatContext,
        action_ctx: ActionContext | None = None,
        parse_ctx: ParseContext | None = None,
        intp_ctx: InterpretContext | bool = None,
        resp_ctx: dict | None = None,
        reasoning_effort: Literal["low", "medium", "high"] | None = None,
        reason: bool = False,
        field_models: list[FieldModel] | None = None,
        handle_validation: HandleValidation = "raise",
        invoke_actions: bool = True,
        clear_messages=False,
        intermediate_response_options: (
            type[BaseModel] | list[type[BaseModel]]
        ) = None,
        intermediate_listable: bool = False,
        intermediate_nullable: bool = False,
        max_extensions: int | None = 0,
        extension_allowed: bool = True,
        verbose_analysis: bool = False,
        display_as: Literal["yaml", "json"] = "yaml",
        verbose_length: int = None,
        continue_after_failed_response: bool = False,
    ) -> AsyncGenerator:
        """
        Stream ReAct (Reasoning-Acting) operation results.

        .. deprecated::
            Use ReAct(stream=True) instead. This method will be removed in a future version.

        Args:
            Same as ReAct()

        Yields:
            Analysis results at each step
        """
        params = {
            k: v
            for k, v in locals().items()
            if k not in ["self"] and v is not None
        }

        from .ops.ReAct import ReActStream

        async for result in ReActStream(self, **params):
            yield result


# File: lionagi/session/branch.py
