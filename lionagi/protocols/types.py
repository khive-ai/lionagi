# Copyright (c) 2023 - 2025, HaiyangLi <quantocean.li at gmail dot com>
#
# SPDX-License-Identifier: Apache-2.0

from ._concepts import Collective, Communicatable, Condition, Manager
from ._concepts import Observable as LegacyObservable
from ._concepts import Observer, Ordering, Relational, Sendable
from .action.manager import ActionManager, FunctionCalling, Tool, ToolRef
from .contracts import Observable, ObservableProto
from .forms.flow import FlowDefinition, FlowStep
from .forms.report import BaseForm, Form, Report
from .generic.element import ID, Element, IDError, IDType, validate_order
from .generic.event import Event, EventStatus, Execution
from .generic.log import (
    DataLogger,
    DataLoggerConfig,
    Log,
    LogManager,
    LogManagerConfig,
)
from .generic.pile import Pile, to_list_type
from .generic.processor import Executor, Processor
from .generic.progression import Progression, prog
from .graph.edge import EdgeCondition
from .graph.graph import Edge, Graph, Node
from .ids import canonical_id, to_uuid
from .mail.exchange import Exchange, Mail, Mailbox, Package, PackageCategory
from .mail.manager import MailManager
from .messages.base import (
    MESSAGE_FIELDS,
    MessageField,
    MessageFlag,
    MessageRole,
    validate_sender_recipient,
)
from .messages.manager import (
    ActionRequest,
    ActionResponse,
    AssistantResponse,
    Instruction,
    MessageManager,
    RoledMessage,
    SenderRecipient,
    System,
)
from .operatives.step import Operative, Step

__all__ = (
    "Collective",
    "Communicatable",
    "Condition",
    "Manager",
    "Observable",  # V1 Protocol (preferred)
    "ObservableProto",  # Explicit V1 Protocol name
    "LegacyObservable",  # V0 ABC (deprecated)
    "Observer",
    "Ordering",
    "Relational",
    "Sendable",
    "canonical_id",  # V0/V1 bridge utility
    "to_uuid",  # ID conversion utility
    "ID",
    "Element",
    "IDError",
    "IDType",
    "validate_order",
    "Event",
    "EventStatus",
    "Execution",
    "Log",
    "LogManager",
    "LogManagerConfig",
    "Pile",
    "to_list_type",
    "Executor",
    "Processor",
    "Progression",
    "prog",
    "EdgeCondition",
    "Edge",
    "Graph",
    "Node",
    "Exchange",
    "Mail",
    "Mailbox",
    "Package",
    "PackageCategory",
    "MESSAGE_FIELDS",
    "MessageField",
    "MessageFlag",
    "MessageRole",
    "validate_sender_recipient",
    "ActionRequest",
    "ActionResponse",
    "AssistantResponse",
    "Instruction",
    "MessageManager",
    "RoledMessage",
    "SenderRecipient",
    "System",
    "FlowDefinition",
    "FlowStep",
    "BaseForm",
    "Form",
    "Report",
    "Operative",
    "Step",
    "ActionManager",
    "Tool",
    "FunctionCalling",
    "ToolRef",
    "MailManager",
    "DataLogger",
    "DataLoggerConfig",
)
