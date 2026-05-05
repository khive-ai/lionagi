# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from lionagi.beta.resource.flow import Flow
from lionagi.beta.session.session import Session
from lionagi.protocols.messages import Message


def test_beta_flow_accepts_canonical_message():
    flow = Flow(item_type=Message)
    message = Message(content={"text": "hello"})

    flow.add_item(message)

    assert flow.items[message.id] is message


def test_beta_flow_round_trips_canonical_message():
    flow = Flow(item_type=Message)
    message = Message(content={"text": "hello"})
    flow.add_item(message)

    restored = Flow.from_dict(flow.to_dict())
    restored_message = next(iter(restored.items))

    assert type(restored_message) is Message
    assert restored_message.content == {"text": "hello"}


def test_beta_session_accepts_canonical_message():
    session = Session()
    branch = session.default_branch
    message = Message(content={"text": "hello"})

    session.add_message(message, branches=branch)

    assert session.messages[message.id] is message
    assert message.id in branch
