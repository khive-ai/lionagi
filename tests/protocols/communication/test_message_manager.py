import pytest
from pydantic import BaseModel

from lionagi.protocols.types import (
    ActionRequest,
    ActionResponse,
    AssistantResponse,
    Instruction,
    MessageManager,
    MessageRole,
    Pile,
    System,
)


class RequestModel(BaseModel):
    """Model for testing request fields"""

    name: str
    age: int


@pytest.fixture
def message_manager():
    """Fixture providing a clean MessageManager instance"""
    return MessageManager()


def test_message_manager_initialization():
    """Test basic initialization of MessageManager"""
    manager = MessageManager()
    assert isinstance(manager.messages, Pile)
    assert not manager.messages
    assert manager.system is None


def test_message_manager_initialization_with_messages():
    """Test MessageManager initialization with existing messages"""
    instruction = Instruction(content={"instruction": "Test"})
    manager = MessageManager(messages=[instruction])

    assert len(manager.messages) == 1
    assert instruction in manager.messages


def test_message_manager_initialization_with_dict_messages():
    """Test MessageManager initialization with dict messages"""
    instruction_dict = Instruction(content={"instruction": "Test"}).to_dict()
    manager = MessageManager(messages=[instruction_dict])

    assert len(manager.messages) == 1


def test_message_manager_with_system():
    """Test MessageManager initialization with system message"""
    system = System(content={"system_message": "Test system"})
    manager = MessageManager(system=system)

    assert manager.system == system
    assert system in manager.messages
    assert len(manager.messages) == 1


def test_message_manager_with_invalid_system():
    """Test MessageManager with invalid system type"""
    with pytest.raises(
        ValueError, match="System message must be a System instance"
    ):
        MessageManager(system="not a system object")


def test_set_system():
    """Test setting and replacing system message"""
    manager = MessageManager()
    system1 = System(content={"system_message": "System 1"})
    system2 = System(content={"system_message": "System 2"})

    # Set first system
    manager.set_system(system1)
    assert manager.system == system1
    assert system1 in manager.messages
    assert len(manager.messages) == 1

    # Replace with second system
    manager.set_system(system2)
    assert manager.system == system2
    assert system1 not in manager.messages
    assert system2 in manager.messages
    assert len(manager.messages) == 1


def test_create_instruction_basic():
    """Test creating basic instruction message"""
    instruction = Instruction(
        content={"instruction": "Test instruction"},
        sender="user",
        recipient="assistant",
    )

    assert isinstance(instruction, Instruction)
    assert instruction.content.instruction == "Test instruction"
    assert instruction.sender == "user"
    assert instruction.recipient == "assistant"


def test_create_instruction_with_all_params():
    """Test creating instruction with all parameters"""
    instruction = Instruction(
        content={
            "instruction": "Test instruction",
            "context": {"test": "context"},
            "guidance": "Test guidance",
            "images": ["image1.png"],
            "request_fields": {"field1": "value1"},
            "plain_content": "Plain text",
            "image_detail": "high",
            "response_format": RequestModel,
            "tool_schemas": {"tool1": {}},
        },
        sender="user",
        recipient="assistant",
    )

    assert isinstance(instruction, Instruction)
    assert instruction.content.instruction == "Test instruction"
    assert instruction.content.guidance == "Test guidance"
    assert instruction.sender == "user"
    assert instruction.recipient == "assistant"
    assert instruction.content.image_detail == "high"
    # response_format as BaseModel gets converted to response_schema + auto-generated format
    assert instruction.content.response_schema is not None
    assert isinstance(instruction.content.response_schema, dict)
    assert instruction.content.response_format is not None
    assert isinstance(instruction.content.response_format, dict)


def test_create_instruction_update_existing():
    """Test updating existing instruction"""
    instruction = Instruction(content={"instruction": "Original"})
    instruction.update(guidance="New guidance", sender="user")

    assert instruction.content.guidance == "New guidance"
    assert instruction.sender == MessageRole.USER


def test_create_instruction_default_context_extend(message_manager):
    """Updating instruction without handle flag should extend context."""
    instruction = message_manager.create_instruction(
        instruction="First",
        context=["base"],
    )

    message_manager.create_instruction(
        instruction=instruction,
        context=["new"],
    )

    assert instruction.content.context == ["base", "new"]


def test_create_instruction_context_replace(message_manager):
    """handle_context='replace' should overwrite existing context."""
    instruction = message_manager.create_instruction(
        instruction="First",
        context=["base"],
    )

    message_manager.create_instruction(
        instruction=instruction,
        context=["new"],
        handle_context="replace",
    )

    assert instruction.content.context == ["new"]


def test_create_instruction_response_format_instance(message_manager):
    """BaseModel instances for response_format should be accepted."""

    class InstanceModel(BaseModel):
        value: int

    instruction = message_manager.create_instruction(
        instruction="Test",
        response_format=InstanceModel(value=3),
    )

    assert instruction.content.response_schema["title"] == "InstanceModel"
    assert instruction.content.response_format == {"value": 3}


def test_add_message_instruction_context_extend(message_manager):
    """add_message defaults to extending context when updating."""
    initial = message_manager.add_message(
        instruction="Original",
        context=["base"],
    )

    message_manager.add_message(
        instruction=initial,
        context=["new"],
    )

    assert initial.content.context == ["base", "new"]


def test_add_message_instruction_context_replace(message_manager):
    """add_message can replace context when handle_context='replace'."""
    initial = message_manager.add_message(
        instruction="Original",
        context=["base"],
    )

    message_manager.add_message(
        instruction=initial,
        context=["new"],
        handle_context="replace",
    )

    assert initial.content.context == ["new"]


def test_create_system_basic():
    """Test creating basic system message"""
    system = System(
        content={"system_message": "Test system"},
        sender="system",
        recipient="user",
    )

    assert isinstance(system, System)
    assert system.content.system_message == "Test system"
    assert system.sender == "system"
    assert system.recipient == "user"


def test_create_system_with_datetime():
    """Test creating system message with datetime"""
    system = System(
        content={"system_message": "Test system", "system_datetime": True}
    )

    assert isinstance(system, System)
    # System datetime should be included in the message


def test_create_system_update_existing():
    """Test updating existing system message"""
    system = System(content={"system_message": "Original"})
    system.update(sender="system")

    assert system.sender == MessageRole.SYSTEM


def test_create_assistant_response_basic():
    """Test creating basic assistant response"""
    response = AssistantResponse(
        content={"assistant_response": "Test response"},
        sender="assistant",
        recipient="user",
    )

    assert isinstance(response, AssistantResponse)
    assert response.content.assistant_response == "Test response"
    assert response.sender == "assistant"
    assert response.recipient == "user"


def test_create_assistant_response_update_existing():
    """Test updating existing assistant response"""
    response = AssistantResponse(content={"assistant_response": "Original"})
    response.update(sender="assistant")

    assert response.sender == MessageRole.ASSISTANT


def test_create_action_request_basic():
    """Test creating basic action request"""
    request = ActionRequest(
        content={"function": "test_function", "arguments": {"arg": "value"}},
        sender="user",
        recipient="system",
    )

    assert isinstance(request, ActionRequest)
    assert request.content.function == "test_function"
    assert request.content.arguments == {"arg": "value"}
    assert request.sender == "user"
    assert request.recipient == "system"


def test_create_action_request_update_existing():
    """Test updating existing action request"""
    request = ActionRequest(
        content={"function": "original", "arguments": {}},
        sender="user",
        recipient="system",
    )
    request.content.function = "updated_function"

    assert request.content.function == "updated_function"


def test_create_action_response_basic():
    """Test creating basic action response"""
    request = ActionRequest(
        content={"function": "test", "arguments": {}},
        sender="user",
        recipient="system",
    )
    response = ActionResponse(
        content={
            "function": "test",
            "arguments": {},
            "output": {"result": "success"},
            "action_request_id": request.id,
        },
        sender="system",
        recipient="user",
    )
    # Link the request to the response
    request.content.action_response_id = response.id

    assert isinstance(response, ActionResponse)
    assert response.content.output == {"result": "success"}
    assert request.is_responded()


def test_create_action_response_without_request():
    """Test that action response requires valid request ID"""
    # ActionResponse can be created without a request, but won't have proper linking
    response = ActionResponse(
        content={
            "function": "test",
            "arguments": {},
            "output": {"result": "success"},
        }
    )
    # Without action_request_id, the response is valid but unlinked
    assert isinstance(response, ActionResponse)
    assert response.content.action_request_id is None


def test_create_action_response_update_existing():
    """Test updating existing action response"""
    request = ActionRequest(
        content={"function": "test", "arguments": {}},
        sender="user",
        recipient="system",
    )
    response = ActionResponse(
        content={
            "function": "test",
            "arguments": {},
            "output": {"old": "data"},
            "action_request_id": request.id,
        }
    )
    response.content.output = {"new": "data"}

    assert response.content.output == {"new": "data"}


def test_add_message_instruction(message_manager):
    """Test adding instruction via add_message"""
    instruction = message_manager.add_message(
        instruction="Test instruction",
        context={"key": "value"},
        guidance="Some guidance",
        sender="user",
        recipient="assistant",
    )

    assert isinstance(instruction, Instruction)
    assert instruction in message_manager.messages
    assert len(message_manager.messages) == 1
    assert instruction.content.instruction == "Test instruction"


def test_add_message_system(message_manager):
    """Test adding system message via add_message"""
    system = message_manager.add_message(
        system="Test system",
        sender="system",
        recipient="user",
    )

    assert isinstance(system, System)
    assert system in message_manager.messages
    assert message_manager.system == system
    assert len(message_manager.messages) == 1


def test_add_message_assistant_response(message_manager):
    """Test adding assistant response via add_message"""
    response = message_manager.add_message(
        assistant_response="Test response",
        sender="assistant",
        recipient="user",
    )

    assert isinstance(response, AssistantResponse)
    assert response in message_manager.messages
    assert len(message_manager.messages) == 1


def test_add_message_action_request(message_manager):
    """Test adding action request via add_message"""
    request = message_manager.add_message(
        action_function="test_function",
        action_arguments={"arg": "value"},
        sender="user",
        recipient="system",
    )

    assert isinstance(request, ActionRequest)
    assert request in message_manager.messages
    assert len(message_manager.messages) == 1


def test_add_message_action_response(message_manager):
    """Test adding action response via add_message"""
    # First create and add a request
    request = message_manager.add_message(
        action_function="test_function",
        action_arguments={},
        sender="user",
        recipient="system",
    )

    # Now add the response
    response = message_manager.add_message(
        action_request=request,
        action_output={"result": "success"},
        sender="system",
        recipient="user",
    )

    assert isinstance(response, ActionResponse)
    assert response in message_manager.messages
    assert len(message_manager.messages) == 2
    assert request.is_responded()


def test_add_message_with_metadata(message_manager):
    """Test adding message with metadata"""
    metadata = {"custom_key": "custom_value", "priority": "high"}
    msg = message_manager.add_message(
        instruction="Test",
        metadata=metadata,
        sender="user",
        recipient="assistant",
    )

    assert msg.metadata["extra"] == metadata


def test_add_message_update_existing(message_manager):
    """Test updating existing message via add_message"""
    # Add initial message
    msg1 = message_manager.add_message(
        instruction="First version",
        sender="user",
    )

    # Update the same message
    msg2 = message_manager.add_message(
        instruction=msg1,
        guidance="Added guidance",
    )

    assert msg1 is msg2  # Same object
    assert len(message_manager.messages) == 1
    assert msg2.content.guidance == "Added guidance"


def test_add_message_multiple_types_error(message_manager):
    """Test that adding multiple message types raises error"""
    with pytest.raises(
        ValueError, match="Only one message type can be added at a time"
    ):
        message_manager.add_message(
            instruction="Test",
            assistant_response="Response",
            sender="user",
            recipient="assistant",
        )


def test_add_message_system_instruction_error(message_manager):
    """Test that adding system and instruction together raises error"""
    with pytest.raises(
        ValueError, match="Only one message type can be added at a time"
    ):
        message_manager.add_message(
            system="System message",
            instruction="Instruction",
        )


def test_clear_messages_no_system(message_manager):
    """Test clearing messages when no system message exists"""
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    message_manager.add_message(
        assistant_response="Response", sender="assistant", recipient="user"
    )

    assert len(message_manager.messages) == 2
    message_manager.clear_messages()
    assert len(message_manager.messages) == 0


def test_clear_messages_preserves_system(message_manager):
    """Test clearing messages preserves system message"""
    system = message_manager.add_message(system="Test system")
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    message_manager.add_message(
        assistant_response="Response", sender="assistant", recipient="user"
    )

    assert len(message_manager.messages) == 3
    message_manager.clear_messages()
    assert len(message_manager.messages) == 1
    assert system in message_manager.messages


async def test_async_add_message(message_manager):
    """Test async message addition"""
    msg = await message_manager.a_add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    assert isinstance(msg, Instruction)
    assert len(message_manager.messages) == 1


async def test_async_clear_messages(message_manager):
    """Test async clear messages"""
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    assert len(message_manager.messages) == 1

    await message_manager.aclear_messages()
    assert len(message_manager.messages) == 0


def test_last_response(message_manager):
    """Test last_response property"""
    # No responses initially
    assert message_manager.last_response is None

    # Add an instruction
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    assert message_manager.last_response is None

    # Add first response
    response1 = message_manager.add_message(
        assistant_response="Response 1", sender="assistant", recipient="user"
    )
    assert message_manager.last_response == response1

    # Add second response
    response2 = message_manager.add_message(
        assistant_response="Response 2", sender="assistant", recipient="user"
    )
    assert message_manager.last_response == response2


def test_last_instruction(message_manager):
    """Test last_instruction property"""
    # No instructions initially
    assert message_manager.last_instruction is None

    # Add first instruction
    instruction1 = message_manager.add_message(
        instruction="First", sender="user", recipient="assistant"
    )
    assert message_manager.last_instruction == instruction1

    # Add a response
    message_manager.add_message(
        assistant_response="Response", sender="assistant", recipient="user"
    )
    assert message_manager.last_instruction == instruction1

    # Add second instruction
    instruction2 = message_manager.add_message(
        instruction="Second", sender="user", recipient="assistant"
    )
    assert message_manager.last_instruction == instruction2


def test_assistant_responses_property(message_manager):
    """Test assistant_responses property"""
    assert len(message_manager.assistant_responses) == 0

    # Add instruction (not included)
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    assert len(message_manager.assistant_responses) == 0

    # Add responses
    response1 = message_manager.add_message(
        assistant_response="Response 1", sender="assistant", recipient="user"
    )
    response2 = message_manager.add_message(
        assistant_response="Response 2", sender="assistant", recipient="user"
    )

    responses = message_manager.assistant_responses
    assert isinstance(responses, Pile)
    assert len(responses) == 2
    assert response1 in responses
    assert response2 in responses


def test_instructions_property(message_manager):
    """Test instructions property"""
    assert len(message_manager.instructions) == 0

    # Add instructions
    instruction1 = message_manager.add_message(
        instruction="First", sender="user", recipient="assistant"
    )
    instruction2 = message_manager.add_message(
        instruction="Second", sender="user", recipient="assistant"
    )

    # Add response (not included)
    message_manager.add_message(
        assistant_response="Response", sender="assistant", recipient="user"
    )

    instructions = message_manager.instructions
    assert isinstance(instructions, Pile)
    assert len(instructions) == 2
    assert instruction1 in instructions
    assert instruction2 in instructions


def test_action_requests_property(message_manager):
    """Test action_requests property"""
    assert len(message_manager.action_requests) == 0

    # Add action requests
    request1 = message_manager.add_message(
        action_function="func1", action_arguments={}, sender="user"
    )
    request2 = message_manager.add_message(
        action_function="func2", action_arguments={}, sender="user"
    )

    requests = message_manager.action_requests
    assert isinstance(requests, Pile)
    assert len(requests) == 2
    assert request1 in requests
    assert request2 in requests


def test_action_responses_property(message_manager):
    """Test action_responses property"""
    assert len(message_manager.action_responses) == 0

    # Add action request and response
    request = message_manager.add_message(
        action_function="func", action_arguments={}, sender="user"
    )
    response = message_manager.add_message(
        action_request=request, action_output={"result": "success"}
    )

    responses = message_manager.action_responses
    assert isinstance(responses, Pile)
    assert len(responses) == 1
    assert response in responses


def test_actions_property(message_manager):
    """Test actions property (both requests and responses)"""
    assert len(message_manager.actions) == 0

    # Add action request
    request = message_manager.add_message(
        action_function="func", action_arguments={}, sender="user"
    )
    assert len(message_manager.actions) == 1

    # Add action response
    response = message_manager.add_message(
        action_request=request, action_output={"result": "success"}
    )

    actions = message_manager.actions
    assert isinstance(actions, Pile)
    assert len(actions) == 2
    assert request in actions
    assert response in actions


def test_to_chat_msgs_basic(message_manager):
    """Test conversion to chat messages"""
    message_manager.add_message(
        instruction="Test instruction", sender="user", recipient="assistant"
    )
    message_manager.add_message(
        assistant_response="Test response",
        sender="assistant",
        recipient="user",
    )

    chat_msgs = message_manager.to_chat_msgs()
    assert len(chat_msgs) == 2
    assert all(isinstance(msg, dict) for msg in chat_msgs)
    assert all("role" in msg and "content" in msg for msg in chat_msgs)


def test_to_chat_msgs_with_progression(message_manager):
    """Test conversion to chat messages with specific progression"""
    msg1 = message_manager.add_message(
        instruction="First", sender="user", recipient="assistant"
    )
    msg2 = message_manager.add_message(
        assistant_response="Second", sender="assistant", recipient="user"
    )
    msg3 = message_manager.add_message(
        instruction="Third", sender="user", recipient="assistant"
    )

    # Get only first two messages
    chat_msgs = message_manager.to_chat_msgs(progression=[msg1.id, msg2.id])
    assert len(chat_msgs) == 2


def test_to_chat_msgs_empty_progression(message_manager):
    """Test conversion with empty progression"""
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )

    chat_msgs = message_manager.to_chat_msgs(progression=[])
    assert chat_msgs == []


def test_to_chat_msgs_invalid_progression(message_manager):
    """Test conversion with invalid progression raises error"""
    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )

    with pytest.raises(ValueError, match="invalid"):
        message_manager.to_chat_msgs(progression=["invalid_id"])


def test_remove_last_instruction_tool_schemas(message_manager):
    """Test removing tool schemas from last instruction"""
    instruction = message_manager.add_message(
        instruction="Test",
        tool_schemas={"tool1": {}, "tool2": {}},
        sender="user",
    )

    assert instruction.content.tool_schemas is not None
    assert len(instruction.content.tool_schemas) > 0
    message_manager.remove_last_instruction_tool_schemas()
    assert len(instruction.content.tool_schemas) == 0


def test_remove_last_instruction_tool_schemas_no_instruction(message_manager):
    """Test removing tool schemas when no instruction exists"""
    # Should not raise error
    message_manager.remove_last_instruction_tool_schemas()


def test_concat_recent_action_responses_to_instruction(message_manager):
    """Test concatenating action responses to instruction"""
    instruction = message_manager.add_message(
        instruction="Test", context=[], sender="user"
    )

    # Add action request and response
    request = message_manager.add_message(
        action_function="func", action_arguments={}
    )
    response = message_manager.add_message(
        action_request=request, action_output={"result": "success"}
    )

    # Concat responses to instruction
    message_manager.concat_recent_action_responses_to_instruction(instruction)

    # Check that response content was added to instruction context
    assert len(instruction.content.context) > 0


def test_progression_property(message_manager):
    """Test progression property"""
    msg1 = message_manager.add_message(
        instruction="First", sender="user", recipient="assistant"
    )
    msg2 = message_manager.add_message(
        assistant_response="Second", sender="assistant", recipient="user"
    )

    progress = message_manager.progression
    assert list(progress) == [msg1.id, msg2.id]


def test_message_manager_bool(message_manager):
    """Test bool evaluation of message manager"""
    assert not message_manager

    message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )
    assert message_manager


def test_message_manager_contains(message_manager):
    """Test contains operator for message manager"""
    msg = message_manager.add_message(
        instruction="Test", sender="user", recipient="assistant"
    )

    assert msg in message_manager

    other_msg = Instruction(content={"instruction": "Other"})
    assert other_msg not in message_manager


def test_message_manager_with_response_format(message_manager):
    """Test message manager with response format"""
    instruction = message_manager.add_message(
        instruction="Test",
        response_format=RequestModel,
        sender="user",
        recipient="assistant",
    )

    # response_format as BaseModel gets converted to response_schema + auto-generated format
    assert instruction.content.response_schema is not None
    assert isinstance(instruction.content.response_schema, dict)
    assert instruction.content.response_format is not None
    assert isinstance(instruction.content.response_format, dict)


def test_message_manager_with_request_model(message_manager):
    """Test message manager with request model"""
    instruction = message_manager.add_message(
        instruction="Test",
        request_model=RequestModel,
        sender="user",
        recipient="assistant",
    )

    # request_model may be stored in content or may not be exposed directly
    # Check if it's stored in the content or metadata
    assert hasattr(instruction.content, "response_schema") or hasattr(
        instruction, "metadata"
    )


def test_message_manager_with_images(message_manager):
    """Test message manager with images"""
    instruction = message_manager.add_message(
        instruction="Describe this image",
        images=["image1.png", "image2.jpg"],
        image_detail="high",
        sender="user",
    )

    assert instruction.content.images == ["image1.png", "image2.jpg"]
    assert instruction.content.image_detail == "high"


def test_message_manager_with_tool_schemas(message_manager):
    """Test message manager with tool schemas"""
    tool_schemas = {
        "tool1": {"type": "function", "function": {"name": "tool1"}},
        "tool2": {"type": "function", "function": {"name": "tool2"}},
    }

    instruction = message_manager.add_message(
        instruction="Use these tools",
        tool_schemas=tool_schemas,
        sender="user",
    )

    # tool_schemas dict gets wrapped in a list
    assert instruction.content.tool_schemas == [tool_schemas]


def test_complete_conversation_flow(message_manager):
    """Test a complete conversation flow"""
    # Set system message
    system = message_manager.add_message(system="You are a helpful assistant")
    assert message_manager.system == system

    # User instruction
    instruction1 = message_manager.add_message(
        instruction="What is 2+2?", sender="user", recipient="assistant"
    )

    # Assistant response
    response1 = message_manager.add_message(
        assistant_response="2+2 equals 4", sender="assistant", recipient="user"
    )

    # User follow-up
    instruction2 = message_manager.add_message(
        instruction="What about 3+3?", sender="user", recipient="assistant"
    )

    # Assistant with action request
    request = message_manager.add_message(
        action_function="calculate",
        action_arguments={"a": 3, "b": 3},
        sender="assistant",
    )

    # Action response
    action_response = message_manager.add_message(
        action_request=request,
        action_output={"result": 6},
    )

    # Final assistant response
    response2 = message_manager.add_message(
        assistant_response="3+3 equals 6", sender="assistant", recipient="user"
    )

    # Verify the flow
    assert len(message_manager.messages) == 7
    assert message_manager.last_instruction == instruction2
    assert message_manager.last_response == response2
    assert len(message_manager.instructions) == 2
    assert len(message_manager.assistant_responses) == 2
    assert len(message_manager.action_requests) == 1
    assert len(message_manager.action_responses) == 1
    assert len(message_manager.actions) == 2

    # Test chat conversion
    chat_msgs = message_manager.to_chat_msgs()
    assert len(chat_msgs) == 7
