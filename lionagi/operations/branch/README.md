## Usage clarification

Primary interface should be `OperationContext`, which is represented by the `Instruct` object. This object determines the specific orchestration pattern to be applied to a space (`Branch`) and its states. To start with, we have the chat operation, which is a stateless invocation of the branch's chat model (`iModel`) and receives a response. It uses an openai compatible user-assistant pattern for schema and message formatting.

- 0. chat: a stateless invokation of a branch conversation history and an `iModel`, triggers api calling event creation and invokcation.

There are the following three primitive orchestration patterns provided in the branch level operations:
- 1. communicate: chat + append the message to its conversation history
- 2. operate: communicate + can trigger function calling event creation and invokcation 
- 3. ReAct: a loop of operate and communicate operations

and a convinience method `instruct` that chooses the appropriate orchestration pattern basing on the `Instruct` object. To put in daily usage terms, use `communicate` if you want to receive a text response and have the conversation history updated. Use `operate` if additionally, you also grant the branch capability to perform actions, (you can use the `invoke_action` flag to decide whether the real function calling event should be invoked). Use `ReAct` if you want a more "autonomous" operation, where we provide the branch multiple opportunities to iteratively perform `operate` to achieve the desired outcome.


## 1. Chat Method interface
```python
from lionagi.protocols import Instruction, SenderRecipient, Progression
from lionagi.service import iModel
from pydantic import JsonValue

async def chat(
    branch: "Branch",
    instruction: JsonValue | Instruction = None,
    guidance: JsonValue = None,
    context: JsonValue = None,
    sender: SenderRecipient = None,
    recipient: SenderRecipient = None,
    response_format: type[BaseModel] = None,
    progression: Progression = None,
    imodel: iModel = None,
    tool_schemas: list[dict] = None,
    images: list = None,
    image_detail: Literal["low", "high", "auto"] = None,
    plain_content: str = None,
    return_ins_res_message: bool = False,
    include_token_usage_to_model: bool = False,
    **kwargs,   # additional parameters for underlying api calling 
) -> tuple[Instruction, AssistantResponse] | AssistantResponse:
    ...
```

a branch is a manager of managers. In chat method, we primarily utilize the functionalities of the `MessageManager`

```python
class MessageManager(Manager):
    """
    A manager maintaining an ordered list of `RoledMessage` items.
    Capable of setting or replacing a system message, adding instructions,
    assistant responses, or actions, and retrieving them conveniently.
    """

    def __init__(
        self,
        messages: list[RoledMessage] | None = None,
        progression: Progression | None = None,
        system: System | None = None,
    ):
        ...
```

When using chat, you are essentially passing in a list of `RoledMessage` objects, you can select which ones gets sent into the input for api calling by using the `progression` parameter, which is a list of IDs of the messages you want to include in the input. If unspecified, it defaults to all messages, if provide a empty list `[]`, it will only include the system message. chat method **does not** update the conversation history, it is **designed to be stateless**.


The entire parameter interface of chat is designed to create an `Instruction` object. 

One of the following parameters is required to be provided:
- instruction: can be a `Instruction` object, which is a `RoledMessage` for `MessageRole.User`















