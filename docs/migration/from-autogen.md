# Migrating from AutoGen

Direct comparisons showing AutoGen patterns and LionAGI equivalents.

## Two-Agent Conversation

**AutoGen:**

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig(api_type="openai", model="gpt-4o-mini")

assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

human = ConversableAgent(name="human", human_input_mode="ALWAYS")
human.initiate_chat(assistant, message="Hello! What's 2 + 2?")
```

**LionAGI:**

```python
from lionagi import Branch, iModel

assistant = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a helpful assistant."
)

response = await assistant.communicate("Hello! What's 2 + 2?")
```

## Multi-Agent GroupChat

**AutoGen:**

```python
from autogen import AssistantAgent, GroupChat, GroupChatManager

coder = AssistantAgent(name="Coder", llm_config=config_list)
reviewer = AssistantAgent(name="Reviewer", llm_config=config_list)

groupchat = GroupChat(agents=[coder, reviewer], messages=[], max_round=5)
manager = GroupChatManager(groupchat=groupchat, llm_config=config_list)

user.initiate_chat(manager, message="Generate a Python function")
```

**LionAGI:**

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

coder = builder.add_operation(
    "communicate", 
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    instruction="Generate a Python function"
)

reviewer = builder.add_operation(
    "communicate",
    depends_on=[coder],
    chat_model=iModel(provider="openai", model="gpt-4o-mini"), 
    instruction="Review the generated code"
)

result = await session.flow(builder.get_graph())
```

## Tool Integration

**AutoGen:**

```python
from autogen import ConversableAgent, register_function
from datetime import datetime

def get_weekday(date_string: str) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

date_agent = ConversableAgent(name="date_agent", llm_config=llm_config)
executor = ConversableAgent(name="executor", human_input_mode="NEVER")

register_function(get_weekday, caller=date_agent, executor=executor)
result = executor.initiate_chat(date_agent, message="What day was March 25, 1995?")
```

**LionAGI:**

```python
from lionagi import Branch
from datetime import datetime

def get_weekday(date_string: str) -> str:
    date = datetime.strptime(date_string, "%Y-%m-%d")
    return date.strftime("%A")

date_branch = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    tools=[get_weekday]  # Direct function passing
)

result = await date_branch.ReAct(
    instruct={"instruction": "What day was March 25, 1995?"},
    max_extensions=2
)
```

## Parallel Research Workflow

**AutoGen:**

```python
# Sequential GroupChat approach
agents = [researcher1, researcher2, analyst]
groupchat = GroupChat(agents=agents, max_round=10)
manager = GroupChatManager(groupchat=groupchat)
result = user.initiate_chat(manager, message="Research AI trends")
```

**LionAGI:**

```python
from lionagi import Session, Builder
import asyncio

session = Session()
builder = Builder()

# Parallel research
research_nodes = []
for i, topic in enumerate(["transformers", "multimodal", "reasoning"]):
    node = builder.add_operation(
        "communicate",
        instruction=f"Research {topic} developments in 2024"
    )
    research_nodes.append(node)

# Synthesis
synthesis = builder.add_operation(
    "communicate",
    depends_on=research_nodes,
    instruction="Synthesize research findings into comprehensive report"
)

result = await session.flow(builder.get_graph())
```

## State Management

**AutoGen:**

```python
def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is architect:
        return implementer
    elif last_speaker is implementer:
        return tester
    return architect

groupchat = GroupChat(
    agents=[architect, implementer, tester],
    speaker_selection_method=state_transition
)
```

**LionAGI:**

```python
# Explicit dependencies handle state transitions
architect_node = builder.add_operation("communicate", instruction="Design system")
impl_node = builder.add_operation("communicate", depends_on=[architect_node], instruction="Implement design")  
test_node = builder.add_operation("communicate", depends_on=[impl_node], instruction="Test implementation")

await session.flow(builder.get_graph())
```

## Enterprise Advantages

**AutoGen's "Too Auto" Problem**: AutoGen makes autonomous decisions that lack
enterprise controls

**LionAGI's Enterprise Features:**

```python
# Explicit control over agent behavior
session = Session()
builder = Builder()

# Predictable execution paths
architect_node = builder.add_operation("communicate", instruction="Design system")
review_node = builder.add_operation("communicate", depends_on=[architect_node], instruction="Review design")

# Built-in cost tracking
result = await session.flow(builder.get_graph())
from lionagi.protocols.messages.assistant_response import AssistantResponse

# Built-in cost tracking
costs = 0
def get_context(node_id):
    nonlocal costs
    graph = builder.get_graph()
    node = graph.internal_nodes[node_id]
    branch = session.get_branch(node.branch_id, None)
    if branch and len(branch.messages) > 0:
        if isinstance(msg := branch.messages[-1], AssistantResponse):
            costs += msg.model_response.get("total_cost_usd") or 0

# Track costs across workflow  
for node in [architect_node, review_node]:
    get_context(node)

# Detailed audit trail
for node_id, node in builder.get_graph().internal_nodes.items():
    branch = session.get_branch(node.branch_id, None)
    print(f"Node {node_id}: {len(branch.messages)} messages")
```

**Enterprise Requirements:** 

✅ **Predictable Costs**: Built-in usage tracking vs AutoGen's unknown spending  
✅ **Deterministic Flow**: Explicit dependencies vs AutoGen's autonomous decisions  
✅ **Audit Compliance**: Full execution logs vs AutoGen's black-box conversations  
✅ **Error Recovery**: Granular failure handling vs AutoGen's all-or-nothing  
✅ **Resource Control**: Bounded execution vs AutoGen's unlimited autonomy

## Key Migration Points

- **Conversation → Graph**: AutoGen's linear conversations become explicit dependency graphs  
- **Agents → Branches**: ConversableAgent functionality maps to Branch instances  
- **GroupChat → Builder**: Multi-agent coordination uses Builder pattern with dependencies  
- **Speaker Selection → Dependencies**: State transitions become explicit `depends_on` relationships  
- **Initiate Chat → Session Flow**: Conversation starts become graph execution  
- **Autonomous → Controlled**: Replace AutoGen's unpredictable autonomy with enterprise controls
