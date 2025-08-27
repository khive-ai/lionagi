# Migrating from LangChain

LangChain's complex abstraction layers → LionAGI's direct simplicity.

## Basic LLM Chain

**LangChain (Verbose LCEL):**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}")
])

model = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

chain = prompt | model | output_parser

result = chain.invoke({"input": "What is 2 + 2?"})
```

**LionAGI (Direct):**

```python
from lionagi import Branch, iModel

assistant = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    system="You are a helpful assistant"
)

result = await assistant.communicate("What is 2 + 2?")
```

!!! success "Why This Is Better"
    **90% less boilerplate**: No need for separate prompt templates, output parsers, or chain assembly  
    **Native async**: Built for async/await from the ground up, not retrofitted  
    **Direct API**: Call `communicate()` instead of learning LCEL pipe syntax  
    **Automatic memory**: No manual memory management required

## Agent with Tools

**LangChain (Complex Setup):**

```python
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y

search = TavilySearchResults()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

agent = create_openai_functions_agent(llm, [multiply, search], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[multiply, search], verbose=True)

result = agent_executor.invoke({"input": "What is 25 times 4?"})
```

**LionAGI (Clean):**

```python
from lionagi import Branch

def multiply(x: float, y: float) -> float:
    """Multiply two numbers"""
    return x * y

def search(query: str) -> str:
    """Search for information"""
    return f"Search results for {query}"

agent = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini"),
    tools=[multiply, search]  # Direct function passing
)

result = await agent.ReAct(
    instruct={"instruction": "What is 25 times 4?"},
    max_extensions=3
)
```

!!! success "Why This Is Better"
    **Direct function passing**: No decorators or complex tool registration  
    **Built-in ReAct**: No need to implement reasoning loops manually  
    **Cleaner imports**: Everything from one package, not scattered across langchain-*  
    **Simpler debugging**: Direct function calls, not abstracted away behind agents and executors

## Multi-Agent RAG Workflow

**LangChain (LangGraph Required):**

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from typing import TypedDict, List
import operator

class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    research_results: List[str]
    final_answer: str

def researcher_node(state: AgentState):
    # Research implementation
    research_llm = ChatOpenAI(model="gpt-4o-mini")
    result = research_llm.invoke(state["messages"][-1])
    return {"research_results": [result.content]}

def analyst_node(state: AgentState):
    # Analysis implementation  
    analyst_llm = ChatOpenAI(model="gpt-4o-mini")
    context = "\n".join(state["research_results"])
    result = analyst_llm.invoke(f"Analyze: {context}")
    return {"final_answer": result.content}

workflow = StateGraph(AgentState)
workflow.add_node("researcher", researcher_node)
workflow.add_node("analyst", analyst_node)
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "analyst")
workflow.add_edge("analyst", END)

app = workflow.compile()
result = app.invoke({"messages": [HumanMessage(content="Research AI trends")]})
```

**LionAGI (Natural Graph):**

```python
from lionagi import Session, Builder

session = Session()
builder = Builder()

research = builder.add_operation(
    "communicate",
    instruction="Research AI trends in detail"
)

analysis = builder.add_operation(
    "communicate",
    depends_on=[research],
    instruction="Analyze research findings and provide insights"
)

result = await session.flow(builder.get_graph())
```

!!! success "Why This Is Better"
    **No state classes**: LionAGI eliminates the need for TypedDict state definitions - the framework handles data flow automatically  
    **Zero boilerplate**: Compare 6 lines of LionAGI vs 30+ lines of LangGraph for the same workflow  
    **Automatic orchestration**: Dependencies are declared once with `depends_on`, not manually configured with edges  
    **Natural syntax**: Code reads like the business logic you want to accomplish, not framework mechanics  
    **Built-in parallelism**: When dependencies allow, operations run concurrently without additional configuration

## Parallel Processing

**LangChain (Complex State Management):**

```python
from langgraph.graph import StateGraph
from typing import Annotated
import operator

class ParallelState(TypedDict):
    topics: List[str]
    results: Annotated[List[str], operator.add]

def research_node_1(state):
    # Research topic 1
    pass

def research_node_2(state):
    # Research topic 2  
    pass

def research_node_3(state):
    # Research topic 3
    pass

def synthesis_node(state):
    # Combine results
    pass

workflow = StateGraph(ParallelState)
workflow.add_node("research_1", research_node_1)  
workflow.add_node("research_2", research_node_2)
workflow.add_node("research_3", research_node_3)
workflow.add_node("synthesis", synthesis_node)

# Complex parallel configuration
workflow.set_entry_point("research_1")
workflow.set_entry_point("research_2") 
workflow.set_entry_point("research_3")
workflow.add_edge(["research_1", "research_2", "research_3"], "synthesis")

app = workflow.compile()
```

**LionAGI (Automatic Parallel):**

```python
# Natural parallel execution
topics = ["transformers", "multimodal", "reasoning"]
research_nodes = []

for topic in topics:
    node = builder.add_operation(
        "communicate", 
        instruction=f"Research {topic} developments"
    )
    research_nodes.append(node)

synthesis = builder.add_operation(
    "communicate",
    depends_on=research_nodes,
    instruction="Synthesize all research findings"
)

result = await session.flow(builder.get_graph())  # Automatic parallel
```

!!! success "Why This Is Better"
    **True parallelism**: LionAGI runs independent operations simultaneously  
    **No manual coordination**: Dependencies automatically determine execution order  
    **Dynamic graphs**: Can generate parallel operations programmatically  
    **Optimal performance**: 3-4x faster than sequential execution for multi-step workflows

## Memory and Context

**LangChain (Manual Memory Management):**

```python
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Manual memory updates
memory.chat_memory.add_user_message("Hello")
memory.chat_memory.add_ai_message("Hi there!")

# Complex integration with chains
chain_with_memory = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)
```

**LionAGI (Built-in Memory):**

```python
# Automatic persistent memory
assistant = Branch(
    chat_model=iModel(provider="openai", model="gpt-4o-mini")
)

# Memory handled automatically
response1 = await assistant.communicate("Hello")
response2 = await assistant.communicate("What did I just say?")  # Remembers context
```

!!! success "Why This Is Better"
    **Automatic persistence**: No manual memory add operations  
    **Isolated contexts**: Each Branch maintains its own conversation history  
    **No configuration**: Works out of the box with sensible defaults  
    **Multi-agent memory**: Each agent remembers its own interactions independently

## Error Handling and Debugging

**LangChain (Limited Observability):**

```python
import langsmith
from langchain_openai import ChatOpenAI

# Requires external LangSmith setup for debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"

try:
    result = chain.invoke(input_data)
except Exception as e:
    # Limited error context
    print(f"Chain failed: {e}")
```

**LionAGI (Rich Error Context):**

```python
try:
    result = await session.flow(builder.get_graph())
except Exception as e:
    import traceback
    traceback.print_exc()  # Full context
    
    # Access detailed execution state
    for node_id, node in builder.get_graph().internal_nodes.items():
        branch = session.get_branch(node.branch_id, None)
        if branch:
            print(f"Node {node_id}: {len(branch.messages)} messages")
```

!!! success "Why This Is Better"
    **Full execution context**: Access to all Branch states and message history  
    **No external dependencies**: Built-in debugging tools, no LangSmith needed  
    **Rich error information**: See exactly where workflows fail with full context  
    **Production debugging**: Easy to add monitoring and observability

## Key Simplifications

- **No LCEL Syntax**: Direct function calls instead of pipe operators  
- **No State Management**: Automatic memory and context handling  
- **No Complex Setup**: Simple imports and initialization  
- **No External Dependencies**: Built-in observability and debugging  
- **No Manual Orchestration**: Automatic parallel execution

## Migration Benefits

✅ **90% Less Code**: Remove LCEL, state management, memory setup  
✅ **Natural Async**: Built for async/await from the ground up  
✅ **Automatic Parallelism**: No complex graph configuration needed  
✅ **Simpler Debugging**: Direct access to execution state  
✅ **Built-in Memory**: No manual memory management required  
✅ **Cost Tracking**: Native usage monitoring vs external tools
