# LlamaIndex Integration

Use LlamaIndex for RAG (Retrieval-Augmented Generation) within LionAGI
workflows.

## Why LlamaIndex + LionAGI?

**LlamaIndex**: Best-in-class RAG capabilities, document indexing, retrieval
**LionAGI**: Superior orchestration, parallel execution, multi-agent
coordination

**Together**: Powerful RAG workflows with intelligent orchestration.

## Basic Integration

Wrap LlamaIndex queries as custom operations:

```python
from lionagi import Branch, Builder, Session
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Your existing LlamaIndex setup - no changes needed!
def setup_llamaindex():
    documents = SimpleDirectoryReader("./data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

query_engine = setup_llamaindex()

# Custom operation wrapping LlamaIndex
async def rag_query(branch: Branch, question: str, **kwargs):
    """RAG query using LlamaIndex - keeps your existing code"""
    # Your LlamaIndex query logic unchanged
    response = query_engine.query(question)
    
    # Return in LionAGI format
    return await branch.chat(f"Based on this retrieved information: {response}\n\nQuestion: {question}")

# Use in LionAGI workflow
session = Session()
builder = Builder("rag_workflow")

rag_op = builder.add_operation(
    operation=rag_query,
    question="What are the key findings in the research papers?"
)

result = await session.flow(builder.get_graph())
```

## Multi-Source RAG

Orchestrate multiple knowledge sources in parallel:

```python
from lionagi import Session, Builder, Branch

# Different RAG sources
def setup_financial_rag():
    # Your financial documents RAG
    return financial_query_engine

def setup_technical_rag():
    # Your technical documents RAG  
    return technical_query_engine

# Custom operations for each source
async def financial_rag(branch: Branch, question: str, **kwargs):
    query_engine = setup_financial_rag()
    response = query_engine.query(question)
    return await branch.chat(f"Financial data: {response}\nAnalyze: {question}")

async def technical_rag(branch: Branch, question: str, **kwargs):
    query_engine = setup_technical_rag()
    response = query_engine.query(question) 
    return await branch.chat(f"Technical data: {response}\nAnalyze: {question}")

# Orchestrate multiple RAG sources in parallel
session = Session()
builder = Builder("multi_rag")

question = "What are the technical and financial risks of AI adoption?"

# Parallel RAG queries
financial_op = builder.add_operation(
    operation=financial_rag,
    question=question
)

technical_op = builder.add_operation(
    operation=technical_rag,
    question=question
)

# Synthesize results
synthesis = builder.add_aggregation(
    "communicate",
    source_node_ids=[financial_op, technical_op],
    instruction="Combine financial and technical analysis into comprehensive risk assessment"
)

result = await session.flow(builder.get_graph())
```

## RAG + Analysis Pipeline

Chain RAG retrieval with specialized analysis:

```python
from lionagi import Session, Builder, Branch, iModel

session = Session()
builder = Builder("rag_analysis_pipeline")

# Specialized analysis branches
retriever = Branch(system="Document retrieval specialist")
analyzer = Branch(
    chat_model=iModel(provider="anthropic", model="claude-3-5-sonnet-20240620"),
    system="Deep analysis specialist"
)
synthesizer = Branch(system="Report synthesis specialist")

session.include_branches([retriever, analyzer, synthesizer])

# Phase 1: RAG retrieval
retrieval_op = builder.add_operation(
    operation=rag_query,
    branch=retriever,
    question="What are current market trends in AI?"
)

# Phase 2: Deep analysis (depends on retrieval)
analysis_op = builder.add_operation(
    "communicate",
    branch=analyzer,
    instruction="Provide detailed analysis of the retrieved information",
    depends_on=[retrieval_op]
)

# Phase 3: Final synthesis
synthesis_op = builder.add_operation(
    "communicate",
    branch=synthesizer, 
    instruction="Create executive summary of analysis",
    depends_on=[analysis_op]
)

result = await session.flow(builder.get_graph())
```

## Advanced: Multi-Modal RAG

Handle different document types with specialized processing:

```python
# Different document processors
async def pdf_rag(branch: Branch, query: str, **kwargs):
    # Your LlamaIndex PDF processing
    pdf_engine = setup_pdf_rag()
    response = pdf_engine.query(query)
    return await branch.chat(f"PDF data: {response}\nQuery: {query}")

async def web_rag(branch: Branch, query: str, **kwargs):
    # Your LlamaIndex web scraping  
    web_engine = setup_web_rag()
    response = web_engine.query(query)
    return await branch.chat(f"Web data: {response}\nQuery: {query}")

async def database_rag(branch: Branch, query: str, **kwargs):
    # Your LlamaIndex database querying
    db_engine = setup_database_rag()
    response = db_engine.query(query)
    return await branch.chat(f"Database data: {response}\nQuery: {query}")

# Orchestrate all sources
builder = Builder("multimodal_rag")
query = "What are the latest developments in AI regulations?"

# Parallel multi-modal retrieval
pdf_op = builder.add_operation(operation=pdf_rag, query=query)
web_op = builder.add_operation(operation=web_rag, query=query)  
db_op = builder.add_operation(operation=database_rag, query=query)

# Cross-reference and validate
validator = Branch(system="Information validation specialist")
validation_op = builder.add_operation(
    "communicate",
    branch=validator,
    instruction="Cross-reference all sources and validate information consistency",
    depends_on=[pdf_op, web_op, db_op]
)

result = await session.flow(builder.get_graph())
```

## Error Handling with RAG

Handle RAG failures gracefully:

```python
async def resilient_rag(branch: Branch, question: str, **kwargs):
    """RAG with fallback strategies"""
    
    # Try primary RAG source
    try:
        primary_engine = setup_primary_rag()
        response = primary_engine.query(question)
        return await branch.chat(f"Retrieved: {response}\nAnalyze: {question}")
    
    except Exception as e:
        print(f"Primary RAG failed: {e}")
        
        # Fallback to secondary source
        try:
            fallback_engine = setup_fallback_rag()
            response = fallback_engine.query(question)
            return await branch.chat(f"Fallback data: {response}\nAnalyze: {question}")
        
        except Exception as e2:
            print(f"Fallback RAG failed: {e2}")
            
            # Ultimate fallback: use LLM knowledge
            return await branch.chat(f"Using model knowledge to answer: {question}")

# Use resilient RAG in workflow
builder.add_operation(operation=resilient_rag, question="Complex query here")
```

## Performance Optimization

Optimize RAG workflows for production:

```python
import asyncio

async def batch_rag_queries(branch: Branch, questions: list, **kwargs):
    """Process multiple RAG queries in parallel"""
    
    async def single_rag_query(question):
        response = query_engine.query(question)
        return {"question": question, "response": str(response)}
    
    # Parallel RAG queries within the operation
    results = await asyncio.gather(*[
        single_rag_query(q) for q in questions
    ])
    
    # Synthesize all results
    combined = "\n\n".join([f"Q: {r['question']}\nA: {r['response']}" for r in results])
    return await branch.chat(f"Synthesize these RAG results:\n{combined}")

# Use in high-throughput scenarios
questions = ["Question 1", "Question 2", "Question 3"]
batch_op = builder.add_operation(
    operation=batch_rag_queries,
    questions=questions
)
```

## Key Benefits

1. **Zero Migration**: Keep your existing LlamaIndex code unchanged
2. **Superior Orchestration**: LionAGI handles parallel execution, dependencies,
   error handling
3. **Multi-Source**: Easily orchestrate multiple RAG sources
4. **Scalable**: Built-in performance controls and monitoring
5. **Flexible**: Mix RAG with other AI operations seamlessly

LlamaIndex provides the RAG capabilities, LionAGI provides the orchestration
intelligence.
