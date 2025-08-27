# Academic Claim Extraction

ReAct-based claim extraction with sequential document analysis and structured
outputs.

## Basic Claim Extraction

```python
from typing import Literal
from pathlib import Path
from pydantic import BaseModel, Field
from lionagi import Branch, Session, Builder, types, iModel
from lionagi.tools.types import ReaderTool

# Structured claim models
class Claim(BaseModel):
    claim: str
    type: Literal["citation", "performance", "technical", "other"]
    location: str = Field(..., description="Section/paragraph reference")
    verifiability: Literal["high", "medium", "low"]
    search_strategy: str = Field(..., description="How to verify this claim")

class ClaimExtraction(BaseModel):
    claims: list[Claim]

async def extract_claims_from_document(document_path: str):
    """Extract verifiable claims using ReAct pattern"""
    
    # Create ReAct-enabled branch with ReaderTool
    extractor = Branch(
        tools=[ReaderTool],
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        name="claim_extractor"
    )
    
    # Use ReAct for systematic claim extraction
    result = await extractor.ReAct(
        instruct=types.Instruct(
            instruction=(
                f"Use ReaderTool to analyze document at {document_path}. "
                "Extract 5-7 specific, verifiable claims. Focus on citations, "
                "performance metrics, and technical assertions."
            ),
            context={"document_path": document_path}
        ),
        response_format=ClaimExtraction,
        tools=["reader_tool"],
        max_extensions=4,
        verbose=True
    )
    
    return result

# Usage
document_path = "data/research_paper.pdf"
claims = await extract_claims_from_document(document_path)

for claim in claims.claims:
    print(f"[{claim.type.upper()}] {claim.claim}")
    print(f"Location: {claim.location}")
    print(f"Verifiability: {claim.verifiability}\n")
```

## Sequential Document Analysis

```python
async def sequential_claim_analysis(document_path: str):
    """Progressive document analysis: open → analyze → extract claims"""
    
    # Create branch with ReaderTool
    analyzer = Branch(
        tools=[ReaderTool],
        chat_model=iModel(provider="openai", model="gpt-4o-mini")
    )
    session = Session(default_branch=analyzer)
    builder = Builder("ClaimAnalysis")
    
    # Step 1: Document exploration
    doc_reader = builder.add_operation(
        "ReAct",
        node_id="explore_document",
        instruct=types.Instruct(
            instruction=(
                "Open and explore the document structure. "
                "Identify sections containing verifiable claims."
            ),
            context={"document_path": document_path}
        ),
        tools=["reader_tool"],
        max_extensions=2,
        verbose=True
    )
    
    # Step 2: Content analysis
    content_analyzer = builder.add_operation(
        "ReAct",
        node_id="analyze_content",
        depends_on=[doc_reader],
        instruct=types.Instruct(
            instruction=(
                "Analyze key sections for citations, technical claims, "
                "and performance metrics that can be verified."
            )
        ),
        response_format=types.Outline,
        tools=["reader_tool"],
        max_extensions=3,
        verbose=True
    )
    
    # Step 3: Claim extraction
    claim_extractor = builder.add_operation(
        "ReAct",
        node_id="extract_claims",
        depends_on=[content_analyzer],
        instruct=types.Instruct(
            instruction=(
                "Extract specific verifiable claims based on analysis. "
                "Prioritize citations, performance data, and technical assertions."
            )
        ),
        response_format=ClaimExtraction,
        tools=["reader_tool"],
        max_extensions=3,
        verbose=True
    )
    
    # Execute sequential workflow
    graph = builder.get_graph()
    result = await session.flow(graph, parallel=False, verbose=True)
    
    return result["operation_results"][claim_extractor]

# Usage
claims = await sequential_claim_analysis("data/ai_safety_paper.pdf")
```

## Multi-Document Claim Extraction

```python
from typing import Dict
import asyncio

class DocumentClaims(BaseModel):
    document: str
    claims: list[Claim]
    summary: str

async def extract_from_multiple_documents(document_paths: list[str]):
    """Parallel claim extraction from multiple documents"""
    
    async def process_document(doc_path: str) -> DocumentClaims:
        """Process single document"""
        extractor = Branch(
            tools=[ReaderTool],
            chat_model=iModel(provider="openai", model="gpt-4o-mini"),
            name=f"extractor_{Path(doc_path).stem}"
        )
        
        # Extract claims using ReAct
        result = await extractor.ReAct(
            instruct=types.Instruct(
                instruction=(
                    f"Analyze {doc_path} and extract verifiable claims. "
                    "Focus on novel findings and key assertions."
                ),
                context={"document": doc_path}
            ),
            response_format=ClaimExtraction,
            tools=["reader_tool"],
            max_extensions=3
        )
        
        # Generate summary
        summary = await extractor.communicate(
            "Provide brief summary of the document's main contributions"
        )
        
        return DocumentClaims(
            document=doc_path,
            claims=result.claims,
            summary=summary
        )
    
    # Process documents in parallel
    tasks = [process_document(doc) for doc in document_paths]
    results = await asyncio.gather(*tasks)
    
    return results

# Usage
papers = [
    "data/transformer_paper.pdf",
    "data/bert_paper.pdf", 
    "data/gpt_paper.pdf"
]
all_claims = await extract_from_multiple_documents(papers)
```

## Claim Validation Pipeline

```python
class ValidationResult(BaseModel):
    claim: str
    validation_status: Literal["verified", "disputed", "unclear", "unverifiable"]
    evidence: list[str]
    confidence: float

class ClaimValidator(BaseModel):
    validations: list[ValidationResult]

def search_evidence(claim: str) -> str:
    """Mock search function - replace with actual search API"""
    return f"Search results for: {claim}"

def cross_reference(claim: str, reference_docs: list[str]) -> str:
    """Cross-reference claim against known sources"""
    return f"Cross-reference results for: {claim}"

async def validate_claims(claims: list[Claim], reference_docs: list[str] = None):
    """Validate extracted claims using ReAct reasoning"""
    
    validator = Branch(
        tools=[search_evidence, cross_reference],
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        name="claim_validator"
    )
    
    validation_tasks = []
    
    for claim in claims:
        # Use ReAct to validate each claim
        task = validator.ReAct(
            instruct=types.Instruct(
                instruction=(
                    f"Validate this claim: '{claim.claim}' "
                    "Use available tools to search for evidence and cross-reference."
                ),
                context={
                    "claim": claim.model_dump(),
                    "reference_docs": reference_docs or []
                }
            ),
            response_format=ValidationResult,
            max_extensions=4,
            verbose=True
        )
        validation_tasks.append(task)
    
    # Execute validations in parallel
    validations = await asyncio.gather(*validation_tasks)
    
    return ClaimValidator(validations=validations)

# Usage
extracted_claims = claims.claims  # From previous extraction
validation_results = await validate_claims(extracted_claims)

for validation in validation_results.validations:
    print(f"Claim: {validation.claim}")
    print(f"Status: {validation.validation_status}")
    print(f"Confidence: {validation.confidence}\n")
```

## Citation-Specific Extraction

```python
class Citation(BaseModel):
    text: str
    authors: list[str]
    year: int
    title: str
    venue: str
    context: str = Field(..., description="Context where citation appears")

class CitationExtraction(BaseModel):
    citations: list[Citation]

async def extract_citations(document_path: str):
    """Extract and structure citations from academic papers"""
    
    citation_extractor = Branch(
        tools=[ReaderTool],
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You specialize in extracting citations from academic papers",
        name="citation_extractor"
    )
    
    # Step 1: Identify citation patterns
    result = await citation_extractor.ReAct(
        instruct=types.Instruct(
            instruction=(
                "Scan document for citations and references. "
                "Extract complete citation information including context."
            ),
            context={"document_path": document_path}
        ),
        response_format=CitationExtraction,
        tools=["reader_tool"],
        max_extensions=5,
        verbose=True
    )
    
    return result

# Usage
citations = await extract_citations("data/survey_paper.pdf")
print(f"Found {len(citations.citations)} citations")
```

## Performance Claims Analysis

```python
class PerformanceClaim(BaseModel):
    metric: str
    value: str
    baseline: str = None
    improvement: str = None
    dataset: str
    methodology: str
    location: str

class PerformanceExtraction(BaseModel):
    performance_claims: list[PerformanceClaim]

async def extract_performance_claims(document_path: str):
    """Extract performance metrics and benchmarks"""
    
    performance_extractor = Branch(
        tools=[ReaderTool],
        chat_model=iModel(provider="openai", model="gpt-4o-mini"),
        system="You extract performance metrics and experimental results from research papers",
        name="performance_extractor"
    )
    
    result = await performance_extractor.ReAct(
        instruct=types.Instruct(
            instruction=(
                "Extract performance claims including metrics, values, "
                "baselines, datasets, and methodology details."
            ),
            context={"document_path": document_path}
        ),
        response_format=PerformanceExtraction,
        tools=["reader_tool"],
        max_extensions=4,
        verbose=True
    )
    
    return result

# Usage
performance_data = await extract_performance_claims("data/benchmark_paper.pdf")
```

## Production Pipeline

```python
async def production_claim_extraction_pipeline(
    document_paths: list[str],
    validate_claims: bool = True,
    extract_citations: bool = True
):
    """Complete production pipeline for claim extraction"""
    
    try:
        results = {}
        
        for doc_path in document_paths:
            print(f"Processing: {doc_path}")
            
            # Sequential analysis
            claims = await sequential_claim_analysis(doc_path)
            results[doc_path] = {"claims": claims}
            
            # Optional citation extraction
            if extract_citations:
                citations = await extract_citations(doc_path)
                results[doc_path]["citations"] = citations
            
            # Optional claim validation
            if validate_claims:
                validations = await validate_claims(claims.claims)
                results[doc_path]["validations"] = validations
                
            print(f"Completed: {doc_path}")
        
        return results
        
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage
documents = [
    "data/paper1.pdf",
    "data/paper2.pdf",
    "data/paper3.pdf"
]

pipeline_results = await production_claim_extraction_pipeline(
    documents,
    validate_claims=True,
    extract_citations=True
)
```

## Key Features

**ReAct Integration:**

- Systematic reasoning before extraction
- Tool-assisted document analysis
- Progressive understanding building

**Structured Outputs:**

- Pydantic models for reliable parsing
- Type-safe claim categorization
- Consistent data formats

**Sequential Processing:**

- Document exploration → analysis → extraction
- Context-aware claim identification
- Dependency-based operation flow

**Validation Pipeline:**

- Evidence searching and cross-referencing
- Confidence scoring for claims
- Multi-source verification

**Production Ready:**

- Error handling and recovery
- Parallel processing support
- Comprehensive logging and monitoring
