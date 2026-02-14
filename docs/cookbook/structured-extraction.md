# Structured Data Extraction

Extract typed, validated data from unstructured text using `response_format`
and Pydantic models.

## Invoice Extraction

```python
import asyncio
from pydantic import BaseModel, Field
from lionagi import Branch, iModel

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float
    total: float

class Invoice(BaseModel):
    vendor: str
    invoice_number: str
    date: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float

async def extract_invoice(text: str) -> Invoice:
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system="Extract structured invoice data. Be precise with numbers.",
    )
    result = await branch.communicate(
        f"Extract the invoice details:\n\n{text}",
        response_format=Invoice,
    )
    return result

# Usage
raw_text = """
INVOICE #2024-0847
From: Acme Cloud Services
Date: 2024-12-15

- GPU Compute (A100), 200 hrs @ $3.50/hr = $700.00
- Storage (S3-compatible), 5 TB @ $23.00/TB = $115.00
- Egress bandwidth, 2 TB @ $90.00/TB = $180.00

Subtotal: $995.00
Tax (8.5%): $84.58
Total: $1,079.58
"""

invoice = asyncio.run(extract_invoice(raw_text))
print(f"Vendor: {invoice.vendor}")
print(f"Items: {len(invoice.line_items)}")
print(f"Total: ${invoice.total:.2f}")
```

## Batch Extraction with Validation

Process multiple documents and validate consistency:

```python
from typing import Literal

class ContactInfo(BaseModel):
    name: str
    email: str | None = None
    phone: str | None = None
    company: str | None = None
    role: str | None = None
    confidence: Literal["high", "medium", "low"]

class ExtractionResult(BaseModel):
    contacts: list[ContactInfo]
    source_summary: str

async def extract_contacts(documents: list[str]) -> list[ExtractionResult]:
    """Extract contacts from multiple documents in parallel."""
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system=(
            "Extract contact information from text. "
            "Set confidence based on how complete the data is."
        ),
    )

    tasks = [
        branch.communicate(
            f"Extract all contacts from:\n\n{doc}",
            response_format=ExtractionResult,
        )
        for doc in documents
    ]
    return await asyncio.gather(*tasks)

# Usage
docs = [
    "Meeting with Sarah Chen (sarah@acme.io, VP Engineering) about the API.",
    "Call from 555-0142, John from DataCorp asking about pricing.",
    "Email thread with team@startup.co — Alice (CTO) and Bob (Lead Dev).",
]

results = asyncio.run(extract_contacts(docs))
for r in results:
    for c in r.contacts:
        print(f"  {c.name} ({c.confidence}) - {c.email or c.phone or 'no contact'}")
```

## Nested Extraction with Enums

Handle complex schemas with nested objects and constrained fields:

```python
from enum import Enum

class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"

class AffectedSystem(BaseModel):
    name: str
    version: str | None = None
    component: str | None = None

class SecurityFinding(BaseModel):
    title: str
    severity: Severity
    description: str
    affected_systems: list[AffectedSystem]
    remediation: str
    cve_ids: list[str] = Field(default_factory=list)

class SecurityReport(BaseModel):
    findings: list[SecurityFinding]
    executive_summary: str
    risk_score: float = Field(ge=0, le=10)

async def parse_security_report(report_text: str) -> SecurityReport:
    branch = Branch(
        chat_model=iModel(provider="openai", model="gpt-4.1"),
        system="Parse security audit reports into structured findings.",
    )
    return await branch.communicate(
        f"Parse this security report:\n\n{report_text}",
        response_format=SecurityReport,
    )
```

## Extraction with Tool-Assisted Enrichment

Use `operate()` to extract data and enrich it with tool calls:

```python
def lookup_company(name: str) -> str:
    """Look up company information by name."""
    # Replace with actual API call
    return f"{name}: Fortune 500, HQ in San Francisco, 10k employees"

def validate_email(email: str) -> str:
    """Validate an email address format."""
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    valid = bool(re.match(pattern, email))
    return f"{email}: {'valid' if valid else 'invalid'} format"

class EnrichedContact(BaseModel):
    name: str
    email: str | None = None
    company: str | None = None
    company_info: str | None = None
    email_valid: bool | None = None

async def extract_and_enrich(text: str) -> EnrichedContact:
    branch = Branch(
        tools=[lookup_company, validate_email],
        chat_model=iModel(provider="openai", model="gpt-4.1-mini"),
        system="Extract contact info, then use tools to enrich and validate.",
    )

    result = await branch.operate(
        instruction=f"Extract the contact, look up their company, validate email:\n\n{text}",
        response_format=EnrichedContact,
        actions=True,
    )
    return result

# Usage
text = "Just spoke with Maria Lopez (maria.lopez@techcorp.com) from TechCorp."
contact = asyncio.run(extract_and_enrich(text))
print(f"{contact.name} at {contact.company} — email valid: {contact.email_valid}")
```

## Handling Parse Failures

Control behavior when extraction fails:

```python
# Strict: raise on failure
result = await branch.communicate(
    "Extract data from this messy text...",
    response_format=Invoice,
    # communicate uses num_parse_retries internally
    num_parse_retries=3,
)

# Via operate: explicit validation handling
result = await branch.operate(
    instruction="Extract data from this messy text...",
    response_format=Invoice,
    handle_validation="return_value",  # Return best-effort parse
    # Other options: "raise", "return_none"
)
```

## When to Use

**Perfect for:** Invoice processing, email parsing, log analysis, report
digitization, CRM data entry, compliance document processing.

**Key patterns:**

- Use `response_format` on `communicate()` for simple extraction
- Use `operate()` with tools when extraction needs enrichment
- Use `handle_validation="return_value"` for best-effort parsing on messy input
- Create separate Pydantic models per document type for type safety
