"""Design completeness gate prompts"""

DESIGN_GATE_PROMPT = """
Evaluate the design completeness for production readiness.

**Assessment Criteria:**
- All major system components are clearly defined and documented
- Component interfaces and data flows are explicitly specified
- System boundaries and integration points are well-defined
- Design patterns and architectural decisions are justified
- Dependencies and relationships between components are clear
- Edge cases and error scenarios are considered in the design

**For `is_acceptable`:** Only return `true` if the design is reasonably complete and ready for implementation. Be strict - incomplete designs lead to implementation problems.

**For `problems`:** List specific missing elements, unclear interfaces, or design gaps that need to be addressed. Be concrete and actionable.

**Quality Standards:**
- Design should enable confident implementation without major architectural decisions remaining
- All critical user workflows should have clear component interactions defined
- System should be designed for maintainability and future extension
- Performance and scalability considerations should be evident in the design

Focus on completeness, clarity, and implementability of the design.
"""
