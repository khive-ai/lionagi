"""Documentation validation gate prompts"""

DOCUMENTATION_GATE_PROMPT = """
Evaluate documentation completeness for development team handoff and production support.

**Assessment Criteria:**
- User-facing documentation enables successful system usage
- Developer documentation supports effective maintenance and extension
- Operational documentation enables reliable production deployment and support
- Documentation is clear, accurate, and up-to-date
- Examples and tutorials help users understand common scenarios
- Troubleshooting guides help resolve common issues
- API documentation is comprehensive and includes examples

**For `is_acceptable`:** Only return `true` if documentation is sufficient for team handoff and production operation. Poor documentation leads to confusion, errors, and maintenance difficulties.

**For `problems`:** List specific documentation gaps, unclear sections, or missing information that would hinder effective system use or maintenance.

**Documentation Categories:**
- User Documentation: User guides, tutorials, feature documentation, FAQ
- API Documentation: Endpoint descriptions, request/response examples, authentication guides
- Developer Documentation: Setup guides, architecture overview, code organization, contribution guidelines
- Operational Documentation: Deployment procedures, configuration management, monitoring setup
- Troubleshooting: Common issues, error messages, debugging procedures, performance tuning

**Quality Standards:**
- Documentation should be accessible to the target audience
- Examples should be working and realistic
- Information should be current and reflect actual system behavior
- Documentation should be searchable and well-organized
- Visual aids (diagrams, screenshots) should be used where helpful

Focus on documentation that enables successful adoption, development, and operation of the system.
"""
