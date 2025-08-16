"""Testing validation gate prompts"""

TESTING_GATE_PROMPT = """
Evaluate test coverage and quality for confident production deployment.

**Assessment Criteria:**
- Unit tests cover core functionality with good edge case coverage
- Integration tests validate component interactions and data flows
- End-to-end tests verify critical user workflows and system behavior
- Test quality is high with clear assertions and proper test data
- Testing strategy covers different types of failures and error conditions
- Performance and load testing validates system behavior under stress
- Security testing verifies protection against common vulnerabilities

**For `is_acceptable`:** Only return `true` if test coverage is sufficient to catch issues before production. Inadequate testing leads to production bugs and system failures.

**For `problems`:** List specific gaps in test coverage, areas where testing is insufficient, or test quality issues that need improvement.

**Testing Categories:**
- Unit Testing: Individual functions and components, edge cases, error handling
- Integration Testing: API endpoints, database interactions, service communications
- End-to-End Testing: Complete user workflows, cross-system functionality
- Performance Testing: Load testing, stress testing, scalability validation
- Security Testing: Input validation, authentication, authorization, data protection
- Regression Testing: Automated tests to prevent breaking existing functionality

**Quality Standards:**
- Tests should be maintainable, reliable, and fast to execute
- Test data should be realistic and cover various scenarios
- Tests should clearly document expected behavior and failure modes
- Automated test execution should be integrated into CI/CD pipeline

Focus on test coverage that gives confidence in system reliability and correctness.
"""
