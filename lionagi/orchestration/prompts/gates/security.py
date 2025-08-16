"""Security validation gate prompts"""

SECURITY_GATE_PROMPT = """
Evaluate security requirements and protections for production deployment.

**Assessment Criteria:**
- Authentication mechanisms are properly designed and implemented
- Authorization controls are comprehensive and correctly applied
- Sensitive data is protected with appropriate encryption and access controls
- Input validation prevents injection attacks and malformed data
- Communication channels are secured with proper encryption
- Security best practices are followed throughout the system
- Attack surface is minimized and potential vulnerabilities are addressed

**For `is_acceptable`:** Only return `true` if security requirements are comprehensively addressed and the system is safe for production. Security failures can be catastrophic.

**For `problems`:** List specific security vulnerabilities, missing protections, or security gaps that create risk. Include both implementation issues and design flaws.

**Critical Security Areas:**
- Authentication: Strong password policies, secure session management, multi-factor authentication where appropriate
- Authorization: Role-based access control, principle of least privilege, proper permission boundaries
- Data Protection: Encryption at rest and in transit, secure key management, data privacy compliance
- Input Security: Comprehensive input validation, parameterized queries, output encoding
- Infrastructure: Secure defaults, regular security updates, monitoring and alerting

Be thorough - missing security controls can lead to data breaches and system compromise.
"""
