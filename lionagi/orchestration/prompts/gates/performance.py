"""Performance validation gate prompts"""

PERFORMANCE_GATE_PROMPT = """
Evaluate performance characteristics and scalability for production requirements.

**Assessment Criteria:**
- System meets defined performance requirements and SLAs
- Architecture can scale to handle expected load and growth
- Resource utilization is efficient and optimized
- Performance bottlenecks are identified and addressed
- Caching strategies and optimization techniques are properly applied
- Database queries and data access patterns are optimized
- System can handle peak loads without degradation

**For `is_acceptable`:** Only return `true` if performance requirements are met and the system can handle production load. Poor performance affects user experience and system reliability.

**For `problems`:** List specific performance bottlenecks, scalability limitations, or optimization opportunities. Include both current issues and potential future problems.

**Performance Areas:**
- Response Times: API endpoints, database queries, user interface responsiveness
- Throughput: Requests per second, data processing capacity, concurrent user support
- Scalability: Horizontal and vertical scaling capabilities, load distribution
- Resource Efficiency: CPU usage, memory consumption, network bandwidth, storage I/O
- Optimization: Caching layers, query optimization, algorithm efficiency, asset optimization
- Monitoring: Performance metrics, alerting thresholds, capacity planning

Consider both current performance and ability to scale with growth. Include recommendations for performance improvements and monitoring.
"""
