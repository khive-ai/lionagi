# Agentic AI Framework Performance Benchmarks

Comparative performance analysis of major Python agentic AI frameworks, focusing on cold-start performance and memory usage.

## Executive Summary

Benchmark comparison of five Python agentic AI frameworks measuring cold-start performance and memory usage.

Key findings:
- Core object creation: LionAGI 158ms, LangGraph 266ms, AutoGen 898ms
- Practical setup time: LionAGI 276ms, AutoGen 904ms, LangChain 1925ms
- Memory usage: LionAGI 27-50MB, LangGraph 45-49MB, AutoGen 125-145MB
- Import performance: LangGraph 0.1ms, LionAGI 35ms, AutoGen 882ms

## Benchmark Results

*20 runs per test • Python 3.10.15 • Process isolation per run*

### Import Performance
| Framework | Median (ms) | Range | Memory (MB) |
|-----------|-------------|-------|-------------|
| **LangGraph** | 0.1 | 0.1-0.1 | 0.0 |
| **LionAGI** | 34.9 | 32.4-55.5 | 10.1 |
| **LangChain** | 59.3 | 57.2-72.5 | 15.6 |
| **LlamaIndex** | 535.0 | 504.9-784.5 | 92.9 |
| **AutoGen** | 881.6 | 857.2-1314.7 | 130.6 |

### Core Object Creation
| Framework | Median (ms) | Relative | Memory (MB) | Use Case |
|-----------|-------------|----------|-------------|----------|
| **LionAGI** | 158.0 | 1.0x | 26.6 | AI model abstraction |
| **LangGraph** | 266.4 | 1.7x | 48.5 | State machine |
| **LlamaIndex** | 540.7 | 3.4x | 91.0 | Document processing |
| **AutoGen** | 898.3 | 5.7x | 125.6 | Agent conversation |
| **LangChain** | 2030.0 | 12.8x | 245.7 | Prompt template |

### High-Level Orchestration
| Framework | Median (ms) | Relative | Memory (MB) | Capabilities |
|-----------|-------------|----------|-------------|--------------|
| **LangGraph** | 250.4 | 0.89x | 45.3 | Graph compilation |
| **LionAGI** | 280.8 | 1.0x | 44.0 | Builder + Session |
| **AutoGen** | 844.3 | 3.0x | 121.3 | Group chat setup |
| **LangChain** | 1931.0 | 6.9x | 242.9 | Runnable sequence |
| **LlamaIndex** | 2258.5 | 8.0x | 282.6 | Settings config |

### Practical Setup
| Framework | Median (ms) | Relative | Memory (MB) | Lines of Code |
|-----------|-------------|----------|-------------|---------------|
| **LionAGI** | 276.3 | 1.0x | 42.6 | 6 lines |
| **AutoGen** | 904.2 | 3.3x | 129.4 | 8 lines |
| **LangGraph** | 1921.6 | 7.0x | 241.6 | 15 lines |
| **LangChain** | 1924.7 | 7.0x | 240.6 | 7 lines |
| **LlamaIndex** | 2504.1 | 9.1x | 323.5 | 9 lines |

## Performance Analysis

### Observations

**Import Performance**
- LangGraph: 0.1ms
- LionAGI and LangChain: 35-59ms
- LlamaIndex: 535ms
- AutoGen: 882ms

**Object Creation Patterns**
- LionAGI: 158ms for core iModel object
- LangGraph: 266ms for state machine setup
- LlamaIndex: 541ms for document indexing setup
- AutoGen: 898ms for conversation agent
- LangChain: 2030ms for prompt template

**Memory Footprint**
- LionAGI: 27-50MB
- LangGraph: 45-49MB
- AutoGen: 120-145MB
- LangChain: 240-250MB
- LlamaIndex: 280-340MB

**Practical Setup Times**
- LionAGI: 276ms (6 lines)
- AutoGen: 904ms (8 lines)
- LangGraph: 1922ms (15 lines)
- LangChain: 1925ms (7 lines)
- LlamaIndex: 2504ms (9 lines)

### Framework Characteristics

| Framework | Primary Use Case | Strengths | Considerations |
|-----------|-----------------|-----------|----------------|
| **LionAGI** | Multi-model orchestration | Consistent performance, low memory | Moderate import time |
| **LangGraph** | State-based workflows | Minimal import overhead | Higher setup complexity |
| **LangChain** | Ecosystem integration | Extensive tooling | Performance overhead |
| **AutoGen** | Multi-agent systems | Conversation patterns | High resource usage |
| **LlamaIndex** | Document processing | Specialized for RAG | Large memory footprint |

## Methodology

### Test Environment
- **Platform**: macOS
- **Python Version**: 3.10.15
- **Runs Per Test**: 20 independent runs
- **Process Isolation**: Fresh Python interpreter per run
- **Memory Measurement**: RSS (Resident Set Size) tracked within subprocess
- **Cache Management**: Module cache cleared between runs

### Test Scenarios

1. **Import Performance**: Time to import the framework module
2. **Core Object Creation**: Initialization of fundamental framework objects
3. **High-Level Orchestration**: Setup of production-ready components
4. **Practical Setup**: Minimal code to send message to AI

### Statistical Approach
- Median values reported as primary metric (more stable than mean)
- Standard deviation included to show consistency
- Min/max range provided for worst-case analysis

## Reproducing Results

```bash
# Clone and setup
git clone https://github.com/lion-agi/lionagi.git
cd lionagi/benchmarks/comparisons

# Install dependencies
uv add --dev langgraph langchain langchain-community
uv add --dev llama-index-core pyautogen psutil

# Run benchmark (20 runs, ~10 minutes)
uv run benchmark_professional.py

# View results
ls benchmark_*
```

## Use Case Considerations

### Serverless Deployments
- Import time and memory usage directly impact cold start performance and costs
- Frameworks with sub-100ms import times suitable for latency-sensitive applications
- Memory footprint affects container sizing and concurrent execution limits

### Development Workflow
- Faster initialization times provide quicker feedback during development
- Lower memory usage allows running more tests in parallel
- Code complexity (lines required) affects maintainability

## Summary

The benchmarks show distinct performance profiles across frameworks:

- **LangGraph**: Minimal import overhead (0.1ms) with moderate setup times
- **LionAGI**: Low setup times (158-276ms) with moderate memory usage (27-50MB)
- **AutoGen**: Higher initialization costs (900ms) focused on multi-agent conversations
- **LangChain**: Extensive ecosystem with corresponding performance overhead (2000ms+)
- **LlamaIndex**: Document-processing specialized with largest resource requirements

Framework selection should consider specific use case requirements including cold-start sensitivity, memory constraints, feature needs, and development complexity.

---

### Raw Data
- **Summary CSV**: Statistical analysis across all frameworks
- **Detailed CSV**: Individual run data with memory tracking
- **JSON Export**: Complete benchmark metadata and results

*Last updated: September 2025 • LionAGI v1.0 • [Benchmark methodology](./benchmark_professional.py)*
