# Agentic AI Framework Performance Benchmarks

Rigorous apples-to-apples performance comparison of major Python agentic AI
frameworks, focusing on real-world cold-start performance and memory efficiency.

## Executive Summary

- LionAGI delivers 25-45% faster cold-start performance than the next-best
  framework (LangGraph) across composites, up to ~4.9× faster on a realistic
  data-processing workload, and ~36% lower memory (RSS).

### Key Results (20 runs per test • Python 3.10.15)

- **Cold Composite Performance** (geomean of medians; excludes imports): LionAGI
  **233 ms** vs LangGraph **421 ms** → **LangGraph is 81% slower** (**LionAGI is
  45% faster**)
- **Memory Efficiency**: LionAGI 40.5MB RSS vs LangGraph 63.3MB (36% lower)
- **Consistency**: First-place performance in 4/4 cold-start categories
- **Orchestrator Setup**: LionAGI 299ms vs LangGraph 410ms (37% faster)
- **Operational impact**: ~**188 ms saved per cold start** vs LangGraph (≈ **3.1
  min saved per 1k cold starts**)

## Headline Performance Metrics

### Cold Composite Performance (Geometric Mean of medians, excludes imports)

| Rank | Framework      | Composite (ms) | RSS (MB) | USS (MB) | vs Best            |
| ---- | -------------- | -------------- | -------- | -------- | ------------------ |
| 1    | **LionAGI**    | 233.0          | 40.5     | 36.0     | —                  |
| 2    | LangGraph      | 420.8          | 63.3     | 54.0     | **+80.6% slower**  |
| 3    | LlamaIndex     | 803.0          | 130.1    | 113.8    | **+244.6% slower** |
| 4    | AutoGen        | 1243.7         | 142.6    | 120.4    | **+433.6% slower** |
| 5    | LangChain Core | 2260.4         | 248.8    | 188.4    | **+870.1% slower** |

**Notes** • Composite excludes `imports` because sub-millisecond baselines
(e.g., lazy imports) are dominated by timer granularity. • Conservative
composite (excluding `data_processing`): LionAGI **322.5 ms** vs LangGraph
**417.2 ms** → **LangGraph is 29% slower** (**LionAGI is 23% faster**).

## Detailed Benchmark Results

_Process isolation per run • Module cache cleared • CPU pinning enabled_

### Orchestrators (Cold) - Production-Ready State

| Framework      | Median (ms) | P95 (ms) | Range     | RSS (MB) | vs Best |
| -------------- | ----------- | -------- | --------- | -------- | ------- |
| **LionAGI**    | 299.4       | 347.1    | 284-412   | 45.7     | —       |
| LangGraph      | 410.1       | 447.0    | 370-509   | 62.9     | +37.0%  |
| LlamaIndex     | 838.3       | 932.2    | 781-972   | 143.0    | +180.0% |
| AutoGen        | 1257.0      | 1347.8   | 1217-1502 | 142.8    | +319.8% |
| LangChain Core | 2302.8      | 2846.5   | 2123-2978 | 248.0    | +669.1% |

### Basic Primitives (Cold) - Core Building Blocks

| Framework      | Median (ms) | P95 (ms) | Range     | RSS (MB) | vs Best |
| -------------- | ----------- | -------- | --------- | -------- | ------- |
| **LionAGI**    | 326.6       | 377.6    | 304-391   | 47.6     | —       |
| LangGraph      | 409.0       | 540.0    | 376-649   | 63.8     | +25.2%  |
| LlamaIndex     | 752.8       | 855.2    | 681-935   | 95.7     | +130.5% |
| AutoGen        | 1228.9      | 1448.3   | 1156-1577 | 144.1    | +276.3% |
| LangChain Core | 2267.0      | 2658.1   | 2070-3209 | 248.4    | +594.1% |

### Workflow Setup (Cold) - Multi-Component Coordination

| Framework      | Median (ms) | P95 (ms) | Range     | RSS (MB) | vs Best |
| -------------- | ----------- | -------- | --------- | -------- | ------- |
| **LionAGI**    | 343.0       | 421.5    | 307-576   | 48.4     | —       |
| LangGraph      | 432.8       | 510.2    | 408-696   | 64.1     | +26.2%  |
| LlamaIndex     | 814.3       | 890.7    | 786-996   | 141.0    | +137.4% |
| AutoGen        | 1248.5      | 1340.9   | 1213-1558 | 141.4    | +264.0% |
| LangChain Core | 2203.8      | 2579.5   | 2083-2720 | 251.4    | +542.5% |

### Data Processing (Cold) - Realistic Workload

| Framework      | Median (ms) | P95 (ms) | Range     | RSS (MB) | vs Best  |
| -------------- | ----------- | -------- | --------- | -------- | -------- |
| **LionAGI**    | 87.8        | 116.5    | 83-131    | 20.2     | —        |
| LangGraph      | 432.0       | 509.8    | 403-541   | 62.4     | +392.0%  |
| LlamaIndex     | 808.9       | 894.8    | 779-902   | 140.7    | +821.3%  |
| AutoGen        | 1240.5      | 1391.1   | 1162-1541 | 142.1    | +1312.9% |
| LangChain Core | 2269.0      | 2600.5   | 2084-2841 | 247.4    | +2484.3% |

## Performance Analysis

### Memory Efficiency

- **LionAGI**: 40.5 MB average RSS (36.0 MB USS) — most memory efficient
- **LangGraph**: 63.3 MB average RSS (**+56% vs LionAGI**)
- **LlamaIndex**: 130.1 MB average RSS (**+221%**)
- **AutoGen**: 142.6 MB average RSS (**+252%**)
- **LangChain Core**: 248.8 MB average RSS (**+514%**)

### Consistency & Reliability

- **Low variability**: See CSV for MAD/stdev; LionAGI shows tight ranges in
  cold-path categories.
- **P95 performance**: Sub-400 ms P95 in most cold categories (orchestrators &
  primitives).
- **Range stability**: Small min-max spans across cold categories indicate
  predictable cold behavior.

## Feature Parity Matrix

All frameworks tested with equivalent, normalized workloads:

| Framework      | Object Built           | Core-Only | LLM Used | Network | Notes                     |
| -------------- | ---------------------- | --------- | -------- | ------- | ------------------------- |
| LionAGI        | Session()              | Yes       | None     | No      | Minimal runtime container |
| LangGraph      | StateGraph.compile()   | Yes       | None     | No      | One-node identity graph   |
| LangChain Core | PromptTemplate\|Lambda | Yes       | None     | No      | LCEL chain, no community  |
| LlamaIndex     | SimpleChatEngine       | Yes       | MockLLM  | No      | Built-in mock, no network |
| AutoGen        | ConversableAgent       | Yes       | None     | No      | LLM disabled              |

## Methodology

### Measurement Approach

- **Cold Mode**: Full import + object construction (serverless scenario)
- **Process Isolation**: Fresh Python interpreter per measurement
- **Statistical Rigor**: Median, P95, MAD, trimmed mean for outlier resistance
- **Memory Tracking**: RSS (Resident Set Size) and USS (Unique Set Size)

### Environmental Controls

- CPU pinning for reduced scheduler noise
- Deterministic hashing (PYTHONHASHSEED=0)
- Module cache clearing between runs
- API keys blanked to prevent network calls
- 30-second timeout to prevent hangs

### Excluded from Headlines

- **Import-only tests**: Dominated by lazy loading and timer granularity
- **Sub-millisecond measurements**: Below meaningful timer resolution

## Reproducing Results

```bash
# Clone repository
git clone https://github.com/lion-agi/lionagi.git
cd lionagi/benchmarks/comparisons

# Install dependencies
uv add --dev langgraph langchain-core llama-index-core pyautogen psutil
# (Recommended) Pin exact versions for reproducibility
uv lock
# Or export a frozen requirements file:
uv export --frozen --format requirements.txt > bench.requirements.txt

# Run full benchmark (20 runs, ~15 minutes)
uv run python benchmark_professional.py --runs 20 --report

# Generate report
uv run python generate_benchmark_report.py

# Quick test (3 runs)
uv run python benchmark_professional.py --runs 3
```

### Environment & Versions

Test environment and package versions used for benchmarks:

```
Hardware: Apple M2 Max, 32GB RAM
OS: macOS (Darwin 24.6.0)
Python: 3.10.15
LionAGI: v0.17.7
langgraph: 0.6.7
langchain-core: 0.3.76
llama-index-core: 0.14.2
pyautogen: 0.10.0
psutil: 7.1.0
```

## Use Case Recommendations

### When to Choose LionAGI

- **Serverless/Lambda Functions**: 233ms cold start vs 421ms+ for alternatives
- **Memory-Constrained Environments**: 36% lower memory footprint
- **High-Frequency Operations**: Consistent sub-100ms data processing
- **Cost-Sensitive Deployments**: Lower memory = more concurrent executions

### Framework Selection Guide

| Use Case              | Recommended | Reasoning                            |
| --------------------- | ----------- | ------------------------------------ |
| Serverless/Lambda     | LionAGI     | Fastest cold start (233ms composite) |
| Memory-Limited        | LionAGI     | Lowest footprint (40.5MB)            |
| State Machines        | LangGraph   | Purpose-built for graph workflows    |
| Document RAG          | LlamaIndex  | Specialized document processing      |
| Multi-Agent Chat      | AutoGen     | Conversation-focused patterns        |
| Ecosystem Integration | LangChain   | Extensive tool library               |

## Summary

The benchmarks show LionAGI's performance characteristics in cold-start
scenarios:

- **25-45% faster** cold-start performance vs next-best across composites
- Up to **~4.9× faster** on the data-processing workload
- **36% lower memory usage** (40.5MB vs 63.3MB for LangGraph)
- **Consistent performance** with tight P95 bounds and low variance
- **Fast orchestrator initialization** at 299ms median

These characteristics are relevant for:

- Serverless and edge deployments where cold-start performance impacts costs
- Applications with memory constraints or high concurrency requirements
- Use cases requiring predictable performance characteristics

---

### Data Files

- **Summary**: `benchmark_summary_*.csv` - Statistical aggregates
- **Detailed**: `benchmark_detailed_*.csv` - Individual run data
- **Full Export**: `benchmark_results_*.json` - Complete metadata

_Last updated: September 2025 • LionAGI v0.17.7 • Python 3.10.15_ _Benchmark
version: Apples-to-Apples Framework Benchmark v2.0_ _(composite excludes
imports; see Methodology)_
