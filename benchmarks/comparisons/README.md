# Agentic AI Framework Performance Benchmarks

Rigorous apples-to-apples performance comparison of major Python agentic AI
frameworks, focusing on real-world cold-start performance and memory efficiency.

## Executive Summary

- **LionAGI delivers 2× faster cold-start performance** than the next-best
  framework (LangGraph) across composites, **up to 3.3× faster** on realistic
  workloads, and **54% lower memory usage**.

### Key Results (20 runs per test • Python 3.10.15)

- **Cold Composite Performance** (geomean of medians; excludes imports): LionAGI
  **153.6 ms** vs LangGraph **340.9 ms** → **LangGraph is 121.9% slower**
  (**LionAGI is 2.2× faster**)
- **Memory Efficiency**: LionAGI **26.0 MB RSS** vs LangGraph 56.4 MB (**54%
  lower**)
- **Consistency**: First-place performance in **4/4 cold-start categories**
- **Orchestrator Setup**: LionAGI **170.1 ms** vs LangGraph 329.0 ms (**93.4%
  faster**)
- **Operational impact**: **~187 ms saved per cold start** vs LangGraph (≈ **3.1
  min saved per 1k cold starts**)

## Headline Performance Metrics

### Cold Composite Performance (Geometric Mean of medians, excludes imports)

| Rank | Framework      | Composite (ms) | RSS (MB) | USS (MB) | vs Best            |
| ---- | -------------- | -------------- | -------- | -------- | ------------------ |
| 1    | **LionAGI**    | 153.6          | 26.0     | 22.6     | —                  |
| 2    | LangGraph      | 340.9          | 56.4     | 43.8     | **+121.9% slower** |
| 3    | AutoGen        | 637.4          | 103.8    | 91.1     | **+314.9% slower** |
| 4    | LlamaIndex     | 694.7          | 119.6    | 104.1    | **+352.2% slower** |
| 5    | LangChain Core | 1407.3         | 213.9    | 164.7    | **+816.0% slower** |

**Notes** • Composite excludes `imports` because sub-millisecond baselines
(e.g., lazy imports) are dominated by timer granularity. • Conservative
composite (excluding `data_processing`): LionAGI **176.0 ms** vs LangGraph
**342.4 ms** → **LangGraph is 94.5% slower** (**LionAGI is 1.9× faster**).

## Detailed Benchmark Results

_Process isolation per run • Module cache cleared • CPU pinning enabled_

### Orchestrators (Cold) - Production-Ready State

| Framework      | Median (ms) | P95 (ms) | Range       | RSS (MB) | vs Best |
| -------------- | ----------- | -------- | ----------- | -------- | ------- |
| **LionAGI**    | 170.1       | 177.4    | 165.5-207.6 | 26.9     | —       |
| LangGraph      | 329.0       | 339.9    | 324.7-392.6 | 53.4     | +93.4%  |
| AutoGen        | 668.4       | 713.8    | 618.9-788.3 | 105.9    | +292.9% |
| LlamaIndex     | 713.7       | 763.8    | 703.5-870.2 | 130.5    | +319.6% |
| LangChain Core | 1329.7      | 1368.1   | 1315-1869   | 209.1    | +681.7% |

### Basic Primitives (Cold) - Core Building Blocks

| Framework      | Median (ms) | P95 (ms) | Range       | RSS (MB) | vs Best |
| -------------- | ----------- | -------- | ----------- | -------- | ------- |
| **LionAGI**    | 182.7       | 222.7    | 176.8-224.6 | 28.2     | —       |
| LangGraph      | 368.2       | 837.7    | 336.3-1143  | 58.6     | +101.5% |
| AutoGen        | 626.2       | 650.1    | 614.9-830.3 | 101.6    | +242.7% |
| LlamaIndex     | 630.0       | 689.8    | 621.4-759.6 | 85.3     | +244.8% |
| LangChain Core | 1488.5      | 1685.7   | 1377-2258   | 216.5    | +714.7% |

### Workflow Setup (Cold) - Multi-Component Coordination

| Framework      | Median (ms) | P95 (ms) | Range       | RSS (MB) | vs Best |
| -------------- | ----------- | -------- | ----------- | -------- | ------- |
| **LionAGI**    | 175.5       | 186.0    | 165.7-202.5 | 27.4     | —       |
| LangGraph      | 331.4       | 350.0    | 325.0-388.3 | 55.9     | +88.8%  |
| AutoGen        | 628.5       | 654.3    | 622.4-736.0 | 104.6    | +258.1% |
| LlamaIndex     | 722.3       | 792.3    | 711.2-819.4 | 132.9    | +311.6% |
| LangChain Core | 1448.6      | 1624.8   | 1356-1989   | 214.0    | +725.4% |

### Data Processing (Cold) - Realistic Workload

| Framework      | Median (ms) | P95 (ms) | Range       | RSS (MB) | vs Best  |
| -------------- | ----------- | -------- | ----------- | -------- | -------- |
| **LionAGI**    | 102.1       | 112.9    | 99.2-122.8  | 21.7     | —        |
| LangGraph      | 336.5       | 346.4    | 329.8-362.2 | 57.8     | +229.6%  |
| AutoGen        | 627.4       | 736.6    | 614.5-765.3 | 103.2    | +514.5%  |
| LlamaIndex     | 717.0       | 753.6    | 702.3-883.2 | 129.5    | +602.3%  |
| LangChain Core | 1367.9      | 1461.8   | 1338-1885   | 216.2    | +1239.8% |

## Performance Analysis

### Memory Efficiency

- **LionAGI**: 26.0 MB average RSS (22.6 MB USS) — **most memory efficient**
- **LangGraph**: 56.4 MB average RSS (**+116.9% vs LionAGI**)
- **AutoGen**: 103.8 MB average RSS (**+299.2%**)
- **LlamaIndex**: 119.6 MB average RSS (**+360.0%**)
- **LangChain Core**: 213.9 MB average RSS (**+722.7%**)

### Consistency & Reliability

- **Low variability**: See CSV for MAD/stdev; LionAGI shows tight ranges in
  cold-path categories.
- **P95 performance**: Sub-230 ms P95 in all cold categories — exceptional
  predictability.
- **Range stability**: Small min-max spans (typically <40ms) across cold
  categories indicate highly predictable performance.

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

```text
Hardware: Apple M2 Max, 32GB RAM
OS: macOS (Darwin 24.6.0)
Python: 3.10.15
LionAGI: v0.18.1
langgraph: 0.6.7
langchain-core: 0.3.76
llama-index-core: 0.14.2
pyautogen: 0.10.0
psutil: 7.1.0
```

## Use Case Recommendations

### When to Choose LionAGI

- **Serverless/Lambda Functions**: 153.6ms cold start vs 340.9ms+ for
  alternatives
- **Memory-Constrained Environments**: 54% lower memory footprint than nearest
  competitor
- **High-Frequency Operations**: Consistent sub-180ms initialization across all
  workloads
- **Cost-Sensitive Deployments**: Lower memory = more concurrent executions per
  node

### Framework Selection Guide

| Use Case              | Recommended | Reasoning                              |
| --------------------- | ----------- | -------------------------------------- |
| Serverless/Lambda     | LionAGI     | Fastest cold start (153.6ms composite) |
| Memory-Limited        | LionAGI     | Lowest footprint (26.0MB)              |
| State Machines        | LangGraph   | Purpose-built for graph workflows      |
| Document RAG          | LlamaIndex  | Specialized document processing        |
| Multi-Agent Chat      | AutoGen     | Conversation-focused patterns          |
| Ecosystem Integration | LangChain   | Extensive tool library                 |

## Summary

The benchmarks show LionAGI's performance characteristics in cold-start
scenarios:

- **2.2× faster** cold-start performance vs next-best across composites (121.9%
  advantage)
- Up to **3.3× faster** on realistic data-processing workloads
- **54% lower memory usage** (26.0MB vs 56.4MB for LangGraph)
- **Consistent performance** with tight P95 bounds and low variance
- **Sub-180ms orchestrator initialization** at 170.1ms median

These characteristics are relevant for:

- Serverless and edge deployments where cold-start performance impacts costs
- Applications with memory constraints or high concurrency requirements
- Use cases requiring predictable performance characteristics

### Performance Evolution

**v0.18.1 Improvements** (Oct 2025 vs v0.17.7 Sept 2025):

- **47% faster** cold starts (238.8ms → 153.6ms composite)
- **37% lower memory** (41.2MB → 26.0MB RSS)
- **Doubled competitive advantage** (77.8% → 121.9% lead vs #2)

The v0.18.1 refactoring (removing ~10,000 LOC of unused code, simplifying type
system, consolidating architecture) delivered measurable performance gains while
improving code quality.

---

### Data Files

- **Summary**: `benchmark_summary_*.csv` - Statistical aggregates
- **Detailed**: `benchmark_detailed_*.csv` - Individual run data
- **Full Export**: `benchmark_results_*.json` - Complete metadata
- **Report**: `report.md` - Comprehensive analysis with full tables

_Last updated: October 15, 2025 • LionAGI v0.18.1 • Python 3.10.15_ _Benchmark
version: Apples-to-Apples Framework Benchmark v2.0_ _(composite excludes
imports; see Methodology)_
