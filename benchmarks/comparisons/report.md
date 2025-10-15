# Apples-to-Apples Agentic Framework Benchmark Report

- Generated: 2025-10-15T14:10:23
- Source summary: `benchmark_summary_20251015_141023.csv`
- Python: 3.10.15 | Runs per case: 20
- Harness: Apples-to-Apples Agentic Framework Benchmark (cold & construct-only)

## basic_primitives — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 182.7       | 222.7    | 176.8-224.6   | 28.2/29.6      | 24.7/26.1      | +0.0%     |
| 2    | langgraph      | 368.2       | 837.7    | 336.3-1143.9  | 58.6/64.0      | 50.0/60.1      | +101.5%   |
| 3    | autogen        | 626.2       | 650.1    | 614.9-830.3   | 101.6/109.2    | 89.2/95.0      | +242.7%   |
| 4    | llamaindex     | 630.0       | 689.8    | 621.4-759.6   | 85.3/90.2      | 77.0/81.5      | +244.8%   |
| 5    | langchain_core | 1488.5      | 1685.7   | 1377.4-2258.1 | 216.5/221.9    | 168.0/176.1    | +714.7%   |

> **LionAGI advantage:** median +101.5% vs #2 in this table.

## basic_primitives — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best   |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | ----------- |
| 1    | langchain_core | 0.0         | 0.1      | 0.0-0.1    | 0.1/0.2        | 0.0/0.0        | +0.0 ms (∞) |
| 2    | autogen        | 0.1         | 0.2      | 0.1-0.2    | 0.0/0.0        | 0.0/0.0        | +0.1 ms (∞) |
| 3    | langgraph      | 0.1         | 0.1      | 0.1-0.1    | 0.0/0.0        | 0.0/0.0        | +0.1 ms (∞) |
| 4    | llamaindex     | 0.1         | 0.1      | 0.0-0.1    | 0.0/0.0        | 0.0/0.0        | +0.1 ms (∞) |
| 5    | lionagi        | 8.9         | 10.0     | 8.3-21.5   | 0.7/0.9        | 0.6/0.8        | +8.9 ms (∞) |

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## data_processing — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 102.1       | 112.9    | 99.2-122.8    | 21.7/23.8      | 18.5/20.5      | +0.0%     |
| 2    | langgraph      | 336.5       | 346.4    | 329.8-362.2   | 57.8/63.0      | 44.0/55.0      | +229.6%   |
| 3    | autogen        | 627.4       | 736.6    | 614.5-765.3   | 103.2/110.9    | 90.3/98.3      | +514.5%   |
| 4    | llamaindex     | 717.0       | 753.6    | 702.3-883.2   | 129.5/135.9    | 111.6/119.5    | +602.3%   |
| 5    | langchain_core | 1367.9      | 1461.8   | 1338.8-1885.3 | 216.2/225.6    | 166.0/177.3    | +1239.8%  |

> **LionAGI advantage:** median +229.6% vs #2 in this table.

## data_processing — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best       |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | --------------- |
| 1    | langgraph      | 0.3         | 0.4      | 0.3-0.4    | 0.0/0.0        | 0.0/0.0        | +0.0 ms (×1)    |
| 2    | lionagi        | 0.3         | 0.3      | 0.3-0.3    | 0.1/0.2        | 0.0/0.1        | +0.0 ms (×1)    |
| 3    | autogen        | 0.5         | 0.6      | 0.5-0.6    | 0.0/0.1        | 0.0/0.0        | +0.2 ms (×2)    |
| 4    | langchain_core | 3.3         | 3.5      | 3.3-3.6    | 0.2/0.8        | 0.1/0.7        | +3.0 ms (×11)   |
| 5    | llamaindex     | 83.3        | 85.2     | 82.7-87.7  | 43.6/47.9      | 33.9/37.8      | +83.0 ms (×278) |

> **LionAGI advantage:** median +0.0% vs #2 in this table.

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## imports — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)  | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best         |
| ---- | -------------- | ----------- | -------- | ----------- | -------------- | -------------- | ----------------- |
| 1    | langgraph      | 0.1         | 0.1      | 0.1-0.2     | 0.0/0.1        | 0.0/0.1        | +0.0 ms (×1)      |
| 2    | lionagi        | 35.1        | 36.1     | 33.5-37.0   | 10.0/10.4      | 8.4/8.8        | +35.0 ms (×351)   |
| 3    | langchain_core | 52.1        | 60.4     | 50.3-66.9   | 13.6/14.2      | 11.6/12.2      | +52.0 ms (×521)   |
| 4    | autogen        | 687.2       | 767.1    | 664.0-967.6 | 106.0/110.5    | 93.2/98.4      | +687.1 ms (×6872) |
| 5    | llamaindex     | 689.9       | 771.7    | 656.3-925.8 | 89.2/92.4      | 81.2/85.4      | +689.8 ms (×6899) |

> **LionAGI advantage:** median +0.0% vs #2 in this table.

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## orchestrators — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 170.1       | 177.4    | 165.5-207.6   | 26.9/28.5      | 23.3/25.0      | +0.0%     |
| 2    | langgraph      | 329.0       | 339.9    | 324.7-392.6   | 53.4/59.7      | 39.2/53.4      | +93.4%    |
| 3    | autogen        | 668.4       | 713.8    | 618.9-788.3   | 105.9/111.8    | 93.1/97.8      | +292.9%   |
| 4    | llamaindex     | 713.7       | 763.8    | 703.5-870.2   | 130.5/135.1    | 112.6/118.0    | +319.6%   |
| 5    | langchain_core | 1329.7      | 1368.1   | 1315.6-1869.6 | 209.1/218.9    | 158.7/167.7    | +681.7%   |

> **LionAGI advantage:** median +93.4% vs #2 in this table.

## orchestrators — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best       |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | --------------- |
| 1    | langchain_core | 0.1         | 0.1      | 0.1-0.1    | 0.1/0.1        | 0.0/0.0        | +0.0 ms (×1)    |
| 2    | autogen        | 0.2         | 0.3      | 0.1-0.5    | 0.0/0.1        | 0.0/0.0        | +0.1 ms (×2)    |
| 3    | langgraph      | 0.3         | 0.3      | 0.3-0.3    | 0.0/0.0        | 0.0/0.0        | +0.2 ms (×3)    |
| 4    | lionagi        | 8.4         | 8.9      | 8.2-9.9    | 0.6/0.8        | 0.6/0.8        | +8.3 ms (×84)   |
| 5    | llamaindex     | 85.2        | 89.2     | 80.7-90.1  | 45.4/49.7      | 35.7/40.1      | +85.1 ms (×852) |

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## workflow_setup — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 175.5       | 186.0    | 165.7-202.5   | 27.4/30.0      | 23.8/26.4      | +0.0%     |
| 2    | langgraph      | 331.4       | 350.0    | 325.0-388.3   | 55.9/65.9      | 41.9/53.5      | +88.8%    |
| 3    | autogen        | 628.5       | 654.3    | 622.4-736.0   | 104.6/108.5    | 92.0/97.6      | +258.1%   |
| 4    | llamaindex     | 722.3       | 792.3    | 711.2-819.4   | 132.9/138.9    | 115.5/120.6    | +311.6%   |
| 5    | langchain_core | 1448.6      | 1624.8   | 1356.6-1989.6 | 214.0/223.4    | 166.0/177.5    | +725.4%   |

> **LionAGI advantage:** median +88.8% vs #2 in this table.

## workflow_setup — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best       |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | --------------- |
| 1    | langchain_core | 0.1         | 0.2      | 0.1-0.2    | 0.1/0.1        | 0.0/0.0        | +0.0 ms (×1)    |
| 2    | langgraph      | 0.4         | 0.5      | 0.4-0.5    | 0.0/0.0        | 0.0/0.0        | +0.3 ms (×4)    |
| 3    | autogen        | 0.5         | 0.5      | 0.5-0.5    | 0.0/0.0        | 0.0/0.0        | +0.4 ms (×5)    |
| 4    | lionagi        | 8.3         | 8.5      | 8.2-8.6    | 0.7/0.7        | 0.6/0.6        | +8.2 ms (×83)   |
| 5    | llamaindex     | 83.1        | 86.4     | 81.9-88.2  | 45.4/49.2      | 35.7/40.2      | +83.0 ms (×831) |

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## Cold Composite (excludes _imports_)

| Rank | Framework      | Composite median (gmean, ms) | Avg RSS (MB) | Avg USS (MB) | Pairs |
| ---- | -------------- | ---------------------------- | ------------ | ------------ | ----- |
| 1    | lionagi        | 153.6                        | 26.0         | 22.6         | 4     |
| 2    | langgraph      | 340.9                        | 56.4         | 43.8         | 4     |
| 3    | autogen        | 637.4                        | 103.8        | 91.1         | 4     |
| 4    | llamaindex     | 694.7                        | 119.6        | 104.1        | 4     |
| 5    | langchain_core | 1407.3                       | 213.9        | 164.7        | 4     |

> **LionAGI composite advantage:** +121.9% vs #2.

## Cold Composite (excludes _imports_ and _data_processing_)

| Rank | Framework      | Composite median (gmean, ms) | Avg RSS (MB) | Avg USS (MB) | Pairs |
| ---- | -------------- | ---------------------------- | ------------ | ------------ | ----- |
| 1    | lionagi        | 176.0                        | 27.5         | 24.0         | 3     |
| 2    | langgraph      | 342.4                        | 56.0         | 43.7         | 3     |
| 3    | autogen        | 640.7                        | 104.0        | 91.4         | 3     |
| 4    | llamaindex     | 687.4                        | 116.2        | 101.7        | 3     |
| 5    | langchain_core | 1420.6                       | 213.2        | 164.2        | 3     |

> **LionAGI composite advantage:** +94.5% vs #2.

## Summary Statistics

- **Headline (cold, excl. imports)**: 4/4 first-place finishes | **Avg rank:**
  1.0
- **All categories**: 4/9 first-place finishes | **Avg rank:** 2.3

## Feature Parity Matrix

This matrix demonstrates the equivalence of test cases across frameworks:

| Framework      | Object                             | Compiled?      | Core-only | LLM?    | Network? | Notes                                    |
| -------------- | ---------------------------------- | -------------- | --------- | ------- | -------- | ---------------------------------------- |
| lionagi        | Session()                          | N/A            | Yes       | No      | No       | Minimal runtime container                |
| langgraph      | StateGraph(...).compile()          | Yes            | Yes       | No      | No       | One-node identity graph (START→node→END) |
| langchain_core | PromptTemplate                     | RunnableLambda | Yes       | Yes     | No       | No                                       |
| llamaindex     | SimpleChatEngine(MockLLM)          | Yes            | Yes       | MockLLM | No       | Built-in MockLLM; no network             |
| autogen        | ConversableAgent(llm_config=False) | N/A            | Yes       | No      | No       | Single agent; LLM disabled               |

### Documentation Sources

The benchmark design follows official documentation from each framework:

- **LangGraph**:
  [Graph API & compile()](https://langchain-ai.github.io/langgraph/reference/graphs/)
- **LangChain Core**:
  [PromptTemplate](https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html)
  |
  [RunnableLambda](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)
- **LlamaIndex**:
  [SimpleChatEngine](https://docs.llamaindex.ai/en/stable/api_reference/chat_engines/simple/)
  |
  [MockLLM](https://docs.llamaindex.ai/en/stable/understanding/evaluating/cost_analysis/)
- **AutoGen**:
  [ConversableAgent](https://microsoft.github.io/autogen/0.2/docs/reference/agentchat/conversable_agent/)
  (llm_config=False)
- **Memory Metrics**: [psutil RSS/USS](https://psutil.readthedocs.io/)

## Methodology

### Measurement Approach

- **Cold Mode**: Full import + object construction (serverless scenario)
- **Construct Mode**: Object construction only (post-import, per-request cost)
- **Memory**: RSS (Resident Set Size) and USS (Unique Set Size) deltas
- **Statistics**: Median, P95, MAD, trimmed mean for robustness against outliers

### Environmental Controls

- CPU pinning for reduced scheduler noise
- Deterministic hashing (PYTHONHASHSEED=0)
- Module cache clearing between runs
- Process isolation per measurement
