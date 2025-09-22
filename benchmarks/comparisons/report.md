# Apples-to-Apples Agentic Framework Benchmark Report

- Generated: 2025-09-22T18:22:17
- Source summary: `benchmark_summary_20250922_182217.csv`
- Python: 3.10.15 | Runs per case: 20
- Harness: Apples-to-Apples Agentic Framework Benchmark (cold & construct-only)

## basic_primitives — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 343.0       | 456.1    | 315.0-457.6   | 48.2/51.5      | 43.1/46.4      | +0.0%     |
| 2    | langgraph      | 430.9       | 513.1    | 409.3-604.5   | 64.9/69.7      | 57.9/63.5      | +25.6%    |
| 3    | llamaindex     | 725.6       | 848.6    | 664.3-1087.8  | 94.9/100.9     | 85.7/90.5      | +111.5%   |
| 4    | autogen        | 1253.9      | 1365.6   | 1183.8-1517.6 | 144.8/151.1    | 122.7/129.1    | +265.6%   |
| 5    | langchain_core | 2278.3      | 2691.6   | 2150.8-3142.9 | 247.3/262.3    | 189.6/206.7    | +564.2%   |

> **LionAGI advantage:** median +25.6% vs #2 in this table.

## basic_primitives — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best     |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | ------------- |
| 1    | langchain_core | 0.1         | 0.1      | 0.0-0.3    | 0.1/0.1        | 0.0/0.0        | +0.0 ms (×1)  |
| 2    | langgraph      | 0.1         | 0.1      | 0.1-0.1    | 0.0/0.0        | 0.0/0.0        | +0.0 ms (×1)  |
| 3    | llamaindex     | 0.1         | 0.1      | 0.1-0.1    | 0.0/0.0        | 0.0/0.0        | +0.0 ms (×1)  |
| 4    | autogen        | 0.2         | 0.2      | 0.1-0.2    | 0.0/0.1        | 0.0/0.0        | +0.1 ms (×2)  |
| 5    | lionagi        | 1.1         | 1.4      | 1.0-3.6    | 0.0/0.0        | 0.0/0.0        | +1.0 ms (×11) |

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## data_processing — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 92.5        | 103.2    | 85.3-116.7    | 20.5/22.0      | 17.7/19.2      | +0.0%     |
| 2    | langgraph      | 416.8       | 461.7    | 395.2-541.6   | 64.5/71.7      | 55.8/60.7      | +350.6%   |
| 3    | llamaindex     | 825.4       | 930.5    | 791.8-1024.0  | 140.7/146.0    | 122.4/125.5    | +792.3%   |
| 4    | autogen        | 1258.4      | 1339.2   | 1197.5-1455.6 | 141.9/150.1    | 121.9/127.4    | +1260.4%  |
| 5    | langchain_core | 2166.0      | 2631.5   | 2105.7-2950.3 | 250.1/259.6    | 187.2/199.2    | +2241.6%  |

> **LionAGI advantage:** median +350.6% vs #2 in this table.

## data_processing — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best       |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | --------------- |
| 1    | lionagi        | 0.3         | 0.4      | 0.3-0.4    | 0.1/0.2        | 0.0/0.1        | +0.0 ms (×1)    |
| 2    | langgraph      | 0.4         | 0.4      | 0.3-0.5    | 0.0/0.0        | 0.0/0.0        | +0.1 ms (×1)    |
| 3    | autogen        | 0.5         | 0.5      | 0.5-0.5    | 0.0/0.0        | 0.0/0.0        | +0.2 ms (×2)    |
| 4    | langchain_core | 3.4         | 3.9      | 3.2-3.9    | 0.1/0.1        | 0.0/0.0        | +3.1 ms (×11)   |
| 5    | llamaindex     | 92.9        | 111.5    | 85.4-132.9 | 44.8/47.1      | 35.4/38.9      | +92.6 ms (×310) |

> **LionAGI advantage:** median +33.3% vs #2 in this table.

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## imports — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best          |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | ------------------ |
| 1    | langgraph      | 0.2         | 1.1      | 0.1-12.9      | 0.0/0.1        | 0.0/0.1        | +0.0 ms (×1)       |
| 2    | lionagi        | 38.8        | 51.8     | 36.3-78.5     | 10.3/10.5      | 8.6/8.9        | +38.6 ms (×194)    |
| 3    | langchain_core | 54.3        | 108.0    | 51.6-121.2    | 14.0/14.8      | 11.9/12.7      | +54.1 ms (×271)    |
| 4    | llamaindex     | 760.5       | 1222.1   | 716.6-1631.9  | 95.8/98.7      | 86.8/90.1      | +760.3 ms (×3802)  |
| 5    | autogen        | 1289.1      | 1720.3   | 1214.0-2176.6 | 141.3/146.4    | 121.7/125.7    | +1288.9 ms (×6445) |

> **LionAGI advantage:** median +0.0% vs #2 in this table.

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## orchestrators — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 329.1       | 431.3    | 303.4-443.7   | 48.9/52.0      | 44.0/46.6      | +0.0%     |
| 2    | langgraph      | 420.4       | 460.2    | 411.3-493.2   | 65.0/71.4      | 57.4/63.4      | +27.7%    |
| 3    | llamaindex     | 832.9       | 1130.7   | 758.7-1260.5  | 141.3/147.1    | 122.1/130.1    | +153.1%   |
| 4    | autogen        | 1242.8      | 1363.2   | 1191.3-1485.3 | 144.4/153.1    | 122.9/132.1    | +277.6%   |
| 5    | langchain_core | 2266.4      | 2791.9   | 2194.6-3132.2 | 246.2/257.5    | 190.2/198.8    | +588.7%   |

> **LionAGI advantage:** median +27.7% vs #2 in this table.

## orchestrators — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best       |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | --------------- |
| 1    | langchain_core | 0.1         | 0.1      | 0.1-0.2    | 0.1/0.1        | 0.0/0.0        | +0.0 ms (×1)    |
| 2    | autogen        | 0.2         | 0.2      | 0.1-0.4    | 0.0/0.0        | 0.0/0.0        | +0.1 ms (×2)    |
| 3    | langgraph      | 0.3         | 0.4      | 0.3-0.6    | 0.0/0.0        | 0.0/0.0        | +0.2 ms (×3)    |
| 4    | lionagi        | 1.2         | 1.6      | 1.1-1.7    | 0.0/0.1        | 0.0/0.1        | +1.1 ms (×12)   |
| 5    | llamaindex     | 89.8        | 95.5     | 82.3-102.7 | 44.6/47.7      | 35.4/37.8      | +89.7 ms (×898) |

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## workflow_setup — **cold**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms)    | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best |
| ---- | -------------- | ----------- | -------- | ------------- | -------------- | -------------- | --------- |
| 1    | lionagi        | 311.4       | 365.3    | 288.6-564.1   | 47.1/50.2      | 41.9/46.0      | +0.0%     |
| 2    | langgraph      | 430.7       | 529.8    | 401.2-560.0   | 61.7/65.8      | 55.3/59.4      | +38.3%    |
| 3    | llamaindex     | 816.1       | 845.7    | 780.7-1114.6  | 141.6/146.0    | 122.9/126.3    | +162.1%   |
| 4    | autogen        | 1252.4      | 1515.9   | 1167.4-1540.7 | 142.7/156.3    | 121.3/129.4    | +302.2%   |
| 5    | langchain_core | 2173.5      | 2301.7   | 2076.3-2957.5 | 250.7/261.1    | 186.9/197.7    | +598.0%   |

> **LionAGI advantage:** median +38.3% vs #2 in this table.

## workflow_setup — **construct**

| Rank | Framework      | Median (ms) | p95 (ms) | Range (ms) | RSS μ/max (MB) | USS μ/max (MB) | Δ vs best       |
| ---- | -------------- | ----------- | -------- | ---------- | -------------- | -------------- | --------------- |
| 1    | langchain_core | 0.1         | 0.2      | 0.1-0.4    | 0.1/0.1        | 0.0/0.0        | +0.0 ms (×1)    |
| 2    | langgraph      | 0.4         | 0.5      | 0.4-0.5    | 0.0/0.0        | 0.0/0.0        | +0.3 ms (×4)    |
| 3    | autogen        | 0.5         | 0.6      | 0.5-0.6    | 0.0/0.1        | 0.0/0.0        | +0.4 ms (×5)    |
| 4    | lionagi        | 1.2         | 1.4      | 1.0-1.5    | 0.1/0.1        | 0.0/0.1        | +1.1 ms (×12)   |
| 5    | llamaindex     | 91.7        | 95.7     | 86.4-95.8  | 45.3/48.8      | 35.6/38.8      | +91.6 ms (×917) |

> ⚠️ Baseline < 1.0 ms; results dominated by timer granularity and lazy imports.
> Excluded from headline composites.

## Cold Composite (excludes _imports_)

| Rank | Framework      | Composite median (gmean, ms) | Avg RSS (MB) | Avg USS (MB) | Pairs |
| ---- | -------------- | ---------------------------- | ------------ | ------------ | ----- |
| 1    | lionagi        | 238.8                        | 41.2         | 36.7         | 4     |
| 2    | langgraph      | 424.7                        | 64.0         | 56.6         | 4     |
| 3    | llamaindex     | 798.8                        | 129.6        | 113.3        | 4     |
| 4    | autogen        | 1251.9                       | 143.4        | 122.2        | 4     |
| 5    | langchain_core | 2220.5                       | 248.6        | 188.5        | 4     |

> **LionAGI composite advantage:** +77.8% vs #2.

## Cold Composite (excludes _imports_ and _data_processing_)

| Rank | Framework      | Composite median (gmean, ms) | Avg RSS (MB) | Avg USS (MB) | Pairs |
| ---- | -------------- | ---------------------------- | ------------ | ------------ | ----- |
| 1    | lionagi        | 327.6                        | 48.1         | 43.0         | 3     |
| 2    | langgraph      | 427.3                        | 63.9         | 56.9         | 3     |
| 3    | llamaindex     | 790.1                        | 125.9        | 110.2        | 3     |
| 4    | autogen        | 1249.7                       | 144.0        | 122.3        | 3     |
| 5    | langchain_core | 2238.9                       | 248.1        | 188.9        | 3     |

> **LionAGI composite advantage:** +30.4% vs #2.

## Summary Statistics

- **Headline (cold, excl. imports)**: 4/4 first-place finishes | **Avg rank:**
  1.0
- **All categories**: 5/9 first-place finishes | **Avg rank:** 2.2

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
