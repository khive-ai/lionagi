# benchmark_professional.py
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import textwrap
from datetime import datetime
from string import Template

RUNS = 20

# Comprehensive benchmark categories
benchmark_categories = {
    "imports": {
        "description": "Framework import time (cold start critical)",
        "cases": {
            "lionagi": "import importlib, time; t=time.perf_counter(); importlib.import_module('lionagi'); print((time.perf_counter()-t)*1000)",
            "langgraph": "import importlib, time; t=time.perf_counter(); importlib.import_module('langgraph'); print((time.perf_counter()-t)*1000)",
            "langchain": "import importlib, time; t=time.perf_counter(); importlib.import_module('langchain'); print((time.perf_counter()-t)*1000)",
            "llamaindex": "import importlib, time; t=time.perf_counter(); importlib.import_module('llama_index.core'); print((time.perf_counter()-t)*1000)",
            "autogen": "import importlib, time; t=time.perf_counter(); importlib.import_module('autogen'); print((time.perf_counter()-t)*1000)",
        },
    },
    "basic_primitives": {
        "description": "Core object creation (fundamental building blocks)",
        "cases": {
            "lionagi": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from lionagi import iModel
                i = iModel(provider="openai", model="gpt-4")
                print((time.perf_counter()-t)*1000)
            """
            ),
            "langgraph": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from typing_extensions import TypedDict
                from langgraph.graph import StateGraph
                class S(TypedDict): x: int
                g = StateGraph(S)
                print((time.perf_counter()-t)*1000)
            """
            ),
            "langchain": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from langchain.prompts import PromptTemplate
                prompt = PromptTemplate.from_template("Hello {name}")
                print((time.perf_counter()-t)*1000)
            """
            ),
            "llamaindex": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from llama_index.core.schema import Document
                d = Document(text="test")
                print((time.perf_counter()-t)*1000)
            """
            ),
            "autogen": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from autogen import ConversableAgent
                agent = ConversableAgent(
                    name="test",
                    llm_config=False,
                    human_input_mode="NEVER"
                )
                print((time.perf_counter()-t)*1000)
            """
            ),
        },
    },
    "orchestrators": {
        "description": "High-level orchestration objects (enterprise-grade)",
        "cases": {
            "lionagi": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from lionagi import Builder, Session
                b = Builder()
                s = Session()
                print((time.perf_counter()-t)*1000)
            """
            ),
            "langgraph": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from typing_extensions import TypedDict
                from langgraph.graph import StateGraph, START, END
                class S(TypedDict): x: int
                def node(s: S): return s
                g = StateGraph(S).add_node("node", node).add_edge(START, "node").add_edge("node", END).compile()
                print((time.perf_counter()-t)*1000)
            """
            ),
            "langchain": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from langchain.prompts import PromptTemplate
                from langchain_community.llms.fake import FakeListLLM
                from langchain.schema.output_parser import StrOutputParser

                prompt = PromptTemplate.from_template("Hello {name}")
                llm = FakeListLLM(responses=["test"])
                parser = StrOutputParser()
                chain = prompt | llm | parser
                print((time.perf_counter()-t)*1000)
            """
            ),
            "llamaindex": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from llama_index.core import Settings
                from llama_index.core.llms import MockLLM
                Settings.llm = MockLLM()
                print((time.perf_counter()-t)*1000)
            """
            ),
            "autogen": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from autogen import ConversableAgent, GroupChat, GroupChatManager
                agent1 = ConversableAgent("agent1", llm_config=False, human_input_mode="NEVER")
                agent2 = ConversableAgent("agent2", llm_config=False, human_input_mode="NEVER")
                groupchat = GroupChat(agents=[agent1, agent2], messages=[], max_round=1)
                manager = GroupChatManager(groupchat=groupchat, llm_config=False)
                print((time.perf_counter()-t)*1000)
            """
            ),
        },
    },
    "hello_world": {
        "description": "Minimum code to send 'Hello World' to AI (practical usability)",
        "cases": {
            "lionagi": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from lionagi import Branch, iModel

                # Pick a model
                gpt4o = iModel(provider="openai", model="gpt-4o-mini")

                # Create a Branch (conversation context)
                hunter = Branch(
                    system="you are a helpful assistant",
                    chat_model=gpt4o,
                )
                # Ready to: await hunter.communicate("Hello World")
                print((time.perf_counter()-t)*1000)
            """
            ),
            "langgraph": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from typing_extensions import TypedDict
                from langgraph.graph import StateGraph, START, END
                from langchain_community.llms.fake import FakeListLLM

                # Define state and chat function
                class State(TypedDict):
                    messages: list

                def chat_node(state):
                    llm = FakeListLLM(responses=["Hello World"])
                    return {"messages": state["messages"] + ["Hello World"]}

                # Build and compile graph
                graph = (StateGraph(State)
                    .add_node("chat", chat_node)
                    .add_edge(START, "chat")
                    .add_edge("chat", END)
                    .compile())
                # Ready to: graph.invoke({"messages": ["user input"]})
                print((time.perf_counter()-t)*1000)
            """
            ),
            "langchain": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from langchain.prompts import PromptTemplate
                from langchain_community.llms.fake import FakeListLLM

                # Create LLM and prompt template
                llm = FakeListLLM(responses=["Hello World"])
                prompt = PromptTemplate.from_template("Respond to: {input}")

                # Create chain
                chain = prompt | llm
                # Ready to: chain.invoke({"input": "Hello World"})
                print((time.perf_counter()-t)*1000)
            """
            ),
            "llamaindex": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from llama_index.core import Settings
                from llama_index.core.llms import MockLLM
                from llama_index.core.chat_engine import SimpleChatEngine

                # Configure LLM
                Settings.llm = MockLLM()

                # Create chat engine
                engine = SimpleChatEngine.from_defaults()
                # Ready to: engine.chat("Hello World")
                print((time.perf_counter()-t)*1000)
            """
            ),
            "autogen": textwrap.dedent(
                """
                import time
                t = time.perf_counter()
                from autogen import ConversableAgent

                # Create conversable agent
                assistant = ConversableAgent(
                    name="assistant",
                    system_message="You are a helpful assistant.",
                    llm_config=False,
                    human_input_mode="NEVER"
                )
                # Ready to: assistant.generate_reply(messages=[{"role": "user", "content": "Hello World"}])
                print((time.perf_counter()-t)*1000)
            """
            ),
        },
    },
}


def create_true_cold_start_code(original_code):
    """Wrap the benchmark snippet with timing + memory measurement."""
    cleaned_code = textwrap.dedent(original_code)
    cleaned_code = re.sub(
        r"^\s*print\(\s*\(time\.perf_counter\(\)\s*-\s*t\)\s*\*1000\s*\)\s*$",
        "",
        cleaned_code,
        flags=re.MULTILINE,
    ).strip()

    template = Template(
        """
import gc
import json
import os
import psutil
import sys
import time

gc.collect()

if hasattr(sys, "modules"):
    targets = {"lionagi", "langgraph", "langchain", "llama_index", "autogen"}
    for module_name in list(sys.modules):
        if any(target in module_name for target in targets):
            del sys.modules[module_name]

process = psutil.Process(os.getpid())
memory_before = process.memory_info().rss / (1024 * 1024)

start = time.perf_counter()
$CODE
end = time.perf_counter()

memory_after = process.memory_info().rss / (1024 * 1024)
memory_delta = memory_after - memory_before

print(json.dumps({
    "execution_time_ms": round((end - start) * 1000, 6),
    "memory_before_mb": round(memory_before, 6),
    "memory_after_mb": round(memory_after, 6),
    "memory_delta_mb": round(memory_delta, 6)
}))
"""
    )

    # Ensure the embedded code is separated from the template context.
    code_block = cleaned_code + ("\n" if cleaned_code else "")
    return template.substitute(CODE=code_block)


def run_case(label, code):
    individual_runs = []
    times = []
    memory_deltas = []

    # Create the cold start version of the code
    cold_start_code = create_true_cold_start_code(code)

    for run_number in range(RUNS):
        try:
            # Each run gets completely fresh Python process
            start_time = datetime.now()
            result = subprocess.run(
                [sys.executable, "-c", cold_start_code],
                capture_output=True,
                text=True,
                check=True,
                # Ensure completely fresh environment
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            end_time = datetime.now()

            # Parse JSON output (last non-empty line to ignore stray prints)
            stdout_lines = [
                line for line in result.stdout.splitlines() if line.strip()
            ]
            if not stdout_lines:
                raise RuntimeError("No benchmark output produced")

            data = json.loads(stdout_lines[-1])

            execution_time = data["execution_time_ms"]
            memory_before = data["memory_before_mb"]
            memory_after = data["memory_after_mb"]
            memory_delta = data["memory_delta_mb"]

            times.append(execution_time)
            memory_deltas.append(memory_delta)

            # Store individual run data
            individual_runs.append(
                {
                    "framework": label,
                    "run_number": run_number + 1,
                    "execution_time_ms": round(execution_time, 3),
                    "memory_before_mb": round(memory_before, 3),
                    "memory_after_mb": round(memory_after, 3),
                    "memory_delta_mb": round(memory_delta, 3),
                    "wall_clock_start": start_time.isoformat(),
                    "wall_clock_end": end_time.isoformat(),
                    "wall_clock_duration_ms": round(
                        (end_time - start_time).total_seconds() * 1000, 3
                    ),
                }
            )

        except subprocess.CalledProcessError as e:
            print(f"ERROR in {label} run {run_number + 1}: {e.stderr}")
            individual_runs.append(
                {
                    "framework": label,
                    "run_number": run_number + 1,
                    "execution_time_ms": None,
                    "error": e.stderr,
                    "memory_before_mb": None,
                    "memory_after_mb": None,
                    "memory_delta_mb": None,
                    "wall_clock_start": None,
                    "wall_clock_end": None,
                    "wall_clock_duration_ms": None,
                }
            )
        except Exception as e:
            print(
                f"UNEXPECTED ERROR in {label} run {run_number + 1}: {str(e)}"
            )
            individual_runs.append(
                {
                    "framework": label,
                    "run_number": run_number + 1,
                    "execution_time_ms": None,
                    "error": str(e),
                    "memory_before_mb": None,
                    "memory_after_mb": None,
                    "memory_delta_mb": None,
                    "wall_clock_start": None,
                    "wall_clock_end": None,
                    "wall_clock_duration_ms": None,
                }
            )

    if not times:
        return None, individual_runs

    return {
        "framework": label,
        "mean_ms": round(statistics.mean(times), 1),
        "median_ms": round(statistics.median(times), 1),
        "stdev_ms": round(statistics.pstdev(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "mean_memory_mb": round(statistics.mean(memory_deltas), 3),
        "max_memory_mb": round(max(memory_deltas), 3),
        "successful_runs": len(times),
        "total_runs": RUNS,
    }, individual_runs


def main():
    print(f"🦁 Agentic AI Framework Benchmark — {RUNS} runs")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    all_results = []
    all_individual_runs = []

    for category_name, category_data in benchmark_categories.items():
        print(f"\n📊 {category_data['description']}")
        print("=" * 110)
        print(
            f"{'Framework':<12} {'Median (ms)':<12} {'Min (ms)':<10} {'Max (ms)':<10} {'StdDev':<8} {'Range':<15} {'Memory Usage':<25}"
        )
        print("-" * 110)

        category_results = []
        for framework, code in category_data["cases"].items():
            result, individual_runs = run_case(framework, code)
            if result:
                category_results.append((result, individual_runs))

        # Sort by median for better presentation
        category_results.sort(key=lambda x: x[0]["median_ms"])

        for result, individual_runs in category_results:
            framework = result["framework"]
            median = result["median_ms"]
            min_val = result["min_ms"]
            max_val = result["max_ms"]
            stddev = result["stdev_ms"]
            range_val = f"{min_val:.1f}-{max_val:.1f}"

            # Use memory stats from result
            if result.get("mean_memory_mb") is not None:
                mean_memory = result["mean_memory_mb"]
                max_memory = result["max_memory_mb"]
                memory_info = f"{mean_memory:.1f}MB (max: {max_memory:.1f}MB)"
            else:
                memory_info = "N/A"

            print(
                f"{framework:<12} {median:<12.1f} {min_val:<10.1f} {max_val:<10.1f} {stddev:<8.1f} {range_val:<15} {memory_info}"
            )

            # Memory stats are already in result from run_case function

            # Add category info for summary CSV
            result["category"] = category_name
            result["category_description"] = category_data["description"]
            all_results.append(result)

            # Add category info for detailed CSV
            for run in individual_runs:
                run["category"] = category_name
                run["category_description"] = category_data["description"]
            all_individual_runs.extend(individual_runs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export summary CSV
    summary_csv = f"benchmark_summary_{timestamp}.csv"
    with open(summary_csv, "w", newline="") as csvfile:
        fieldnames = [
            "category",
            "category_description",
            "framework",
            "mean_ms",
            "median_ms",
            "stdev_ms",
            "min_ms",
            "max_ms",
            "mean_memory_mb",
            "max_memory_mb",
            "successful_runs",
            "total_runs",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    # Export detailed CSV with all individual runs
    detailed_csv = f"benchmark_detailed_{timestamp}.csv"
    with open(detailed_csv, "w", newline="") as csvfile:
        fieldnames = [
            "category",
            "category_description",
            "framework",
            "run_number",
            "execution_time_ms",
            "memory_before_mb",
            "memory_after_mb",
            "memory_delta_mb",
            "wall_clock_start",
            "wall_clock_end",
            "wall_clock_duration_ms",
            "error",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for run in all_individual_runs:
            writer.writerow(run)

    # Export to JSON
    json_filename = f"benchmark_results_{timestamp}.json"
    with open(json_filename, "w") as jsonfile:
        json.dump(
            {
                "metadata": {
                    "runs": RUNS,
                    "python_version": sys.version.split()[0],
                    "timestamp": datetime.now().isoformat(),
                    "description": "Agentic AI Framework Cold-Start Performance Benchmark",
                },
                "categories": benchmark_categories,
                "summary_results": all_results,
                "detailed_results": all_individual_runs,
            },
            jsonfile,
            indent=2,
        )

    print(f"📄 Results exported to:")
    print(f"  📊 Summary: {summary_csv}")
    print(f"  📋 Detailed: {detailed_csv}")
    print(f"  📦 JSON: {json_filename}")


if __name__ == "__main__":
    main()
