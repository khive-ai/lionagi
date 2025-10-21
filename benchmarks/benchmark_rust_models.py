"""
Comprehensive benchmarking suite for Rust vs Python model performance.

This suite measures actual performance improvements from Rust acceleration,
providing before/after comparisons with statistical analysis.
"""

import gc
import json
import platform
import random
import statistics
import string
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Import Python implementations of utilities
from lionagi.ln._hash import hash_dict as py_hash_dict
from lionagi.ln._utils import now_utc as py_now_utc
from lionagi.ln.fuzzy._string_similarity import (
    jaro_winkler_similarity as py_jaro_winkler,
)
from lionagi.ln.fuzzy._string_similarity import (
    levenshtein_distance as py_levenshtein,
)
from lionagi.models.field_model import FieldModel as PyFieldModel
from lionagi.models.hashable_model import HashableModel as PyHashableModel

# Import both implementations for comparison
from lionagi.models.operable_model import OperableModel as PyOperableModel

# Try to import Rust implementations
RUST_AVAILABLE = False
try:
    from lionagi_rust import (
        RustFieldStore,
    )
    from lionagi_rust import create_id_v7 as rust_create_id
    from lionagi_rust import hash_dict as rust_hash_dict
    from lionagi_rust import jaro_winkler_similarity as rust_jaro_winkler
    from lionagi_rust import levenshtein_distance as rust_levenshtein
    from lionagi_rust import now_timestamp as rust_now_timestamp

    from lionagi.models.rust_accelerated import (
        FastFieldModel,
        FastHashableModel,
        FastOperableModel,
    )

    RUST_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Rust extensions not available: {e}")
    FastOperableModel = PyOperableModel
    FastFieldModel = PyFieldModel
    FastHashableModel = PyHashableModel


class BenchmarkTimer:
    """High-precision timer for benchmarking."""

    def __init__(self, name: str, iterations: int = 1000):
        self.name = name
        self.iterations = iterations
        self.times: List[float] = []

    def __enter__(self):
        gc.collect()  # Force garbage collection before timing
        return self

    def __exit__(self, *args):
        pass

    def time_function(
        self, func: Callable, *args, **kwargs
    ) -> Dict[str, float]:
        """Time a function with warmup and statistics."""
        # Warmup
        for _ in range(min(100, self.iterations // 10)):
            func(*args, **kwargs)

        # Actual timing
        times = []
        for _ in range(self.iterations):
            start = time.perf_counter_ns()
            func(*args, **kwargs)
            elapsed = time.perf_counter_ns() - start
            times.append(elapsed)

        # Calculate statistics (in microseconds)
        times_us = [t / 1000 for t in times]
        return {
            "mean": statistics.mean(times_us),
            "median": statistics.median(times_us),
            "stdev": statistics.stdev(times_us) if len(times_us) > 1 else 0,
            "min": min(times_us),
            "max": max(times_us),
            "p95": statistics.quantiles(times_us, n=20)[18],  # 95th percentile
            "p99": statistics.quantiles(times_us, n=100)[
                98
            ],  # 99th percentile
        }


class ModelBenchmarks:
    """Benchmarks for model operations."""

    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get system information for benchmark context."""
        return {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "processor": platform.processor(),
            "python_version": sys.version,
            "rust_available": RUST_AVAILABLE,
            "timestamp": datetime.now().isoformat(),
        }

    def benchmark_field_operations(self) -> Dict:
        """Benchmark field add/get/set operations."""
        results = {}
        iterations = 10000

        # Python implementation
        py_model = PyOperableModel()
        timer = BenchmarkTimer("Python Field Ops", iterations)

        # Add field
        results["py_add_field"] = timer.time_function(
            py_model.add_field,
            "test_field",
            value="test_value",
            annotation=str,
        )

        # Get field
        py_model.add_field("getter", value="value")
        results["py_get_field"] = timer.time_function(
            lambda: getattr(py_model, "getter")
        )

        # Set field
        results["py_set_field"] = timer.time_function(
            lambda: setattr(py_model, "getter", "new_value")
        )

        if RUST_AVAILABLE:
            # Rust implementation
            rust_model = FastOperableModel()
            timer = BenchmarkTimer("Rust Field Ops", iterations)

            # Add field
            results["rust_add_field"] = timer.time_function(
                rust_model.add_field,
                "test_field_rust",
                value="test_value",
                annotation=str,
            )

            # Get field
            rust_model.add_field("getter", value="value")
            results["rust_get_field"] = timer.time_function(
                lambda: getattr(rust_model, "getter")
            )

            # Set field
            results["rust_set_field"] = timer.time_function(
                lambda: setattr(rust_model, "getter", "new_value")
            )

            # Calculate speedups
            results["speedup_add"] = (
                results["py_add_field"]["median"]
                / results["rust_add_field"]["median"]
            )
            results["speedup_get"] = (
                results["py_get_field"]["median"]
                / results["rust_get_field"]["median"]
            )
            results["speedup_set"] = (
                results["py_set_field"]["median"]
                / results["rust_set_field"]["median"]
            )

        return results

    def benchmark_serialization(self) -> Dict:
        """Benchmark serialization operations."""
        results = {}
        iterations = 5000

        # Create models with data
        py_model = PyOperableModel()
        for i in range(20):
            py_model.add_field(f"field_{i}", value=f"value_{i}")

        timer = BenchmarkTimer("Python Serialization", iterations)
        results["py_to_dict"] = timer.time_function(py_model.to_dict)

        if RUST_AVAILABLE:
            rust_model = FastOperableModel()
            for i in range(20):
                rust_model.add_field(f"field_{i}", value=f"value_{i}")

            timer = BenchmarkTimer("Rust Serialization", iterations)
            results["rust_to_dict"] = timer.time_function(rust_model.to_dict)

            results["speedup"] = (
                results["py_to_dict"]["median"]
                / results["rust_to_dict"]["median"]
            )

        return results

    def benchmark_hashing(self) -> Dict:
        """Benchmark hash operations."""
        results = {}
        iterations = 5000

        # Test data
        test_dict = {
            "a": 1,
            "b": "string",
            "c": [1, 2, 3],
            "d": {"nested": "dict"},
            "e": True,
        }

        # Python hash
        timer = BenchmarkTimer("Python Hash", iterations)
        results["py_hash"] = timer.time_function(py_hash_dict, test_dict)

        if RUST_AVAILABLE:
            # Rust hash
            timer = BenchmarkTimer("Rust Hash", iterations)
            results["rust_hash"] = timer.time_function(
                rust_hash_dict, test_dict
            )

            results["speedup"] = (
                results["py_hash"]["median"] / results["rust_hash"]["median"]
            )

        return results

    def benchmark_string_operations(self) -> Dict:
        """Benchmark string similarity operations."""
        results = {}
        iterations = 1000

        # Test strings
        s1 = "hello world this is a test"
        s2 = "hallo world this is a text"

        # Python Levenshtein
        timer = BenchmarkTimer("Python Levenshtein", iterations)
        results["py_levenshtein"] = timer.time_function(py_levenshtein, s1, s2)

        # Python Jaro-Winkler
        timer = BenchmarkTimer("Python Jaro-Winkler", iterations)
        results["py_jaro_winkler"] = timer.time_function(
            py_jaro_winkler, s1, s2
        )

        if RUST_AVAILABLE:
            # Rust Levenshtein
            timer = BenchmarkTimer("Rust Levenshtein", iterations)
            results["rust_levenshtein"] = timer.time_function(
                rust_levenshtein, s1, s2
            )

            # Rust Jaro-Winkler
            timer = BenchmarkTimer("Rust Jaro-Winkler", iterations)
            results["rust_jaro_winkler"] = timer.time_function(
                rust_jaro_winkler, s1, s2, 0.1
            )

            results["speedup_levenshtein"] = (
                results["py_levenshtein"]["median"]
                / results["rust_levenshtein"]["median"]
            )
            results["speedup_jaro_winkler"] = (
                results["py_jaro_winkler"]["median"]
                / results["rust_jaro_winkler"]["median"]
            )

        return results

    def benchmark_uuid_generation(self) -> Dict:
        """Benchmark UUID generation."""
        results = {}
        iterations = 10000

        import uuid

        # Python UUID v4
        timer = BenchmarkTimer("Python UUID", iterations)
        results["py_uuid"] = timer.time_function(lambda: str(uuid.uuid4()))

        if RUST_AVAILABLE:
            # Rust UUID v7
            timer = BenchmarkTimer("Rust UUID", iterations)
            results["rust_uuid"] = timer.time_function(rust_create_id)

            results["speedup"] = (
                results["py_uuid"]["median"] / results["rust_uuid"]["median"]
            )

        return results

    def benchmark_timestamp(self) -> Dict:
        """Benchmark timestamp generation."""
        results = {}
        iterations = 10000

        # Python timestamp
        timer = BenchmarkTimer("Python Timestamp", iterations)
        results["py_timestamp"] = timer.time_function(
            lambda: py_now_utc().timestamp()
        )

        if RUST_AVAILABLE:
            # Rust timestamp
            timer = BenchmarkTimer("Rust Timestamp", iterations)
            results["rust_timestamp"] = timer.time_function(rust_now_timestamp)

            results["speedup"] = (
                results["py_timestamp"]["median"]
                / results["rust_timestamp"]["median"]
            )

        return results

    def benchmark_large_scale(self) -> Dict:
        """Benchmark with large number of fields."""
        results = {}
        field_counts = [10, 50, 100, 500]

        for count in field_counts:
            # Python model
            py_model = PyOperableModel()
            start = time.perf_counter()
            for i in range(count):
                py_model.add_field(f"field_{i}", value=i)
            py_time = time.perf_counter() - start

            results[f"py_{count}_fields"] = py_time * 1000  # Convert to ms

            if RUST_AVAILABLE:
                # Rust model
                rust_model = FastOperableModel()
                start = time.perf_counter()
                for i in range(count):
                    rust_model.add_field(f"field_{i}", value=i)
                rust_time = time.perf_counter() - start

                results[f"rust_{count}_fields"] = rust_time * 1000
                results[f"speedup_{count}_fields"] = py_time / rust_time

        return results

    def benchmark_real_world_scenario(self) -> Dict:
        """Benchmark a realistic usage scenario."""
        results = {}
        iterations = 100

        def create_model_workflow():
            """Simulate typical model usage."""
            model = PyOperableModel()

            # Add various fields
            model.add_field("id", value="123456", frozen=True)
            model.add_field("name", value="John Doe")
            model.add_field("email", value="john@example.com")
            model.add_field("age", value=30)
            model.add_field("tags", value=["python", "development"])
            model.add_field("metadata", value={"created": time.time()})

            # Access fields
            _ = model.name
            _ = model.email
            _ = model.age

            # Update fields
            model.name = "Jane Doe"
            model.age = 31

            # Serialize
            data = model.to_dict()

            # Hash
            hash_value = hash(model)

            return data, hash_value

        def create_rust_workflow():
            """Simulate typical Rust model usage."""
            model = FastOperableModel()

            # Add various fields
            model.add_field("id", value="123456", frozen=True)
            model.add_field("name", value="John Doe")
            model.add_field("email", value="john@example.com")
            model.add_field("age", value=30)
            model.add_field("tags", value=["python", "development"])
            model.add_field("metadata", value={"created": time.time()})

            # Access fields
            _ = model.name
            _ = model.email
            _ = model.age

            # Update fields
            model.name = "Jane Doe"
            model.age = 31

            # Serialize
            data = model.to_dict()

            # Hash
            hash_value = hash(model)

            return data, hash_value

        # Python workflow
        timer = BenchmarkTimer("Python Workflow", iterations)
        results["py_workflow"] = timer.time_function(create_model_workflow)

        if RUST_AVAILABLE:
            # Rust workflow
            timer = BenchmarkTimer("Rust Workflow", iterations)
            results["rust_workflow"] = timer.time_function(
                create_rust_workflow
            )

            results["speedup"] = (
                results["py_workflow"]["median"]
                / results["rust_workflow"]["median"]
            )

        return results

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks and compile results."""
        print("🚀 Starting comprehensive benchmarks...")
        print(f"   Platform: {self.system_info['platform']}")
        print(f"   Python: {sys.version.split()[0]}")
        print(f"   Rust Available: {RUST_AVAILABLE}")
        print()

        all_results = {"system_info": self.system_info, "benchmarks": {}}

        benchmarks = [
            ("Field Operations", self.benchmark_field_operations),
            ("Serialization", self.benchmark_serialization),
            ("Hashing", self.benchmark_hashing),
            ("String Operations", self.benchmark_string_operations),
            ("UUID Generation", self.benchmark_uuid_generation),
            ("Timestamp Generation", self.benchmark_timestamp),
            ("Large Scale", self.benchmark_large_scale),
            ("Real World Scenario", self.benchmark_real_world_scenario),
        ]

        for name, benchmark_func in benchmarks:
            print(f"⏱️  Running {name}...")
            try:
                results = benchmark_func()
                all_results["benchmarks"][name] = results
                self._print_results(name, results)
            except Exception as e:
                print(f"   ❌ Error: {e}")
                all_results["benchmarks"][name] = {"error": str(e)}

        # Calculate overall statistics
        all_results["summary"] = self._calculate_summary(
            all_results["benchmarks"]
        )

        return all_results

    def _print_results(self, name: str, results: Dict):
        """Pretty print benchmark results."""
        print(f"\n   📊 {name} Results:")

        for key, value in results.items():
            if key.startswith("speedup"):
                if isinstance(value, (int, float)):
                    print(f"      🚀 {key}: {value:.2f}x faster")
            elif key.startswith("py_") or key.startswith("rust_"):
                if isinstance(value, dict) and "median" in value:
                    print(f"      {key}: {value['median']:.2f}μs (median)")
                elif isinstance(value, (int, float)):
                    print(f"      {key}: {value:.2f}ms")

    def _calculate_summary(self, benchmarks: Dict) -> Dict:
        """Calculate overall summary statistics."""
        speedups = []
        for category, results in benchmarks.items():
            if isinstance(results, dict):
                for key, value in results.items():
                    if key.startswith("speedup") and isinstance(
                        value, (int, float)
                    ):
                        speedups.append(value)

        if speedups:
            return {
                "average_speedup": statistics.mean(speedups),
                "median_speedup": statistics.median(speedups),
                "min_speedup": min(speedups),
                "max_speedup": max(speedups),
                "total_benchmarks": len(speedups),
            }
        return {"message": "No speedup data available (Rust not compiled?)"}

    def save_results(
        self, results: Dict, filepath: str = "benchmark_results.json"
    ):
        """Save benchmark results to file."""
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {filepath}")

    def generate_report(self, results: Dict) -> str:
        """Generate a markdown report of benchmark results."""
        report = []
        report.append("# LionAGI Rust Performance Benchmark Report\n")
        report.append(f"Generated: {results['system_info']['timestamp']}\n")
        report.append(f"Platform: {results['system_info']['platform']}\n")
        report.append(
            f"Python: {results['system_info']['python_version'].split()[0]}\n"
        )
        report.append(
            f"Rust Available: {results['system_info']['rust_available']}\n"
        )

        if RUST_AVAILABLE:
            report.append("\n## Summary\n")
            summary = results.get("summary", {})
            if "average_speedup" in summary:
                report.append(
                    f"- **Average Speedup**: {summary['average_speedup']:.2f}x\n"
                )
                report.append(
                    f"- **Median Speedup**: {summary['median_speedup']:.2f}x\n"
                )
                report.append(
                    f"- **Best Speedup**: {summary['max_speedup']:.2f}x\n"
                )
                report.append(
                    f"- **Worst Speedup**: {summary['min_speedup']:.2f}x\n"
                )

            report.append("\n## Detailed Results\n")

            for category, results in results["benchmarks"].items():
                if "error" in results:
                    continue

                report.append(f"\n### {category}\n")
                report.append(
                    "| Operation | Python (μs) | Rust (μs) | Speedup |\n"
                )
                report.append(
                    "|-----------|------------|-----------|----------|\n"
                )

                # Parse and format results
                for key, value in results.items():
                    if key.startswith("py_") and not key.endswith("_fields"):
                        op_name = key[3:]  # Remove "py_" prefix
                        rust_key = f"rust_{op_name}"
                        speedup_key = f"speedup_{op_name}"

                        if isinstance(value, dict) and "median" in value:
                            py_time = value["median"]
                            rust_time = results.get(rust_key, {}).get(
                                "median", "-"
                            )
                            speedup = results.get(speedup_key, "-")

                            if isinstance(speedup, float):
                                speedup_str = f"{speedup:.2f}x"
                            else:
                                speedup_str = "-"

                            if rust_time != "-":
                                report.append(
                                    f"| {op_name} | {py_time:.2f} | {rust_time:.2f} | {speedup_str} |\n"
                                )
        else:
            report.append("\n## Rust Not Available\n")
            report.append(
                "Rust extensions are not compiled. Only Python benchmarks were run.\n"
            )
            report.append(
                "To enable Rust benchmarks, run: `./build_rust.sh --release`\n"
            )

        return "".join(report)


def main():
    """Run all benchmarks and generate report."""
    benchmarks = ModelBenchmarks()

    # Run benchmarks
    results = benchmarks.run_all_benchmarks()

    # Save JSON results
    benchmarks.save_results(results)

    # Generate and save markdown report
    report = benchmarks.generate_report(results)
    report_path = Path("benchmark_report.md")
    report_path.write_text(report)
    print(f"📄 Report saved to {report_path}")

    # Print summary
    if RUST_AVAILABLE and "summary" in results:
        summary = results["summary"]
        if "average_speedup" in summary:
            print("\n" + "=" * 50)
            print("🏆 BENCHMARK SUMMARY")
            print("=" * 50)
            print(f"Average Speedup: {summary['average_speedup']:.2f}x")
            print(f"Median Speedup:  {summary['median_speedup']:.2f}x")
            print(f"Best Speedup:    {summary['max_speedup']:.2f}x")
            print(f"Total Tests:     {summary['total_benchmarks']}")
            print("=" * 50)
    else:
        print(
            "\n⚠️  Rust extensions not available. Compile with: ./build_rust.sh --release"
        )


if __name__ == "__main__":
    main()
