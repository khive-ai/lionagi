## Lionagi V1: Initial Design Document

**Date:** August 27, 2025
**Status:** Draft

### 1. Abstract

Lionagi V1 introduces a foundational rewrite of the Lionagi architecture, shifting from a utility-centric library to an "Agent Kernel" model. V1 is designed as an operating system specialized for the secure, robust, and high-performance execution of autonomous agentic workflows. It integrates principles from OS research, structured concurrency, and capability-based security to provide the rigorous environment required for deploying complex, data-driven operations in production.

### 2. Motivation: The Need for an Agent Kernel

The rapid evolution of autonomous agents and LLM-driven workflows has outpaced the capabilities of existing orchestration frameworks. Current solutions often suffer from critical limitations that hinder their reliability and security:

1.  **Fragile Execution:** Reliance on unstructured concurrency (e.g., `asyncio.create_task`) leads to unpredictable behavior. Failures often leave dangling tasks, causing resource leaks, inconsistent state, and unreliable cleanup.
2.  **Inadequate Security Boundaries:** Security is often an afterthought. Frameworks lack the fine-grained control necessary to constrain agent actions, violating the Principle of Least Privilege (PoLP) and leading to risks like unauthorized data access or uncontrolled resource usage.
3.  **Absence of Guardrails:** Systems struggle to enforce global constraints—such as latency budgets, data integrity rules, or cost limits—across complex, autonomous workflows.
4.  **Performance Bottlenecks:** Reliance on slower serialization methods (like Pydantic or standard JSON) and the accidental use of blocking I/O create significant overhead in high-throughput systems.
5.  **Opaque Data Flow:** Complex workflows often rely on implicit, shared mutable state, making debugging, data provenance, and parallelization difficult.

Lionagi V1 addresses these issues directly. We are not merely building a better orchestration library; we are building the robust operating system required to manage and constrain autonomous agents effectively.

### 3. Design Philosophy: The Agent Kernel

The V1 architecture is guided by five core tenets:

#### 3.1. Structured Execution is Law

Every operation has a strictly managed lifecycle. Concurrency is hierarchical; if a parent operation fails or is cancelled, all children are immediately and reliably terminated. This guarantees resource cleanup and predictable failure propagation.
*   **Implementation:** Standardization on AnyIO and structured concurrency primitives (`TaskGroup`, `fail_after`, deadline-aware `retry`).

#### 3.2. Principle of Least Privilege (PoLP)

Security is foundational. Operations must explicitly declare the resources they require *before* execution. The kernel grants only the minimum necessary capabilities dynamically.
*   **Implementation:** Capability-based security model with dynamic rights calculation (`required_rights`). The system always "fails closed."

#### 3.3. Constraints as Guardrails

The system constantly monitors execution against global rules (Invariants). Violations halt execution immediately. This externalizes cross-cutting concerns from business logic.
*   **Implementation:** The Invariant Protection Unit (IPU), enforcing latency budgets, data schemas, resource limits, and security policies.

#### 3.4. Declarative Data Flow

Workflows are defined by *what* they achieve (data dependencies), not *how* they execute step-by-step. This enables automatic parallelization, high composability, and clear data provenance.
*   **Implementation:** `FormSpec` (declarative definition) compiled into an optimized `OpGraph` (execution plan).

#### 3.5. Performance Through Standardization

Serialization and validation are critical bottlenecks. V1 mandates high-performance tooling across the stack and ensures all I/O is non-blocking.
*   **Implementation:** Standardization on `msgspec` for all core data structures, replacing Pydantic and standard JSON libraries.

### 4. Architecture Overview

The V1 architecture separates the definition, execution, and safety layers.

#### 4.1. Workflow Definition

*   **`FormSpec` / `FlowSpec`:** Immutable, `msgspec`-based structures defining the workflow steps and their data dependencies (inputs/outputs).
*   **`Morphism` / `BaseOp`:** The unit of work (e.g., LLM call, HTTP request). Morphisms are designed to be pure: they receive kwargs and return dicts, remaining isolated from the broader context.

#### 4.2. The Execution Kernel

*   **`Branch`:** The isolated execution context. It holds the mutable state (`ctx`) and the `Capabilities` (the security principal).
*   **`OpGraph`:** A Directed Acyclic Graph (DAG) representing the execution plan, compiled from the `FlowSpec`.
*   **`Runner`:** The core orchestrator (the scheduler). It executes the `OpGraph` within a `Branch`, managing parallelization, dependency resolution, and security enforcement.

#### 4.3. The Interface Layer (Decoupling)

This crucial layer bridges the gap between the pure Morphisms and the shared `Branch.ctx`, ensuring clean decoupling of logic and state.

*   **`BoundOp`:** Wraps a Morphism to pull required inputs from `Branch.ctx` just before execution.
*   **`OpThenPatch`:** Wraps a Morphism to push results back into `Branch.ctx` just after execution.

#### 4.4. The Safety Layer

*   **Capabilities:** Fine-grained permissions (e.g., `fs.read:/data/input.txt`). V1 supports **dynamic rights calculation**, allowing permissions to be determined at runtime based on the input data.
*   **Invariant Protection Unit (IPU):** Enforces global constraints during the execution lifecycle. The IPU validates capabilities, enforces latency budgets (proactively using `fail_after`), verifies data integrity, and monitors resource usage.

### 5. Why Lionagi V1 is Superior

V1 offers significant advantages over traditional orchestration frameworks and contemporary agent libraries by integrating these architectural innovations.

#### 5.1. Production-Grade Reliability via Structured Concurrency

V1 standardizes on the robust concurrency model popularized by Trio and available via AnyIO.
*   **Guaranteed Cleanup:** Tasks are bound to lexical scopes (`TaskGroup`). If any task fails, all siblings are automatically cancelled, and the scope waits for cleanup. This eliminates the resource leaks common in unstructured systems.
*   **Safe Timeouts:** V1 uses scope-based timeouts (`fail_after`) instead of unsafe primitives like `asyncio.wait_for`.
*   **Deadline-Awareness:** All primitives, including the `retry` mechanism, respect the ambient effective deadline, preventing operations from exceeding latency budgets.

#### 5.2. Security by Default via Capabilities

Unlike frameworks with implicit permissions, V1 enforces PoLP through explicit capabilities and the IPU, providing a secure sandbox for agent execution. The "fail closed" design ensures that if security parameters cannot be verified, execution halts.

#### 5.3. Proactive Constraint Enforcement via the IPU

The IPU provides essential guardrails for autonomous execution. By defining invariants (e.g., `LatencyBound`, `ResultShape`), developers can enforce constraints on potentially unreliable operations (like LLM calls) without polluting the business logic. V1 doesn't just monitor constraints; it actively enforces them.

#### 5.4. High Performance and Efficiency

Standardization on `msgspec` offers serialization performance orders of magnitude faster than Pydantic or standard JSON. Combined with a commitment to non-blocking I/O, V1 achieves significantly higher throughput.

#### 5.5. Comparison Table

| Feature | Traditional Approach | Lionagi V1 Approach |
| :--- | :--- | :--- |
| **Concurrency Model** | Unstructured (`asyncio.create_task`). | Structured (AnyIO `TaskGroup`). |
| **Security Model** | Coarse-grained RBAC; often fail open. | Fine-grained Capabilities; always fail closed. |
| **Constraint Enforcement**| Embedded in business logic. | Externalized via the IPU (Invariants). |
| **Latency Control** | Reactive monitoring or unsafe timeouts. | Proactive enforcement (`fail_after`) and deadline-aware `retry`. |
| **Data Flow** | Imperative scripting or implicit state. | Declarative data dependencies (`FormSpec`). |
| **Performance (SerDes)** | Pydantic, standard JSON, or mixed. | Standardization on `msgspec`. |

### 6. Roadmap

The V1 roadmap focuses on stabilizing the kernel, enhancing observability, and expanding execution capabilities.

#### Phase 1: Kernel Stabilization (Immediate Focus)

*   **Base Refactoring:** Integrate structured concurrency primitives into the `Runner`, `EventBus`, and `core.py` operations. Ensure all I/O is non-blocking (e.g., `anyio.Path`).
*   **`msgspec` Consolidation:** Complete the migration of all core utilities (hashing, JSON parsing) to `msgspec`, removing Pydantic and orjson dependencies.
*   **DSL Compiler:** Implement the ergonomic DSL parser (migrated from V0) to compile `a,b->c; c->d` strings directly into V1 `FlowSpec`.
*   **Comprehensive Testing:** Implement adversarial security testing, Property-Based Testing (PBT) for data flow, and stress testing for the `Runner`.

#### Phase 2: Observability and Optimization (Q4 2025)

*   **Observability and Tracing:** Enhance the `EventBus` to integrate with OpenTelemetry (OTel) for detailed execution tracing and structured logging.
*   **IPU Optimization:** Migrate `Branch.ctx` to persistent data structures (e.g., `pyrsistent`) to optimize IPU snapshotting performance.
*   **IPU Enhancements:** Expand the standard library of Invariants (e.g., cost tracking, token usage rate limiting).
*   **Dynamic Control Flow:** Introduce conditional execution capabilities within the `OpGraph` structure.

#### Phase 3: Distributed Execution and Sandboxing (2026)

*   **Durable Execution:** Introduce mechanisms for persisting `Branch` state and `OpGraph` progress to support long-running, resumable workflows.
*   **Multi-Node Orchestration:** Evolve the `Runner` to distribute `OpNode` execution across multiple worker nodes.
*   **Advanced Sandboxing:** Explore advanced sandboxing techniques (e.g., WebAssembly) for executing untrusted or specialized Morphisms securely.