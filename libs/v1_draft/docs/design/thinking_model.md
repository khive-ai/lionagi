### Part 1: The Lionagi v1 Thinking Framework: The "Agent Kernel"

The Lionagi v1 base is best understood as an **"Agent Kernel"**—an operating system designed specifically for the secure, robust, and high-performance execution of agentic workflows. Development within V1 should be guided by the following principles:

#### 1. Structured Execution is Law

*   **The Mental Model:** Every operation has a strictly managed lifecycle. Concurrency is hierarchical. If a parent operation fails or is cancelled, all children are immediately and reliably terminated.
*   **The V1 Implementation:** Standardization on AnyIO-backed structured concurrency (`TaskGroup`, `fail_after`, deadline-aware `retry`).
*   **The Imperative:** Unstructured primitives (`asyncio.create_task`, `asyncio.wait_for`) are prohibited. All I/O must be non-blocking (e.g., using `anyio.Path`) to maintain performance.

#### 2. Security: Principle of Least Privilege (PoLP)

*   **The Mental Model:** Assume untrusted workloads. Operations must explicitly declare required resources.
*   **The V1 Implementation:** Capability-based security enforced by the `Runner`. Dynamic rights calculation (`required_rights`).
*   **The Imperative:** Minimize Morphism requirements. Crucially, the system must always **"fail closed"**—if rights cannot be determined securely, execution must halt.

#### 3. Constraints as Guardrails (The IPU)

*   **The Mental Model:** The system constantly monitors itself against global rules (Invariants). Violations halt execution immediately.
*   **The V1 Implementation:** The Invariant Protection Unit (IPU) enforces cross-cutting concerns (latency, data integrity, resource limits).
*   **The Imperative:** Externalize constraints from business logic. Define Invariants rather than adding checks inside Morphisms.

#### 4. Declarative Data Flow and Composability

*   **The Mental Model:** Define *what* the workflow does (data dependencies), not *how* it executes step-by-step.
*   **The V1 Implementation:** `FormSpec` compiled into an `OpGraph`. The interface layer (`BoundOp`, `OpThenPatch`) decouples logic (Morphisms) from state (`Branch.ctx`).
*   **The Imperative:** Keep Morphisms pure—they receive kwargs and return dicts, remaining isolated from the orchestration context.

#### 5. Performance Through Standardization

*   **The Mental Model:** Serialization and validation are critical performance bottlenecks.
*   **The V1 Implementation:** Standardization on `msgspec` for all core data structures.
*   **The Imperative:** Use `msgspec.Struct` exclusively in the core architecture. Avoid mixing serialization libraries (Pydantic, orjson). Use explicit state handling (`Undefined`, `Unset`).
