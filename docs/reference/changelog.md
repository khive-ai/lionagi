# Changelog

All notable changes to LionAGI are documented here.

## [0.20.0] - 2026-02-13

### Added

- **Core primitives hardening** (krons-informed Tier 1 + Tier 2 migration):
  - `NodeConfig` dataclass and `create_node` factory for typed node creation
  - Node lifecycle methods: `freeze()`, `unfreeze()`, `validate_content()`
  - `Flow` container with ordered progression tracking
  - `Broadcaster` pub/sub pattern with topic-based subscriptions
  - Graph algorithms: `topological_sort()`, `find_path()`, `get_tails()`
  - `Event.completion_event` — lazily-created `asyncio.Event` signalled on terminal status transitions (COMPLETED, FAILED, CANCELLED, ABORTED, SKIPPED)
  - `Event.status` property setter with automatic `completion_event` signalling
  - `EventStatus.ABORTED` status
  - `Execution.add_error()` with `ExceptionGroup` accumulation (capped at 100 errors)
  - `Execution.retryable` flag for safe retry signalling
  - `Event.assert_completed()` for status assertions
  - `Pile.filter()` with callable predicates
  - `Progression.popleft()` — O(1) via deque-backed `order` field
  - `Progression.__contains__` — O(1) membership via internal set
  - Deprecation warnings for silently-ignored legacy params in `step.py` and `operate.py`
  - Comprehensive documentation overhaul with 187 doc example tests
  - Migration guides for AutoGen, CrewAI, and LangChain
  - Enterprise-focused cookbook examples
  - Integration guides for tools, vector stores, and MCP servers

### Changed

- **`Progression.order`**: Migrated from `list` to `collections.deque` for O(1) `popleft()`
- **`iModel.invoke()`**: Replaced busy-polling loop with `asyncio.wait_for(completion_event.wait())` for efficient async waiting
- **`load_pydantic_model_from_schema()`**: Rewritten to use `pydantic.create_model()` instead of `exec_module()` code generation (CWE-94 remediation)
- **`HookedEvent`**: Terminal status transitions now use `self.status` property setter to signal `completion_event`
- `raise exc` → `raise exc from cause` across 39 call sites (B904 compliance)
- 11 `print()` statements replaced with `logging.getLogger(__name__)`
- Simplified tool integration: functions can be passed directly without `func_to_tool` wrapper
- Documentation style updated to focus on practical, production-ready patterns

### Fixed

- **TokenCalculator**: Always resolve `encoding_name` via `get_encoding_name()` before checking if tokenizer is callable, fixing content token counting
- **MCP `ConnectionPool`**: Lazy `asyncio.Lock` initialization to avoid event-loop-bound instantiation
- **Broadcaster**: Strong references for subscriber lambdas to prevent premature GC
- **ExceptionGroup serialization**: Depth cap (100) and cycle detection in `Execution.to_dict()`

### Security

- Removed CWE-94 code injection vector in `load_pydantic_model_from_schema` — no longer generates and executes Python source at runtime

## [0.19.2] - 2025-12-15

### Added

- CLI provider documentation: CLIEndpoint architecture, request parameters, session management
- Context management docs for session rotation

### Changed

- Documentation improvements across CLI provider coverage

## [0.15.11] - 2025-08-24

### Added

- `extract_json` and `fuzzy_json` functions in new `lionagi.ln` module for
  robust JSON handling
- Availability check functions for optional dependencies:
  `check_docling_available`, `check_networkx_available`,
  `check_matplotlib_available`
- `list_adapters` method in Pile class for adapter enumeration
- Content serialization and validation methods in Node class
- New concurrency-related classes exported in `__all__` for better accessibility

### Changed

- **Performance**: Replaced standard `json` with `orjson` for faster JSON
  operations
- **PostgreSQL Adapter**: Major cleanup and refactoring (374 lines removed, 88
  added) with enhanced table creation logic
- **Utils Refactoring**: Moved utilities from monolithic `utils.py` to organized
  `lionagi.ln` module (393 lines removed)
- **Node Serialization**: Updated adaptation methods to use `as_jsonable`
  instead of custom serialization
- **Element Methods**: Refactored serialization methods and enhanced `to_dict`
  functionality
- **MessageManager**: Simplified methods by leveraging `filter_by_type`
  functionality
- **Type Consistency**: Updated Progression class type variable from `E` to `T`
- Updated pydapter dependency to v1.0.2

### Fixed

- Parameter name in MessageManager methods: `reversed` → `reverse`
- Import statement for `fix_json_string` in test files
- Output examples in `persist_to_postgres_supabase` notebook
- Docling import handling in ReaderTool initialization
- Item type validation improvements in Pile class

### Removed

- **Package Management Module**: Deleted entire `lionagi.libs.package` module
  (138 lines removed)
  - Removed `imports.py`, `management.py`, `params.py`, `system.py`
- Redundant import statements and dead code cleanup
- StepModel and related tests from step module

## [0.15.9] - 2025-08-20

### Added

- JSON serialization utilities with orjson support
- Enhanced Element class with orjson-based JSON serialization methods
- User serialization method in Session class

### Changed

- **JSON Performance**: Replaced `json.dumps` with `ln.json_dumps` using orjson
  for consistent, faster serialization
- **EventStatus Refactoring**: Updated to use `ln.Enum` with improved JSON
  serialization
- **CI/CD**: Upgraded actions/checkout to v5, removed documentation build
  workflow

## [0.15.8] - 2025-08-20

### Changed

- Lowered psutil dependency requirements for broader compatibility

## [0.15.7] - 2025-08-18

### Fixed

- Enhanced Params initialization to validate allowed keys
- Parameter validation improvements

## [0.15.6] - 2025-08-18

### Added

- `aicall_params` to register_operation for async parallel execution support

### Fixed

- Flow execution refactored to use `alcall` for improved concurrency handling
- Updated `alcall` and `bcall` parameter handling with better kwargs support
- Import statements for ConcurrencyEvent and Semaphore consistency

## [0.15.5] - 2025-08-17

### Added

- `aiofiles` dependency for async file operations
- Utility functions for union type handling and type annotations
- Async PostgreSQL adapter registration with availability checks

### Changed

- Enhanced Pile class with validation and serialization methods
- Refactored PostgreSQL adapter checks into utility functions

### Fixed

- Parameter name correction: `strict` → `strict_type` in Pile initialization
- Exception handling: `TypeError` → `ValidationError` in collection validation
- Explicit boolean checks for async PostgreSQL availability

## [0.15.4] - 2025-08-17

### Added

- User serialization functionality in Branch class

## [0.15.3] - 2025-08-16

### Added

- Comprehensive tests for operation cancellation and edge conditions
- SKIPPED status to EventStatus for better execution tracking

### Changed

- Execution status set to CANCELLED for cancelled API calls
- Enhanced operation handling with edge condition validation and filtering
  aggregation metadata

### Fixed

- Flow regression issues in operation execution
- Parameter handling cleanup and improved cancellation error handling

## [0.15.2] - 2025-08-16

### Added

- Operation decorator to simplify function registration as operations
- Comprehensive tests for Session.operation() decorator functionality

### Changed

- Updated author information in README

## [0.15.1] - 2025-08-16

### Added

- Mock operation methods for improved async operation handling in tests
- `to_dict` method to Execution class for better serialization

### Changed

- Integrated OperationManager into Branch and Session classes for enhanced
  operation management
- Simplified OperationManager initialization with enhanced operation
  registration logic
- Enhanced Execution class response serialization handling

### Fixed

- Import statement cleanup across multiple files for consistency

## [0.14.11] - 2025-08-14

### Added

- Updated operation_builder notebook to demonstrate graph serialization

### Fixed

- Graph serialization/deserialization in Graph class
- Field parameters in Operation class
- Redundant file mode in open() calls

### Changed

- Organized imports in throttle.py, cleaned up unused imports in test files

## [0.14.10] - 2025-08-08

### Added

- XML parsing and conversion utilities with new XMLParser class

### Fixed

- Hook registry async function calls with proper await usage
- Import path for to_num utility in test files
- Error handling improvements in HookRegistry

## [0.13.7] - 2025-07-21

### Added

- Notebook for sequential analysis of academic claims using Operation Graphs

### Removed

- `action_batch_size` parameter from operate and branch methods

### Changed

- Updated documentation to reflect parameter changes
