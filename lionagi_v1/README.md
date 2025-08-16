# LionAGI v1 (base)

Lean, invariant-aware orchestrator:

- **Branch as a space** (context, summary, capabilities)
- **Morphism** protocol (typed ops with `requires`)
- **OpGraph** (DAG workflows)
- **IPU** (pluggable invariants/observer)
- **EventBus** (node start/finish events)
- **Forms** (assignment DSL with input/output checks)

No Pydantic in the core. Dataclasses with `slots=True`. Keep validation at I/O
boundaries if needed.

## Quick start

```bash
# editable install
pip install -e .

# run tests
pytest -q

# run example
python -m lionagi_v1.examples.basic
```
