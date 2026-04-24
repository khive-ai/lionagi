# Playbook examples

Install by copying into `~/.lionagi/playbooks/`:

```bash
cp examples/playbooks/*.playbook.yaml ~/.lionagi/playbooks/
```

Then invoke:

```bash
li play minimal "explain decorators"
li play audit --mode security "the auth service"
li play chatgpt-orchestrate --tabs 5 --poll "research M5 chip"
li play list
```

Or via the long form:

```bash
li o flow -p audit --mode security "the auth service"
```

## What's declarative vs. runtime

A playbook YAML is the **declaration** — what to do, with which defaults.
Invoking it produces a **flow** — the actual DAG execution.

## Template interpolation

Inside `prompt:`, three things happen:

1. `{input}` → the positional prompt text from the CLI.
2. `{arg_name}` → a value from the `args:` schema (CLI override > `default`).
3. If no `{...}` placeholders are present, the positional text is
   **appended** with a blank line — mirroring how CC slash commands
   accept extra arguments.

## args: schema vs. argument-hint: fallback

Two ways to declare CLI args:

- **Explicit `args:`** (preferred): typed schema with defaults, help text,
  auto-generated `--help` output.
- **`argument-hint:` string** (CC-compatible fallback): parsed from display
  strings like `'[--tabs N] [--poll]'`. Every `[--flag VALUE]` becomes a
  string arg; every bare `[--flag]` becomes a bool arg. No type coercion.

If both are present, `args:` wins.

## Examples in this directory

- `minimal.playbook.yaml` — prompt only, no args, positional appended
- `audit.playbook.yaml` — typed `args:` schema with defaults + template
- `chatgpt-orchestrate.playbook.yaml` — CC-compatible `argument-hint`
- `persistent-chat.playbook.yaml` — uses `team_attach:` for a thread that
  accumulates history across invocations

## Team modes

Playbooks can declare one of two team behaviors:

```yaml
team_mode: fresh-audit      # FRESH team every invocation (new UUID)
# or
team_attach: ongoing-chat   # ATTACH by name — first use creates, subsequent reuse
```

At the CLI, the equivalent flags are `--team-mode` and `--team-attach`
(mutually exclusive). `--team-attach` never requires a pre-existing team —
it creates-on-miss. If you want strict "team must exist" semantics, run
`li team create NAME -m ...` before invoking the playbook.
