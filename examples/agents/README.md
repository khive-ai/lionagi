# Agent profile examples

Install by copying a directory into `~/.lionagi/agents/`:

```bash
cp -r examples/agents/minimal   ~/.lionagi/agents/
cp -r examples/agents/with-refs ~/.lionagi/agents/
```

Then invoke:

```bash
li agent -a minimal "hello"
li o flow -a with-refs "audit the auth service"
```

## Layouts

- `minimal/` — one-file agent, simplest possible profile
- `with-refs/` — agent with supplementary reference files under `patterns/` and `refs/` that the agent can pull mid-run via `li skill` or direct file reads

## Resolution

The CLI looks for (in order):

1. `<project>/.lionagi/agents/<name>/<name>.md` (directory, project-local)
2. `<project>/.lionagi/agents/<name>.md` (flat, project-local, legacy)
3. `~/.lionagi/agents/<name>/<name>.md` (directory, global)
4. `~/.lionagi/agents/<name>.md` (flat, global, legacy)
