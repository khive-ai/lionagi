---
name: commit
description: >
  Conventional Commits style guide + safety rules. Pull this before any
  git commit so the message shape and staging rules stay consistent.
allowed-tools: [Bash, Read]
---

# Commit conventions

## Format

```
type(scope): summary

<optional body — explain the why, not the what>
```

**Types**: `feat | fix | refactor | test | docs | chore`

**Subject line**: imperative mood, ≤ 72 characters, no trailing period.

## Staging rules

1. `git add <specific-file>` — never `git add -A` or `git add .`.
2. Inspect `git diff --cached` before committing. If you see a file you did
   not intend to stage, unstage it with `git restore --staged <file>`.
3. Never commit: `.env`, credentials, API keys, large binaries, generated
   output directories.

## Safety invariants

- Never use `--no-verify` or any flag that skips pre-commit hooks.
- Never force-push (`git push --force`, `git push -f`) without explicit
  user approval.
- Never `git reset --hard` to discard work that isn't yours.
- Never `git clean -fd` without explicit user approval.

## Examples

```
feat(cli): add -p/--playbook flag to li o flow

Resolves NAME to ~/.lionagi/playbooks/<NAME>.playbook.yaml and injects the
declared args schema into the parser so flag values aren't eaten by
positional arguments.
```

```
fix(agent): resolve dir layout before flat for profile load

Previously only <name>.md was consulted. Now tries
<name>/<name>.md first; flat form kept for backward compat.
```

```
docs(cli): document li skill and li play
```
