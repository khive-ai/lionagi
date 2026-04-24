# Commit conventions

When the orchestrator decides a commit is warranted:

1. Stage only the files you intentionally changed — never `git add -A`.
2. Use Conventional Commits: `type(scope): summary` where `type` is
   `feat | fix | refactor | test | docs | chore`.
3. Subject line ≤ 72 chars. Body explains the *why*, not the *what*.
4. Never commit generated artifacts, secrets, or large binaries.
5. Never run `git push --force` or `git reset --hard` without explicit user
   approval.

Example:

```
feat(cli): li play sugar for playbook invocation

Wraps `li o flow -p NAME ...` in a short form. Also handles `li play list`.
```
