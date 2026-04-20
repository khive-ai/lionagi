# Team Coordination

`li team` gives agents in a flow a persistent inbox for mid-run signals. The team
file survives at `~/.lionagi/teams/<id>.json` after the flow exits.

## Setup

```bash
pip install lionagi          # or: uv add lionagi
# claude — Option A (subscription): npm install -g @anthropic-ai/claude-code && claude login
#          Option B (API key):      export ANTHROPIC_API_KEY="sk-ant-..."
```

## Create a team

```bash
li team create "research-review" -m researcher,reviewer,orchestrator
```

```text
# output:
Created team 'research-review' (e7a3d91bf542)
  Members: researcher, reviewer, orchestrator
  File: ~/.lionagi/teams/e7a3d91bf542.json
```

## Run a flow with team mode

```bash
li o flow claude/sonnet \
  "Research Python async patterns, then review the draft for completeness" \
  --team-mode research-review --save ~/team-out
```

```text
# output:
Team 'research-review' created (c4f8b2a01e73)
  ▶ researcher started
  ✓ researcher done (14.2s)
  ▶ reviewer started
  ✓ reviewer done (9.1s)
Saved to /home/user/team-out
[...agent results...]
```

`--team-mode research-review` creates a fresh team per run with that name and
injects `TEAM_COORD_SECTION` into each worker's system prompt.

## Mid-flow signals

Workers call `li team send` while their op is still running:

```bash
li team send \
  "Draft at ~/team-out/agents/researcher/research.md — ready for review" \
  -t research-review --to reviewer --from researcher --from-op o1
```

```text
# output:
Sent to reviewer in 'research-review'
```

The reviewer reads its inbox before starting its own op:

```bash
li team receive -t research-review --as reviewer
```

```text
# output:
[2026-04-20T14:03:08] researcher op=o1 → reviewer
  Draft at ~/team-out/agents/researcher/research.md — ready for review

(1 message)
```

`--as reviewer` marks the message read for that member only. Concurrent sends
serialize under `fcntl.flock` — parallel workers don't clobber each other.

## Inspect the inbox

After the flow, check the full thread:

```bash
li team show research-review
```

```text
# output:
Team: research-review (c4f8b2a01e73)
Created: 2026-04-20T14:02:55.123456+00:00
Members: orchestrator, researcher, reviewer

────────────────────────────────────────────────────────────
  [2026-04-20T14:03:08] researcher op=o1 → reviewer  (read by: reviewer)
    Draft at ~/team-out/agents/researcher/research.md — ready for review

  [2026-04-20T14:04:31] reviewer op=o2 → *
    Review complete. 2 gaps noted in ~/team-out/agents/reviewer/review.md

```

Run `li team list` to see all teams.

## How it works

`--team-mode research-review` creates a team at flow start and injects
`TEAM_COORD_SECTION` into each worker's system prompt — the agent knows its team
id, role, and teammates' names. Workers call `li team send --from-op <id>` to tag
which op the signal belongs to; one agent can run several ops on the same branch
and each message stays traceable. `li team receive --as <name>` returns only
unread messages for that member and timestamps them as read. At flow end, each
worker's result is also posted as a team message, so `li team show` shows
mid-flow signals and final outputs in one thread.

## Next

- [Resumable background](resumable-background.md) — run team flows overnight
- [CLI reference: `li team`](../cli-reference.md#li-team) — all flags
