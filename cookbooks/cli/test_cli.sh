#!/bin/bash
# Test li CLI — common cases and arg combos
# Run one at a time, uncomment as needed

# ═══════════════════════════════════════════════════════════
# li agent — basic
# ═══════════════════════════════════════════════════════════

# 1. Simple agent, no verbose
# uv run li agent claude/sonnet "Say hello in 3 words"

# 2. Agent with verbose
# uv run li agent claude/sonnet "Say hello in 3 words" -v

# 3. Agent with effort in spec
# uv run li agent claude/claude-opus-4-7-high "Say hello in 3 words" -v

# 4. Agent with --effort flag override
# uv run li agent claude/sonnet "Say hello in 3 words" --effort high

# 5. Agent with alias
# uv run li agent claude "Say hello in 3 words"

# 6. Agent with codex
# uv run li agent codex/gpt-5.3-codex-spark "Say hello in 3 words"

# ═══════════════════════════════════════════════════════════
# li agent — resume (run #1 first to get a branch id)
# ═══════════════════════════════════════════════════════════

# 7. Resume by id (no verbose — should NOT show verbose even if original was -v)
# uv run li agent -r BRANCH_ID "what did you just say?"

# 8. Resume by id with verbose
# uv run li agent -r BRANCH_ID "what did you just say?" -v

# 9. Continue last
# uv run li agent claude "what did you just say?" -c

# ═══════════════════════════════════════════════════════════
# li o fanout — homogeneous (Pattern A)
# ═══════════════════════════════════════════════════════════

# 10. Basic fanout, 2 workers, no synthesis, no verbose
# uv run li o fanout claude/sonnet "Name one design pattern. One sentence." -n 2

# 11. Basic fanout with verbose
# uv run li o fanout claude/sonnet "Name one design pattern. One sentence." -n 2 -v

# 12. Fanout with synthesis
# uv run li o fanout claude/sonnet "Name one design pattern. One sentence." -n 2 --with-synthesis

# 13. Fanout 3 workers (default) with synthesis + verbose
# uv run li o fanout claude/sonnet "Name one design pattern. One sentence." --with-synthesis -v

# ═══════════════════════════════════════════════════════════
# li o fanout — heterogeneous workers (Pattern B)
# ═══════════════════════════════════════════════════════════

# 14. Different worker models, synthesis on default model
uv run li o fanout claude/sonnet "Name one design pattern. One sentence." \
    --workers "claude/sonnet, codex/gpt-5.3-codex-spark" --with-synthesis -v

# 15. Different worker models, synthesis on stronger model
# uv run li o fanout claude/sonnet "Name one design pattern. One sentence." \
#     --workers "claude/sonnet, codex/gpt-5.3-codex-spark" \
#     --with-synthesis claude/opus-4-6-medium

# ═══════════════════════════════════════════════════════════
# li o fanout — effort in spec (Pattern C)
# ═══════════════════════════════════════════════════════════

# 16. Effort embedded in model spec
# uv run li o fanout claude/sonnet "Name one design pattern." -n 2 \
#     --with-synthesis claude/opus-4-7-high

# 17. Effort via flag (overrides spec)
# uv run li o fanout claude/sonnet "Name one design pattern." -n 2 \
#     --with-synthesis --effort high

# 18. Mixed effort — workers with different efforts
# uv run li o fanout claude/sonnet "Name one design pattern." \
#     --workers "codex/gpt-5.4-xhigh, claude/sonnet" --with-synthesis

# ═══════════════════════════════════════════════════════════
# li o fanout — output + save
# ═══════════════════════════════════════════════════════════

# 19. JSON output
# uv run li o fanout claude/sonnet "Name one design pattern." -n 2 --output json

# 20. Save to directory
# uv run li o fanout claude/sonnet "Name one design pattern." -n 2 \
#     --with-synthesis --save /tmp/fanout-test

# ═══════════════════════════════════════════════════════════
# li o fanout — resume after fanout
# ═══════════════════════════════════════════════════════════

# 21. Run fanout first (#12), then resume orchestrator
# uv run li agent -r ORC_BRANCH_ID "summarize what we discussed"

# 22. Resume orchestrator with verbose (should show verbose)
# uv run li agent -r ORC_BRANCH_ID "summarize what we discussed" -v

# 23. Resume orchestrator without verbose (should NOT show verbose)
# uv run li agent -r ORC_BRANCH_ID "summarize what we discussed"

# 24. Resume a worker
# uv run li agent -r WORKER_BRANCH_ID "expand on your answer"

# ═══════════════════════════════════════════════════════════
# Edge cases / error handling
# ═══════════════════════════════════════════════════════════

# 25. Gemini with effort (should error)
# uv run li o fanout gemini-code/gemini-3.1-pro-high "hello" -n 2

# 26. Invalid model spec
# uv run li agent invalidmodel "hello"

# 27. Missing model on new session
# uv run li agent "hello"

# 28. Max concurrent
# uv run li o fanout claude/sonnet "Name one pattern." -n 4 --max-concurrent 2 --with-synthesis
