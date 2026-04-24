# Skill examples

Install by copying a directory into `~/.lionagi/skills/`:

```bash
cp -r examples/skills/hello ~/.lionagi/skills/
cp -r examples/skills/commit ~/.lionagi/skills/
```

Then invoke:

```bash
li skill hello          # print body to stdout
li skill list           # list installed skills
li skill show hello     # print full file (frontmatter + body)
```

## CC interop

The format is identical to Claude Code skills (`~/.claude/skills/<name>/SKILL.md`).
You can symlink one source into both roots to share a single file across CC
and lionagi agents:

```bash
ln -s ~/.lionagi/skills/hello ~/.claude/skills/hello
```

Or reverse — keep CC as source of truth and symlink into `~/.lionagi/skills/`.

## Use from agents

An orchestrator agent can shell out mid-run:

```
$ li skill empaco
```

… and inject the stdout back into its own context. Zero extra protocol.

## Layouts

- `hello/` — simplest skill
- `commit/` — conventions-style reference with structured body
