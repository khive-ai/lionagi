# Copyright (c) 2023-2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0
"""`li team` — persistent team messaging (inbox pattern).

Examples:
    li team create "research-team" -m "researcher,writer,reviewer"
    li team list
    li team send "analyze auth middleware" --team abc123 --to all
    li team send "focus on JWT" --team abc123 --to researcher --from writer
    li team receive --team abc123 --as researcher
    li team show abc123
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from ._persistence import LIONAGI_HOME

TEAMS_DIR = LIONAGI_HOME / "teams"


def _teams_dir() -> Path:
    TEAMS_DIR.mkdir(parents=True, exist_ok=True)
    return TEAMS_DIR


def _load_team(team_id: str) -> dict:
    for p in _teams_dir().glob("*.json"):
        data = json.loads(p.read_text())
        if data["id"] == team_id or data["id"].startswith(team_id):
            return data
        if data.get("name") == team_id:
            return data
    raise FileNotFoundError(f"No team found matching '{team_id}'")


def _save_team(data: dict) -> Path:
    p = _teams_dir() / f"{data['id']}.json"
    p.write_text(json.dumps(data, indent=2, default=str))
    return p


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Commands ─────────────────────────────────────────────────────────────


def cmd_create(args: argparse.Namespace) -> int:
    members = [m.strip() for m in args.members.split(",") if m.strip()]
    if not members:
        print("error: --members requires at least one name", file=sys.stderr)
        return 1

    team_id = uuid4().hex[:12]
    data = {
        "id": team_id,
        "name": args.name,
        "members": members,
        "messages": [],
        "created_at": _now_iso(),
    }
    path = _save_team(data)
    print(f"Created team '{args.name}' ({team_id})")
    print(f"  Members: {', '.join(members)}")
    print(f"  File: {path}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    teams_dir = _teams_dir()
    files = sorted(
        teams_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not files:
        print("No teams.")
        return 0

    for p in files:
        data = json.loads(p.read_text())
        n_msgs = len(data.get("messages", []))
        members = ", ".join(data.get("members", []))
        print(f"  {data['id']}  {data['name']:20s}  [{members}]  {n_msgs} msgs")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    data = _load_team(args.team)
    print(f"Team: {data['name']} ({data['id']})")
    print(f"Created: {data['created_at']}")
    print(f"Members: {', '.join(data['members'])}")

    msgs = data.get("messages", [])
    if not msgs:
        print("\nNo messages.")
        return 0

    print(f"\n{'─' * 60}")
    for msg in msgs:
        to_str = msg["to"] if isinstance(msg["to"], str) else ", ".join(msg["to"])
        read_by = msg.get("read_by", [])
        marker = "" if not read_by else f"  (read by: {', '.join(read_by)})"
        ts = msg.get("timestamp", "")[:19]
        print(f"  [{ts}] {msg['from']} → {to_str}{marker}")
        for line in msg["content"].splitlines():
            print(f"    {line}")
        print()
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    data = _load_team(args.team)
    members = data["members"]

    sender = args.sender or "_cli"
    if sender != "_cli" and sender not in members:
        print(f"warning: '{sender}' is not a team member", file=sys.stderr)

    if args.to.lower() == "all":
        recipients = ["*"]
    else:
        recipients = [r.strip() for r in args.to.split(",") if r.strip()]
        for r in recipients:
            if r not in members:
                print(f"warning: '{r}' is not a team member", file=sys.stderr)

    msg = {
        "id": uuid4().hex[:12],
        "from": sender,
        "to": recipients,
        "content": args.content,
        "timestamp": _now_iso(),
        "read_by": [],
    }
    data["messages"].append(msg)
    _save_team(data)

    to_display = "all" if recipients == ["*"] else ", ".join(recipients)
    print(f"Sent to {to_display} in '{data['name']}'")
    return 0


def cmd_receive(args: argparse.Namespace) -> int:
    data = _load_team(args.team)
    me = args.member

    if me and me not in data["members"]:
        print(f"warning: '{me}' is not a member of '{data['name']}'", file=sys.stderr)

    msgs = data.get("messages", [])
    unread = []
    for msg in msgs:
        if me and me in msg.get("read_by", []):
            continue
        targets = msg["to"]
        if targets == ["*"] or (me and me in targets) or not me:
            unread.append(msg)

    if not unread:
        print("No new messages." if me else "No messages.")
        return 0

    changed = False
    for msg in unread:
        to_str = "all" if msg["to"] == ["*"] else ", ".join(msg["to"])
        ts = msg.get("timestamp", "")[:19]
        print(f"[{ts}] {msg['from']} → {to_str}")
        print(f"  {msg['content']}")
        print()
        if me and me not in msg.get("read_by", []):
            msg["read_by"].append(me)
            changed = True

    if changed:
        _save_team(data)

    print(f"({len(unread)} message{'s' if len(unread) != 1 else ''})")
    return 0


# ── CLI registration ─────────────────────────────────────────────────────


def add_team_subparser(subparsers: argparse._SubParsersAction) -> None:
    team = subparsers.add_parser(
        "team",
        help="Team messaging — send/receive between named agents.",
        description="Persistent inbox-style messaging for agent teams.",
    )
    team_sub = team.add_subparsers(dest="team_command", required=True)

    # create
    cr = team_sub.add_parser("create", help="Create a new team.")
    cr.add_argument("name", help="Team name.")
    cr.add_argument(
        "-m",
        "--members",
        required=True,
        help="Comma-separated member names.",
    )

    # list
    team_sub.add_parser("list", aliases=["ls"], help="List all teams.")

    # show
    sh = team_sub.add_parser("show", help="Show team details and messages.")
    sh.add_argument("team", help="Team ID or name.")

    # send
    snd = team_sub.add_parser("send", help="Send a message to team members.")
    snd.add_argument("content", help="Message content.")
    snd.add_argument("--team", "-t", required=True, help="Team ID or name.")
    snd.add_argument(
        "--to",
        required=True,
        help="Recipients: 'all' or comma-separated names.",
    )
    snd.add_argument("--from", dest="sender", default=None, help="Sender name.")

    # receive
    rcv = team_sub.add_parser("receive", aliases=["recv"], help="Read inbox messages.")
    rcv.add_argument("--team", "-t", required=True, help="Team ID or name.")
    rcv.add_argument("--as", dest="member", default=None, help="Read as this member.")


def run_team(args: argparse.Namespace) -> int:
    cmd = args.team_command
    if cmd == "create":
        return cmd_create(args)
    if cmd in ("list", "ls"):
        return cmd_list(args)
    if cmd == "show":
        return cmd_show(args)
    if cmd == "send":
        return cmd_send(args)
    if cmd in ("receive", "recv"):
        return cmd_receive(args)
    print(f"Unknown team command: {cmd}", file=sys.stderr)
    return 1
