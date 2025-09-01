from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    current: dict, baseline: dict, threshold: float
) -> tuple[bool, str]:
    """Return (ok, report). ok=False if any scenario regresses beyond threshold.

    Threshold is relative increase on median time (e.g., 0.2 = 20%).
    """
    lines = []
    lines.append(
        f"Threshold: {threshold:.0%} (negative = faster, positive = slower)"
    )
    ok = True

    cur_results = current.get("results", {})
    base_results = baseline.get("results", {})

    for name, cur in sorted(cur_results.items()):
        base = base_results.get(name)
        if not base:
            lines.append(f"- {name}: no baseline; skipping")
            continue
        cur_med = float(cur.get("median", 0))
        base_med = float(base.get("median", 0))
        if base_med == 0:
            delta = float("inf") if cur_med > 0 else 0.0
        else:
            delta = (cur_med - base_med) / base_med
        lines.append(
            f"- {name}: median {cur_med:.6f}s vs {base_med:.6f}s -> {delta:+.1%}"
        )
        if delta > threshold:
            ok = False

    return ok, "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="Path to baseline JSON")
    ap.add_argument("--current", required=True, help="Path to current JSON")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Relative regression threshold (e.g., 0.2 = 20%)",
    )
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    current_path = Path(args.current)

    if not baseline_path.exists():
        print(
            f"[ci_compare] Baseline missing: {baseline_path}. Skipping gating."
        )
        return 0
    if not current_path.exists():
        print(f"[ci_compare] Current results missing: {current_path}.")
        return 1

    try:
        base = load(baseline_path)
        cur = load(current_path)
        ok, report = compare(cur, base, args.threshold)
        print(report)
        return 0 if ok else 2
    except Exception as e:
        print(f"[ci_compare] Failed to compare: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
