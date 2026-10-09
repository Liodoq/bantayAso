"""Show what Bantay has learned from your teaching (read-only).

    python scripts\\check_examples.py

For each taught action: how many examples are saved and whether they are ACTIVE.
An action only influences recognition when at least two actions are taught, it has
3+ examples, and every one of its examples is closer to its own action than to any
other taught action. One confusing example (e.g. a "sitting" frame that looks like
"lying down") switches that action OFF - the report names the worst example so you
can delete that action's file and re-teach it with clearer poses.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bantayaso import config                          # noqa: E402
from bantayaso.examples import reference_gates       # noqa: E402


def main() -> None:
    d = config.DATA_DIR / "actions"
    files = sorted(d.glob("*.npy")) if d.exists() else []
    if not files:
        print("No taught actions yet. Teach at least TWO different actions (3+ examples each).")
        return
    ex = {f.stem.replace("_", " "): np.load(f, allow_pickle=False) for f in files}
    gates = reference_gates(ex)
    print(f"{'action':<24}{'examples':>9}   status")
    for lab, rows in sorted(ex.items()):
        if len(ex) < 2:
            why = "inactive - teach a second, different action"
        elif len(rows) < 3:
            why = f"inactive - needs {3 - len(rows)} more example(s)"
        elif lab in gates:
            why = f"ACTIVE   (own-match >= {gates[lab][0]:.2f}, margin {gates[lab][1]:.3f})"
        else:
            others = np.vstack([v for k, v in ex.items() if k != lab])
            same = rows @ rows.T
            np.fill_diagonal(same, -np.inf)
            marg = same.max(1) - (rows @ others.T).max(1)
            worst = int(np.argmin(marg))
            near = max((k for k in ex if k != lab), key=lambda k: float((rows[worst] @ ex[k].T).max()))
            why = (f"OFF - example #{worst + 1} looks more like '{near}'. "
                   f"Delete data\\actions\\{lab.replace(' ', '_')}.npy and re-teach clearer poses.")
        print(f"{lab:<24}{len(rows):>9}   {why}")


if __name__ == "__main__":
    main()
