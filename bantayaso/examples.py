"""Conservative reference matching. No extra model and no weight training.

Acceptance bounds come from leave-one-out reference similarities, not identity thresholds.
They are a safeguard, NOT a substitute for evaluation on separately recorded clips.
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np


def save_array(path: Path, array: np.ndarray) -> None:
    """Replace a complete collection; a failed write never leaves a partial .npy."""
    tmp = path.with_suffix('.npy.tmp')
    try:
        with tmp.open('wb') as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def reference_gates(examples: dict) -> dict:
    """Require own-class agreement and separation from every other taught class.

    With fewer than two classes or three references per candidate, abstain. Bounds use
    the weakest leave-one-out nearest match and smallest own-vs-other margin. Overlapping
    reference sets therefore disable that class rather than force a decision.
    """
    if len(examples) < 2:
        return {}
    gates = {}
    for label, rows in examples.items():
        others = [v for k, v in examples.items() if k != label and len(v)]
        if len(rows) < 3 or not others:
            continue
        same = rows @ rows.T
        np.fill_diagonal(same, -np.inf)
        positive = same.max(axis=1)
        negative = (rows @ np.vstack(others).T).max(axis=1)
        floor = float(positive.min())
        margin = float((positive - negative).min())
        if floor > 0 and margin > 1e-6:
            gates[label] = (floor, margin)
    return gates


def blend(labels, examples, gates, p_text, feat):
    if not gates:
        return p_text
    scores = {lab: float(np.max(rows @ feat)) for lab, rows in examples.items() if len(rows)}
    if len(scores) < 2:
        return p_text
    ranked = sorted(scores, key=scores.get, reverse=True)
    winner = ranked[0]
    if winner not in gates or winner not in labels:
        return p_text
    floor, margin = gates[winner]
    best, runner = scores[winner], scores[ranked[1]]
    if best + 1e-6 < floor or best - runner + 1e-6 < margin:
        return p_text
    # Influence decreases with similarity, even after the strict reference gate passes.
    weight = 0.6 * np.clip(best, 0., 1.)
    result = (1. - weight) * p_text.copy()
    result[labels.index(winner)] += weight
    return result / result.sum()
