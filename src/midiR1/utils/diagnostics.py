"""Generation-side diagnostics for MIDI token sequences.

All functions operate on plain Python lists of token IDs or symusic Score
objects — no model or tokenizer needed at call time.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from symusic import Score


# ── Token-level metrics ─────────────────────────────────────────────────────


def ngram_repetition_rate(token_ids: Sequence[int], n: int = 3) -> float:
    """Fraction of *n*-grams that are duplicates.

    Returns 0.0 when the sequence is too short to contain any *n*-gram,
    and 1.0 when every *n*-gram has been seen before.
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    total = len(ngrams)
    unique = len(set(ngrams))
    return 1.0 - unique / total


def token_entropy(token_ids: Sequence[int]) -> float:
    """Shannon entropy (bits) of the token distribution in the sequence.

    Higher entropy → more diverse token usage.
    """
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    total = len(token_ids)
    return -sum(
        (c / total) * math.log2(c / total) for c in counts.values()
    )


# ── Note-level metrics (decoded MIDI) ───────────────────────────────────────


def note_density(score: Score) -> float:
    """Average number of notes per beat across all tracks.

    Returns 0.0 for empty scores or scores with zero duration.
    """
    total_notes = sum(len(t.notes) for t in score.tracks)
    if total_notes == 0:
        return 0.0

    end_tick = score.end()
    if end_tick == 0:
        return 0.0

    tpq = score.ticks_per_quarter
    total_beats = end_tick / tpq
    return total_notes / total_beats


def pitch_range(score: Score) -> dict:
    """Pitch statistics across all tracks.

    Returns ``{"min": int, "max": int, "range": int}`` or all-zero for
    empty scores.
    """
    pitches = [n.pitch for t in score.tracks for n in t.notes]
    if not pitches:
        return {"min": 0, "max": 0, "range": 0}
    lo, hi = min(pitches), max(pitches)
    return {"min": lo, "max": hi, "range": hi - lo}


# ── Aggregate helper ────────────────────────────────────────────────────────


def sequence_diagnostics(
    token_ids: Sequence[int],
    score: Score | None = None,
) -> dict:
    """Compute all available diagnostics for a single generated sequence.

    Parameters
    ----------
    token_ids:
        Raw token IDs of the generated continuation.
    score:
        Decoded ``symusic.Score`` (optional).  When provided, note-level
        metrics are included.

    Returns
    -------
    dict with keys: ``rep_2gram``, ``rep_3gram``, ``rep_4gram``,
    ``token_entropy``, and optionally ``note_density``, ``pitch_min``,
    ``pitch_max``, ``pitch_range``.
    """
    diag: dict = {
        "rep_2gram": ngram_repetition_rate(token_ids, 2),
        "rep_3gram": ngram_repetition_rate(token_ids, 3),
        "rep_4gram": ngram_repetition_rate(token_ids, 4),
        "token_entropy": token_entropy(token_ids),
    }
    if score is not None:
        pr = pitch_range(score)
        diag["note_density"] = note_density(score)
        diag["pitch_min"] = pr["min"]
        diag["pitch_max"] = pr["max"]
        diag["pitch_range"] = pr["range"]
    return diag
