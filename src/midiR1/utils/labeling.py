"""Solo vs rhythm classification for guitar MIDI files.

Uses a polyphony-ratio heuristic: if a large fraction of active beats have
two or more simultaneous note onsets the piece is classified as *rhythm*
(chord / strumming), otherwise as *solo* (single-note lines).
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

from symusic import Score

logger = logging.getLogger(__name__)


def classify_section(score: Score, threshold: float = 0.3) -> str:
    """Classify a MIDI score as ``"solo"`` or ``"rhythm"``.

    Parameters
    ----------
    score:
        A ``symusic.Score`` (typically a single-track guitar excerpt).
    threshold:
        Fraction of active beats that must contain 2+ simultaneous note
        onsets to be labelled *rhythm*.  Lower values bias towards
        ``"rhythm"``, higher values towards ``"solo"``.

    Returns
    -------
    ``"solo"`` or ``"rhythm"``
    """
    tpq = score.ticks_per_quarter
    if tpq == 0:
        return "solo"

    # Collect all note onsets, quantised to beat boundaries
    beat_onsets: Counter[int] = Counter()
    for track in score.tracks:
        for note in track.notes:
            beat = note.start // tpq
            beat_onsets[beat] += 1

    if not beat_onsets:
        return "solo"

    polyphonic_beats = sum(1 for count in beat_onsets.values() if count >= 2)
    polyphony_ratio = polyphonic_beats / len(beat_onsets)

    return "rhythm" if polyphony_ratio > threshold else "solo"


def label_directory(midi_dir: Path, threshold: float = 0.3) -> dict[str, str]:
    """Classify every ``.mid`` file in *midi_dir*.

    Returns a mapping ``{"filename.mid": "solo" | "rhythm", ...}``.
    Files that fail to parse are skipped with a warning.
    """
    labels: dict[str, str] = {}
    for path in sorted(midi_dir.glob("*.mid")):
        try:
            score = Score(str(path))
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            continue
        labels[path.name] = classify_section(score, threshold=threshold)
    return labels
