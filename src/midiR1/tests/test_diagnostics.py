import math
import pytest
from symusic import Score, Track, Note

from src.midiR1.utils.diagnostics import (
    ngram_repetition_rate,
    note_density,
    pitch_range,
    sequence_diagnostics,
    token_entropy,
)


# ── ngram_repetition_rate ───────────────────────────────────────────────────


def test_no_repetition():
    """All unique bigrams → rate 0."""
    ids = [1, 2, 3, 4, 5]
    assert ngram_repetition_rate(ids, n=2) == 0.0


def test_all_repeated():
    """Constant sequence → every n-gram is the same, rate = 1 - 1/total."""
    ids = [7, 7, 7, 7, 7]
    # 4 bigrams, all identical → 1 unique → rate = 1 - 1/4 = 0.75
    assert ngram_repetition_rate(ids, n=2) == pytest.approx(0.75)


def test_too_short_for_ngram():
    """Sequence shorter than n → 0."""
    assert ngram_repetition_rate([1, 2], n=3) == 0.0


def test_empty_sequence():
    assert ngram_repetition_rate([], n=2) == 0.0


def test_trigram_repetition():
    ids = [1, 2, 3, 1, 2, 3]
    # trigrams: (1,2,3), (2,3,1), (3,1,2), (1,2,3) → 4 total, 3 unique → rate = 0.25
    assert ngram_repetition_rate(ids, n=3) == pytest.approx(0.25)


# ── token_entropy ───────────────────────────────────────────────────────────


def test_entropy_uniform():
    """Uniform distribution of 4 tokens → 2 bits."""
    ids = [0, 1, 2, 3]
    assert token_entropy(ids) == pytest.approx(2.0)


def test_entropy_constant():
    """All same token → 0 bits."""
    ids = [5, 5, 5, 5]
    assert token_entropy(ids) == pytest.approx(0.0)


def test_entropy_empty():
    assert token_entropy([]) == 0.0


def test_entropy_two_tokens():
    """50/50 split → 1 bit."""
    ids = [0, 1, 0, 1]
    assert token_entropy(ids) == pytest.approx(1.0)


# ── note_density ────────────────────────────────────────────────────────────


def _make_score(notes, tpq=480):
    """Helper: build a Score with one track containing the given notes."""
    score = Score(tpq)
    track = Track()
    for pitch, start, dur in notes:
        track.notes.append(Note(start, dur, pitch, 64))
    score.tracks.append(track)
    return score


def test_density_basic():
    # 4 notes over 2 beats (480 ticks * 2 = 960 ticks end)
    score = _make_score([
        (60, 0, 240),
        (62, 240, 240),
        (64, 480, 240),
        (65, 720, 240),
    ], tpq=480)
    # end tick = 720 + 240 = 960 → 2 beats → 4/2 = 2.0
    assert note_density(score) == pytest.approx(2.0)


def test_density_empty():
    score = Score(480)
    assert note_density(score) == 0.0


# ── pitch_range ─────────────────────────────────────────────────────────────


def test_pitch_range_basic():
    score = _make_score([(40, 0, 100), (72, 100, 100), (55, 200, 100)])
    pr = pitch_range(score)
    assert pr == {"min": 40, "max": 72, "range": 32}


def test_pitch_range_empty():
    score = Score(480)
    pr = pitch_range(score)
    assert pr == {"min": 0, "max": 0, "range": 0}


def test_pitch_range_single_note():
    score = _make_score([(60, 0, 100)])
    pr = pitch_range(score)
    assert pr == {"min": 60, "max": 60, "range": 0}


# ── sequence_diagnostics ───────────────────────────────────────────────────


def test_diagnostics_token_only():
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    diag = sequence_diagnostics(ids)
    assert "rep_2gram" in diag
    assert "rep_3gram" in diag
    assert "rep_4gram" in diag
    assert "token_entropy" in diag
    # No score → no note-level metrics
    assert "note_density" not in diag


def test_diagnostics_with_score():
    ids = [1, 2, 3, 4]
    score = _make_score([(60, 0, 100), (72, 100, 100)])
    diag = sequence_diagnostics(ids, score=score)
    assert "note_density" in diag
    assert "pitch_min" in diag
    assert diag["pitch_min"] == 60
    assert diag["pitch_max"] == 72
    assert diag["pitch_range"] == 12
