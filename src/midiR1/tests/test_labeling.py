import pytest
from symusic import Score, Track, Note

from src.midiR1.utils.labeling import classify_section, label_directory


# ── helpers ───────────────────────────────────────────────────────────────


def _make_score(notes, tpq=480):
    """Build a Score with one track containing the given notes."""
    score = Score(tpq)
    track = Track()
    for pitch, start, dur in notes:
        track.notes.append(Note(start, dur, pitch, 64))
    score.tracks.append(track)
    return score


# ── classify_section ─────────────────────────────────────────────────────


def test_monophonic_is_solo():
    """Sequential notes with no overlap → solo."""
    score = _make_score([
        (60, 0, 480),
        (62, 480, 480),
        (64, 960, 480),
        (65, 1440, 480),
    ])
    assert classify_section(score) == "solo"


def test_chordal_is_rhythm():
    """All notes start at the same time → rhythm."""
    score = _make_score([
        (60, 0, 480),
        (64, 0, 480),
        (67, 0, 480),
        (60, 480, 480),
        (64, 480, 480),
        (67, 480, 480),
    ])
    assert classify_section(score) == "rhythm"


def test_empty_score_is_solo():
    score = Score(480)
    assert classify_section(score) == "solo"


def test_single_note_is_solo():
    score = _make_score([(60, 0, 480)])
    assert classify_section(score) == "solo"


def test_threshold_boundary():
    """With 4 beats: 1 polyphonic → ratio=0.25. threshold=0.2 → rhythm, 0.3 → solo."""
    score = _make_score([
        (60, 0, 480),   # beat 0 — 2 notes (polyphonic)
        (64, 0, 480),
        (62, 480, 480),  # beat 1 — 1 note
        (65, 960, 480),  # beat 2 — 1 note
        (67, 1440, 480), # beat 3 — 1 note
    ])
    # polyphony_ratio = 1/4 = 0.25
    assert classify_section(score, threshold=0.2) == "rhythm"
    assert classify_section(score, threshold=0.3) == "solo"


def test_mixed_polyphony():
    """Half polyphonic beats → rhythm at default threshold."""
    score = _make_score([
        (60, 0, 480),    # beat 0 — 2 notes
        (64, 0, 480),
        (62, 480, 480),  # beat 1 — 2 notes
        (65, 480, 480),
        (67, 960, 480),  # beat 2 — 1 note
        (69, 1440, 480), # beat 3 — 1 note
    ])
    # polyphony_ratio = 2/4 = 0.5 > 0.3
    assert classify_section(score) == "rhythm"


# ── label_directory ──────────────────────────────────────────────────────


def test_label_directory(tmp_path):
    """label_directory classifies every .mid file in a folder."""
    solo_score = _make_score([(60, 0, 480), (62, 480, 480)])
    chord_score = _make_score([
        (60, 0, 480), (64, 0, 480), (67, 0, 480),
        (60, 480, 480), (64, 480, 480), (67, 480, 480),
    ])

    solo_score.dump_midi(str(tmp_path / "solo.mid"))
    chord_score.dump_midi(str(tmp_path / "chord.mid"))

    labels = label_directory(tmp_path)
    assert labels["solo.mid"] == "solo"
    assert labels["chord.mid"] == "rhythm"


def test_label_directory_empty(tmp_path):
    labels = label_directory(tmp_path)
    assert labels == {}
