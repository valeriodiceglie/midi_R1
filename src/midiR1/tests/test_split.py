import pytest
from symusic import Score, Track, Note

from src.midiR1.data.split import split_directory


def _make_midi(path, pitch=60):
    """Write a minimal MIDI file."""
    score = Score(480)
    track = Track()
    track.notes.append(Note(0, 480, pitch, 64))
    score.tracks.append(track)
    score.dump_midi(str(path))


def test_basic_split(tmp_path):
    src = tmp_path / "midi"
    src.mkdir()
    for i in range(10):
        _make_midi(src / f"file_{i:02d}.mid")

    dst = tmp_path / "split"
    counts = split_directory(src, dst)

    assert counts["train"] == 8
    assert counts["val"] == 1
    assert counts["test"] == 1
    assert len(list((dst / "train").glob("*.mid"))) == 8
    assert len(list((dst / "val").glob("*.mid"))) == 1
    assert len(list((dst / "test").glob("*.mid"))) == 1


def test_reproducible(tmp_path):
    src = tmp_path / "midi"
    src.mkdir()
    for i in range(10):
        _make_midi(src / f"f{i}.mid")

    dst1 = tmp_path / "s1"
    dst2 = tmp_path / "s2"
    split_directory(src, dst1, seed=123)
    split_directory(src, dst2, seed=123)

    files1 = sorted(f.name for f in (dst1 / "train").glob("*.mid"))
    files2 = sorted(f.name for f in (dst2 / "train").glob("*.mid"))
    assert files1 == files2


def test_very_few_files(tmp_path):
    """With only 2 files, train gets both and val gets a duplicate."""
    src = tmp_path / "midi"
    src.mkdir()
    _make_midi(src / "a.mid")
    _make_midi(src / "b.mid")

    dst = tmp_path / "split"
    counts = split_directory(src, dst)

    assert counts["train"] == 2
    assert counts["val"] == 1  # duplicated from train
    assert len(list((dst / "val").glob("*.mid"))) == 1


def test_no_midi_raises(tmp_path):
    src = tmp_path / "empty"
    src.mkdir()
    with pytest.raises(FileNotFoundError):
        split_directory(src, tmp_path / "out")


def test_custom_ratios(tmp_path):
    src = tmp_path / "midi"
    src.mkdir()
    for i in range(20):
        _make_midi(src / f"f{i:02d}.mid")

    dst = tmp_path / "split"
    counts = split_directory(src, dst, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    assert counts["train"] == 14
    assert counts["val"] == 3
    assert counts["test"] == 3
