import json
import pytest
import numpy as np
import torch

from src.midiR1.data.dataset import PieceCropDataset


@pytest.fixture
def cache_dir(tmp_path):
    """Create a synthetic token cache with 3 pieces of different lengths."""
    pieces = {
        "short.npy": np.array([1, 2, 3], dtype=np.int32),       # 3 tokens
        "medium.npy": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.int32),  # 10
        "long.npy": np.arange(1, 201, dtype=np.int32),           # 200 tokens
    }
    manifest = []
    for name, arr in pieces.items():
        np.save(tmp_path / name, arr)
        manifest.append({"file": name, "length": len(arr)})

    with open(tmp_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    return tmp_path


def test_dataset_loads_pieces(cache_dir):
    """Dataset loads all pieces from cache directory."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1)
    assert len(ds.pieces) == 3


def test_short_pieces_filtered(cache_dir):
    """Pieces below min_seq_len are excluded."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=5)
    # "short.npy" has 3 tokens, should be filtered out
    assert len(ds.pieces) == 2


def test_epoch_len(cache_dir):
    """__len__ returns total_tokens // max_seq_len."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1)
    total_tokens = 3 + 10 + 200  # 213
    assert len(ds) == total_tokens // 16


def test_getitem_returns_correct_format(cache_dir):
    """Each sample is a dict with 'input_ids' as a LongTensor."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1)
    sample = ds[0]
    assert "input_ids" in sample
    assert sample["input_ids"].dtype == torch.long
    assert sample["input_ids"].ndim == 1


def test_long_piece_cropped(cache_dir):
    """Pieces longer than max_seq_len are cropped to exactly max_seq_len."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1)
    # Sample many times; any sample drawn from the long piece must have length 16
    for _ in range(50):
        sample = ds[0]
        assert len(sample["input_ids"]) <= 16


def test_short_piece_returned_full(tmp_path):
    """Pieces shorter than max_seq_len are returned at their full length."""
    arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    np.save(tmp_path / "only.npy", arr)
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump([{"file": "only.npy", "length": 5}], f)

    ds = PieceCropDataset(tmp_path, max_seq_len=16, min_seq_len=1)
    sample = ds[0]
    assert len(sample["input_ids"]) == 5
    assert list(sample["input_ids"].numpy()) == [1, 2, 3, 4, 5]


def test_uniform_weighting(cache_dir):
    """length_exponent=0.0 gives equal probability to all pieces."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1, length_exponent=0.0)
    np.testing.assert_allclose(ds.weights, [1/3, 1/3, 1/3])


def test_length_proportional_weighting(cache_dir):
    """length_exponent=1.0 weights proportionally to piece length."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1, length_exponent=1.0)
    total = 3 + 10 + 200
    expected = np.array([3 / total, 10 / total, 200 / total])
    np.testing.assert_allclose(ds.weights, expected)


def test_sqrt_weighting(cache_dir):
    """length_exponent=0.5 gives sqrt-proportional weights."""
    ds = PieceCropDataset(cache_dir, max_seq_len=16, min_seq_len=1, length_exponent=0.5)
    raw = np.array([3, 10, 200], dtype=np.float64) ** 0.5
    expected = raw / raw.sum()
    np.testing.assert_allclose(ds.weights, expected)


def test_missing_manifest_raises(tmp_path):
    """FileNotFoundError raised when manifest.json is missing."""
    with pytest.raises(FileNotFoundError, match="manifest.json"):
        PieceCropDataset(tmp_path, max_seq_len=16)


def test_no_valid_pieces_raises(tmp_path):
    """RuntimeError raised when all pieces are below min_seq_len."""
    arr = np.array([1, 2], dtype=np.int32)
    np.save(tmp_path / "tiny.npy", arr)
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump([{"file": "tiny.npy", "length": 2}], f)

    with pytest.raises(RuntimeError, match="No pieces"):
        PieceCropDataset(tmp_path, max_seq_len=16, min_seq_len=5)


def test_crop_values_are_contiguous(tmp_path):
    """Cropped tokens are a contiguous sub-sequence of the original piece."""
    arr = np.arange(100, dtype=np.int32)
    np.save(tmp_path / "seq.npy", arr)
    with open(tmp_path / "manifest.json", "w") as f:
        json.dump([{"file": "seq.npy", "length": 100}], f)

    ds = PieceCropDataset(tmp_path, max_seq_len=10, min_seq_len=1)
    for _ in range(20):
        crop = ds[0]["input_ids"].numpy()
        assert len(crop) == 10
        # Values should be consecutive integers (since original is arange)
        assert np.all(np.diff(crop) == 1)
