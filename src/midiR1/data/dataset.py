"""Piece-level sampling dataset with random cropping."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PieceCropDataset(Dataset):
    """Sample a piece, then take a random crop of ``max_seq_len`` tokens.

    Parameters
    ----------
    cache_dir:
        Directory containing ``.npy`` token files and a ``manifest.json``.
    max_seq_len:
        Maximum (and target) crop length.
    min_seq_len:
        Pieces shorter than this are skipped.
    pad_token_id:
        Not used internally (short pieces are returned as-is; the collator
        handles cross-batch padding), but stored for reference.
    length_exponent:
        Controls how piece length influences sampling probability.
        ``prob_i ∝ len_i ** length_exponent``.
        * 0.0 — uniform over pieces (default, no length bias).
        * 0.5 — sqrt-length weighting (compromise).
        * 1.0 — length-proportional (recovers old chunk-uniform bias).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        max_seq_len: int,
        min_seq_len: int = 5,
        pad_token_id: int = 0,
        length_exponent: float = 0.0,
    ):
        cache_dir = Path(cache_dir)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {cache_dir}. "
                "Run `python cli.py cache-tokens` first."
            )

        with open(manifest_path) as f:
            manifest: list[dict] = json.load(f)

        # Filter and load pieces into memory
        self.pieces: list[np.ndarray] = []
        lengths: list[int] = []
        for entry in manifest:
            if entry["length"] < min_seq_len:
                continue
            arr = np.load(cache_dir / entry["file"])
            self.pieces.append(arr)
            lengths.append(len(arr))

        if not self.pieces:
            raise RuntimeError(f"No pieces with length >= {min_seq_len} in {cache_dir}")

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Sampling weights
        lengths_arr = np.array(lengths, dtype=np.float64)
        weights = lengths_arr ** length_exponent
        self.weights = weights / weights.sum()

        # Epoch size: total tokens / max_seq_len
        self._epoch_len = max(1, int(lengths_arr.sum()) // max_seq_len)

    def __len__(self) -> int:
        return self._epoch_len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Sample a piece (idx is intentionally unused)
        piece_idx = np.random.choice(len(self.pieces), p=self.weights)
        tokens = self.pieces[piece_idx]

        if len(tokens) <= self.max_seq_len:
            crop = tokens
        else:
            start = np.random.randint(0, len(tokens) - self.max_seq_len + 1)
            crop = tokens[start : start + self.max_seq_len]

        return {"input_ids": torch.from_numpy(crop.copy()).long()}
