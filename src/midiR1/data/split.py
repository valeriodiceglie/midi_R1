"""Split a flat directory of MIDI files into train / val / test sub-directories."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def split_directory(
    src_dir: Path,
    dst_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    copy: bool = True,
) -> dict[str, int]:
    """Split ``*.mid`` files from *src_dir* into train/val/test under *dst_dir*.

    Parameters
    ----------
    src_dir:
        Directory containing ``.mid`` files (not recursive).
    dst_dir:
        Parent directory where ``train/``, ``val/``, ``test/`` will be created.
    train_ratio, val_ratio, test_ratio:
        Proportions (must sum to ~1.0).  With very few files the actual
        counts are rounded so every split gets at least one file when
        possible.
    seed:
        Random seed for reproducible splits.
    copy:
        If *True* (default) copies files; if *False* moves them.

    Returns
    -------
    ``{"train": N, "val": N, "test": N}`` counts.
    """
    midi_files = sorted(src_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid files in {src_dir}")

    rng = random.Random(seed)
    rng.shuffle(midi_files)

    n = len(midi_files)
    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * test_ratio))
    n_train = n - n_val - n_test

    if n_train < 1:
        # If there are very few files, give everything to train and
        # duplicate one file into val (so early stopping can function).
        n_train = n
        n_val = 0
        n_test = 0

    splits = {
        "train": midi_files[:n_train],
        "val": midi_files[n_train:n_train + n_val],
        "test": midi_files[n_train + n_val:n_train + n_val + n_test],
    }

    transfer = shutil.copy2 if copy else shutil.move
    counts: dict[str, int] = {}

    for split_name, files in splits.items():
        split_dir = dst_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            transfer(str(f), str(split_dir / f.name))
        counts[split_name] = len(files)

    # If val ended up empty (very few files), duplicate one train file
    if counts["val"] == 0 and counts["train"] > 0:
        val_dir = dst_dir / "val"
        val_dir.mkdir(parents=True, exist_ok=True)
        first_train = splits["train"][0]
        shutil.copy2(str(first_train), str(val_dir / first_train.name))
        counts["val"] = 1
        logger.warning(
            "Only %d file(s) available — duplicated one into val/ for early stopping.", n,
        )

    logger.info(
        "Split %d files → train=%d, val=%d, test=%d (dst=%s)",
        n, counts["train"], counts["val"], counts["test"], dst_dir,
    )
    return counts
