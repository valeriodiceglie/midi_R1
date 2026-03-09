"""Tokenize full-length MIDI files and cache token IDs as .npy arrays."""

import json
import logging
import random
from pathlib import Path

import numpy as np
from miditok import REMI
from symusic import Score
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def _random_ac_indexes(
    score: Score,
    tokenizer: REMI,
    tracks_ratio_range: tuple[float, float] = (0.4, 0.9),
    bars_ratio_range: tuple[float, float] = (0.4, 0.9),
) -> dict | None:
    """Build random ``attribute_controls_indexes`` for a score.

    Uses MIDITok's ``create_random_ac_indexes`` when available, otherwise
    falls back to a simple manual implementation.
    """
    if not tokenizer.attribute_controls:
        return None

    try:
        from miditok.attribute_controls import create_random_ac_indexes
        return create_random_ac_indexes(
            score, tokenizer.attribute_controls,
            tracks_ratio_range, bars_ratio_range,
        )
    except ImportError:
        pass

    # Manual fallback: enable all ACs for all tracks/bars
    from miditok.attribute_controls import BarAttributeControl
    ac_indexes: dict = {}
    for ti in range(len(score.tracks)):
        track_acs: dict = {}
        for ai, ac in enumerate(tokenizer.attribute_controls):
            if isinstance(ac, BarAttributeControl):
                # Select a random subset of bars
                tpq = score.ticks_per_quarter or 480
                n_bars = max(int(score.end() / (tpq * 4)), 1)
                ratio = random.uniform(*bars_ratio_range)
                n_select = max(1, int(n_bars * ratio))
                bar_idxs = sorted(random.sample(range(n_bars), min(n_select, n_bars)))
                track_acs[ai] = bar_idxs
            else:
                # Track-level: include with a probability within the ratio range
                if random.random() < random.uniform(*tracks_ratio_range):
                    track_acs[ai] = True
        ac_indexes[ti] = track_acs
    return ac_indexes


def build_token_cache(
    midi_dir: Path,
    cache_dir: Path,
    tokenizer: REMI,
    bos_token_id: int | None,
    eos_token_id: int | None,
    min_seq_len: int = 5,
    conditioning_fn=None,
    use_attribute_controls: bool = False,
    ac_tracks_ratio_range: tuple[float, float] = (0.4, 0.9),
    ac_bars_ratio_range: tuple[float, float] = (0.4, 0.9),
) -> list[dict]:
    """Tokenize every MIDI in *midi_dir* and write .npy + manifest to *cache_dir*.

    Parameters
    ----------
    conditioning_fn:
        Optional callable ``(Path) -> (int, str) | None``.  When provided, the
        returned token ID is inserted immediately after BOS (or at position 0
        if there is no BOS).  The label string is stored in the manifest.
        Ignored when *use_attribute_controls* is True.
    use_attribute_controls:
        When True, use MIDITok attribute controls for conditioning instead of
        the custom ``conditioning_fn``.  Requires that the tokenizer was
        configured with attribute controls enabled.
    ac_tracks_ratio_range:
        Range ``(min, max)`` for the fraction of track-level ACs to apply
        per sample during training (random partial application).
    ac_bars_ratio_range:
        Range ``(min, max)`` for the fraction of bars to label with bar-level
        ACs per sample during training (random partial application).

    Returns the manifest (list of {"file": str, "length": int} dicts).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    midi_files = sorted(midi_dir.glob("*.mid"))
    if not midi_files:
        raise FileNotFoundError(f"No .mid files found in {midi_dir}")

    manifest: list[dict] = []
    dropped = 0
    total_tokens = 0

    for path in tqdm(midi_files, desc=f"Caching {midi_dir.name}"):
        try:
            if use_attribute_controls and tokenizer.attribute_controls:
                score = Score(str(path))
                score = tokenizer.preprocess_score(score)
                ac_indexes = _random_ac_indexes(
                    score, tokenizer,
                    ac_tracks_ratio_range, ac_bars_ratio_range,
                )
                tok_result = tokenizer.encode(
                    score,
                    no_preprocess_score=True,
                    attribute_controls_indexes=ac_indexes,
                )
            else:
                tok_result = tokenizer(path)
        except Exception as e:
            logger.warning("Skipping %s: %s", path.name, e)
            dropped += 1
            continue

        ids = tok_result[0].ids

        # Prepend BOS / append EOS
        if bos_token_id is not None:
            ids = [bos_token_id] + ids

        # Insert conditioning token after BOS (or at position 0)
        # (legacy path — skipped when using attribute controls)
        cond_label = None
        if conditioning_fn is not None and not use_attribute_controls:
            cond_result = conditioning_fn(path)
            if cond_result is not None:
                cond_id, cond_label = cond_result
                if bos_token_id is not None:
                    ids = [ids[0], cond_id] + ids[1:]
                else:
                    ids = [cond_id] + ids

        if eos_token_id is not None:
            ids = ids + [eos_token_id]

        if len(ids) < min_seq_len:
            dropped += 1
            continue

        arr = np.array(ids, dtype=np.int32)
        npy_name = f"{path.stem}.npy"
        np.save(cache_dir / npy_name, arr)
        entry = {"file": npy_name, "length": len(ids)}
        if cond_label is not None:
            entry["label"] = cond_label
        if use_attribute_controls:
            entry["ac"] = True
        manifest.append(entry)
        total_tokens += len(ids)

    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f)

    logger.info(
        "Cached %d pieces (%d tokens), dropped %d (dir: %s)",
        len(manifest), total_tokens, dropped, cache_dir,
    )
    return manifest
