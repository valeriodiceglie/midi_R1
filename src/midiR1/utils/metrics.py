"""MIDI-level evaluation metrics: KL divergence, overlapping area, Frechet Music Distance.

Compares distributions of musical features between reference and generated MIDI sets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from symusic import Score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_VEL_BINS = np.linspace(0, 128, 9)       # 8 velocity bins
_DUR_BINS_TICKS = np.array([0, 60, 120, 240, 480, 960, 1920, 3840, 7680])  # 8 duration bins


def extract_midi_features(score: Score) -> np.ndarray:
    """Extract a 40-dimensional feature vector from a MIDI score.

    Features (in order):
        0-11  : Pitch class histogram (12, normalised)
        12-14 : Pitch min, max, mean (3)
        15    : Note density (notes per beat) (1)
        16-17 : Mean velocity, std velocity (2)
        18-25 : Velocity histogram (8, normalised)
        26-27 : Mean duration, std duration (2)
        28-35 : Duration histogram (8, normalised)
        36-37 : Mean IOI, IOI std (2)
        38    : Average polyphony (1)
        39    : Polyphony rate (1)
    """
    pitches = []
    velocities = []
    durations = []
    onsets = []

    for track in score.tracks:
        for note in track.notes:
            pitches.append(note.pitch)
            velocities.append(note.velocity)
            durations.append(note.duration)
            onsets.append(note.time)

    feats = np.zeros(40, dtype=np.float64)

    if not pitches:
        return feats

    pitches = np.array(pitches, dtype=np.float64)
    velocities = np.array(velocities, dtype=np.float64)
    durations = np.array(durations, dtype=np.float64)
    onsets = np.array(onsets, dtype=np.float64)

    # Pitch class histogram (12)
    pc = (pitches.astype(int) % 12).astype(int)
    pc_hist = np.bincount(pc, minlength=12).astype(np.float64)
    pc_sum = pc_hist.sum()
    if pc_sum > 0:
        pc_hist /= pc_sum
    feats[0:12] = pc_hist

    # Pitch stats (3)
    feats[12] = pitches.min()
    feats[13] = pitches.max()
    feats[14] = pitches.mean()

    # Note density (1) — notes per beat
    tpq = score.ticks_per_quarter if score.ticks_per_quarter > 0 else 480
    total_ticks = max(onsets.max() - onsets.min(), 1.0)
    total_beats = total_ticks / tpq
    feats[15] = len(pitches) / max(total_beats, 1.0)

    # Velocity stats (2)
    feats[16] = velocities.mean()
    feats[17] = velocities.std() if len(velocities) > 1 else 0.0

    # Velocity histogram (8)
    vel_hist, _ = np.histogram(velocities, bins=_VEL_BINS)
    vel_hist = vel_hist.astype(np.float64)
    vel_sum = vel_hist.sum()
    if vel_sum > 0:
        vel_hist /= vel_sum
    feats[18:26] = vel_hist

    # Duration stats (2)
    feats[26] = durations.mean()
    feats[27] = durations.std() if len(durations) > 1 else 0.0

    # Duration histogram (8)
    dur_hist, _ = np.histogram(durations, bins=_DUR_BINS_TICKS)
    dur_hist = dur_hist.astype(np.float64)
    dur_sum = dur_hist.sum()
    if dur_sum > 0:
        dur_hist /= dur_sum
    feats[28:36] = dur_hist

    # IOI stats (2)
    sorted_onsets = np.sort(np.unique(onsets))
    if len(sorted_onsets) > 1:
        ioi = np.diff(sorted_onsets)
        feats[36] = ioi.mean()
        feats[37] = ioi.std()

    # Polyphony (2) — computed from onset coincidence
    onset_counts = np.bincount(onsets.astype(int))
    active_steps = onset_counts[onset_counts > 0]
    if len(active_steps) > 0:
        feats[38] = active_steps.mean()                                  # avg polyphony
        feats[39] = (active_steps > 1).sum() / len(active_steps)         # polyphony rate

    return feats


def extract_features_from_dir(midi_dir: Path) -> np.ndarray:
    """Extract feature matrix (N x 40) from all .mid files in a directory."""
    paths = sorted(midi_dir.glob("*.mid"))
    if not paths:
        raise FileNotFoundError(f"No .mid files found in {midi_dir}")

    features = []
    for p in paths:
        try:
            score = Score(str(p))
            feats = extract_midi_features(score)
            features.append(feats)
        except Exception as e:
            logger.warning("Skipping %s: %s", p.name, e)

    if not features:
        raise RuntimeError(f"Could not extract features from any file in {midi_dir}")

    return np.stack(features)


# ---------------------------------------------------------------------------
# Distribution metrics
# ---------------------------------------------------------------------------

_EPS = 1e-10


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(P || Q) for discrete distributions *p* and *q*.

    Both arrays must be non-negative and will be normalised internally.
    Uses epsilon-smoothing to avoid log(0).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0

    p = p / p_sum + _EPS
    q = q / q_sum + _EPS
    p /= p.sum()
    q /= q.sum()

    return float(np.sum(p * np.log(p / q)))


def overlapping_area(p: np.ndarray, q: np.ndarray) -> float:
    """Overlap coefficient OA = sum(min(P_i, Q_i)) for normalised distributions.

    Returns a value in [0, 1]:  1 = identical, 0 = disjoint.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()

    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return 0.0

    p = p / p_sum
    q = q / q_sum

    return float(np.minimum(p, q).sum())


def frechet_music_distance(feats_ref: np.ndarray, feats_gen: np.ndarray) -> float:
    """Compute Frechet Music Distance between two (N x D) feature matrices.

    FMD = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 * sqrtm(S_r @ S_g))
    """
    from scipy import linalg

    mu_r = np.mean(feats_ref, axis=0)
    mu_g = np.mean(feats_gen, axis=0)
    sigma_r = np.cov(feats_ref, rowvar=False)
    sigma_g = np.cov(feats_gen, rowvar=False)

    # Ensure 2-D covariance (single-feature edge case)
    if sigma_r.ndim == 0:
        sigma_r = np.array([[sigma_r]])
        sigma_g = np.array([[sigma_g]])

    diff = mu_r - mu_g
    mean_term = diff.dot(diff)

    eps = 1e-6
    sqrt_product, _ = linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)

    if not np.isfinite(sqrt_product).all():
        offset = np.eye(sigma_r.shape[0]) * eps
        sqrt_product = linalg.sqrtm((sigma_r + offset).dot(sigma_g + offset))

    # Discard small imaginary components from numerical error
    sqrt_product = sqrt_product.real

    trace_term = np.trace(sigma_r) + np.trace(sigma_g) - 2.0 * np.trace(sqrt_product)

    return float(mean_term + trace_term)


# ---------------------------------------------------------------------------
# Per-feature-type KL and OA (pitch / duration / velocity histograms)
# ---------------------------------------------------------------------------

def _aggregate_histogram(features: np.ndarray, start: int, end: int) -> np.ndarray:
    """Average a histogram slice across all samples."""
    return features[:, start:end].mean(axis=0)


def compute_distribution_metrics(
    feats_ref: np.ndarray, feats_gen: np.ndarray,
) -> dict[str, float]:
    """Compute per-feature-type KL divergence and overlapping area.

    Returns a dict with keys like ``pitch_kl``, ``pitch_oa``, etc.
    """
    slices = {
        "pitch":    (0, 12),
        "velocity": (18, 26),
        "duration": (28, 36),
    }

    results: dict[str, float] = {}
    kl_values = []
    oa_values = []

    for name, (s, e) in slices.items():
        hist_ref = _aggregate_histogram(feats_ref, s, e)
        hist_gen = _aggregate_histogram(feats_gen, s, e)
        kl_val = kl_divergence(hist_ref, hist_gen)
        oa_val = overlapping_area(hist_ref, hist_gen)
        results[f"{name}_kl"] = kl_val
        results[f"{name}_oa"] = oa_val
        kl_values.append(kl_val)
        oa_values.append(oa_val)

    results["mean_kl"] = float(np.mean(kl_values))
    results["mean_oa"] = float(np.mean(oa_values))

    return results


# ---------------------------------------------------------------------------
# Top-level evaluation entry point
# ---------------------------------------------------------------------------

def compute_generation_metrics(
    reference_dir: Path, generated_dir: Path,
) -> dict[str, float]:
    """Compute all MIDI-level generation metrics.

    Returns a dict with keys:
        ``pitch_kl``, ``pitch_oa``,
        ``velocity_kl``, ``velocity_oa``,
        ``duration_kl``, ``duration_oa``,
        ``mean_kl``, ``mean_oa``,
        ``fmd``
    """
    feats_ref = extract_features_from_dir(reference_dir)
    feats_gen = extract_features_from_dir(generated_dir)

    metrics = compute_distribution_metrics(feats_ref, feats_gen)
    metrics["fmd"] = frechet_music_distance(feats_ref, feats_gen)

    return metrics
