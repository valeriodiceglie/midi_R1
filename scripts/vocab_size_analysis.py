"""
Vocab Size Analysis for BPE Tokenizer
======================================

Trains REMI+BPE tokenizers at multiple vocab sizes and produces formal
metrics to guide the choice of vocabulary size.

Metrics computed for each candidate vocab size:
  1. Base vocab size vs. number of BPE merges
  2. Compression ratio  (base tokens / BPE tokens)
  3. Token frequency distribution  (how many tokens are underused)
  4. Average sequence length per file  (context efficiency)
  5. Fertility  (BPE tokens per base token, on average)
  6. Embedding parameter budget relative to model size

Outputs:
  - Console summary table
  - plots/vocab_analysis/  with publication-ready figures

Usage:
    python scripts/vocab_size_analysis.py [--midi-dir PATH] [--sample-size N]
"""

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm.auto import tqdm

from miditok import REMI, TokenizerConfig, constants


# ──────────────────────────────────────────────────────────────────────
# Tokenizer configuration (mirrors train_tokenizer.py)
# ──────────────────────────────────────────────────────────────────────

def make_tokenizer_config() -> TokenizerConfig:
    guitar_ranges = [
        inst["pitch_range"]
        for inst in constants.MIDI_INSTRUMENTS
        if "Guitar" in inst["name"]
    ]
    pitch_min = min(r.start for r in guitar_ranges)
    pitch_max = max(r.start for r in guitar_ranges)

    return TokenizerConfig(
        beat_res={(0, 4): 12, (4, 12): 4},
        beat_res_rest={(0, 1): 12, (1, 2): 4, (2, 12): 2},
        encode_ids_split="bar",
        pitch_range=(pitch_min, pitch_max),
        num_velocities=32,
        use_rests=True,
        use_chords=True,
        use_tempos=True,
        use_time_signatures=True,
        use_pitch_bends=True,
        time_signature_range={4: [1, 2, 3, 4, 5, 6, 7, 9], 8: [3, 6, 12]},
    )


# ──────────────────────────────────────────────────────────────────────
# Step 1: Measure the base vocabulary (before BPE)
# ──────────────────────────────────────────────────────────────────────

def count_base_vocab(config: TokenizerConfig) -> int:
    """Create a REMI tokenizer without BPE to count base tokens."""
    tok = REMI(config)
    return tok.vocab_size


# ──────────────────────────────────────────────────────────────────────
# Step 2: Train tokenizers at each candidate size
# ──────────────────────────────────────────────────────────────────────

def train_tokenizer(config: TokenizerConfig,
                    train_files: list[Path],
                    vocab_size: int,
                    save_dir: Path | None = None) -> REMI:
    """Train a REMI+BPE tokenizer at the given vocab size and save it."""
    tok = REMI(config)
    tok.train(files_paths=train_files, vocab_size=vocab_size, model="BPE")
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"tokenizer_{vocab_size}REMI.json"
        tok.save(save_path)
        print(f"  Saved tokenizer to {save_path}")
    return tok


# ──────────────────────────────────────────────────────────────────────
# Step 3: Compute per-tokenizer metrics over a sample
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(tokenizer: REMI,
                    sample_files: list[Path],
                    base_vocab_size: int) -> dict:
    """Tokenize sample files and compute all metrics."""
    all_ids: list[list[int]] = []
    token_counter = Counter()
    total_tokens = 0
    failed = 0

    for path in tqdm(sample_files, desc=f"  Tokenizing (V={tokenizer.vocab_size})",
                     leave=False):
        try:
            result = tokenizer(path)
            ids = result[0].ids
        except Exception:
            failed += 1
            continue
        if len(ids) < 2:
            continue
        all_ids.append(ids)
        token_counter.update(ids)
        total_tokens += len(ids)

    if total_tokens == 0:
        return {}

    vocab_size = tokenizer.vocab_size
    num_bpe_merges = vocab_size - base_vocab_size

    # --- Compression ---
    # Tokenize the same files with a base (non-BPE) tokenizer for reference.
    # We cache this externally, so here we estimate from the vocab structure.
    seq_lengths = np.array([len(s) for s in all_ids])

    # --- Frequency distribution ---
    freq_values = np.array(list(token_counter.values()))
    tokens_used = len(token_counter)
    tokens_unused = vocab_size - tokens_used

    freq_thresholds = [1, 10, 50, 100, 500]
    tokens_below_threshold = {}
    for t in freq_thresholds:
        count_below = sum(1 for f in freq_values if f < t)
        # Include completely unused tokens
        count_below += tokens_unused
        tokens_below_threshold[t] = count_below

    # --- Entropy ---
    probs = freq_values / freq_values.sum()
    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(vocab_size)
    entropy_ratio = entropy / max_entropy  # 1.0 = perfectly uniform

    # --- Embedding budget ---
    hidden_dim = 384  # from ModelConfig
    embedding_params = vocab_size * hidden_dim

    return {
        "vocab_size": vocab_size,
        "base_vocab_size": base_vocab_size,
        "num_bpe_merges": num_bpe_merges,
        "total_tokens": total_tokens,
        "num_files_ok": len(all_ids),
        "num_files_failed": failed,
        "mean_seq_len": float(seq_lengths.mean()),
        "median_seq_len": float(np.median(seq_lengths)),
        "p95_seq_len": float(np.percentile(seq_lengths, 95)),
        "tokens_used": tokens_used,
        "tokens_unused": tokens_unused,
        "pct_vocab_used": 100.0 * tokens_used / vocab_size,
        "tokens_below_threshold": tokens_below_threshold,
        "entropy_bits": float(entropy),
        "max_entropy_bits": float(max_entropy),
        "entropy_ratio": float(entropy_ratio),
        "embedding_params": embedding_params,
        "embedding_params_M": embedding_params / 1e6,
        "token_counter": token_counter,
        "seq_lengths": seq_lengths,
    }


# ──────────────────────────────────────────────────────────────────────
# Step 4: Compute compression ratio relative to base tokenizer
# ──────────────────────────────────────────────────────────────────────

def compute_base_token_counts(config: TokenizerConfig,
                              sample_files: list[Path]) -> dict[str, int]:
    """Tokenize with base (no-BPE) tokenizer for compression baseline."""
    tok = REMI(config)
    total = 0
    lengths = []
    failed = 0
    for path in tqdm(sample_files, desc="  Base tokenizer (no BPE)", leave=False):
        try:
            result = tok(path)
            ids = result[0].ids
        except Exception:
            failed += 1
            continue
        if len(ids) < 2:
            continue
        total += len(ids)
        lengths.append(len(ids))
    return {
        "total_tokens": total,
        "mean_seq_len": float(np.mean(lengths)) if lengths else 0,
        "num_files": len(lengths),
    }


# ──────────────────────────────────────────────────────────────────────
# Step 5: Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_results(all_metrics: list[dict],
                 base_info: dict,
                 output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab_sizes = [m["vocab_size"] for m in all_metrics]
    x_labels = [str(v // 1000) + "k" if v >= 1000 else str(v) for v in vocab_sizes]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Vocab Size Analysis — REMI + BPE on Guitar MIDI", fontsize=14, y=0.98)

    # ── 1. Compression ratio ──
    ax = axes[0, 0]
    if base_info["total_tokens"] > 0:
        compression_ratios = [base_info["total_tokens"] / m["total_tokens"]
                              for m in all_metrics]
        ax.plot(range(len(vocab_sizes)), compression_ratios, "o-", color="#2196F3",
                linewidth=2, markersize=7)
        ax.set_ylabel("Compression ratio (base / BPE tokens)")
    ax.set_xticks(range(len(vocab_sizes)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Vocab size")
    ax.set_title("Compression Ratio")
    ax.grid(True, alpha=0.3)

    # ── 2. Mean sequence length ──
    ax = axes[0, 1]
    means = [m["mean_seq_len"] for m in all_metrics]
    ax.plot(range(len(vocab_sizes)), means, "s-", color="#4CAF50",
            linewidth=2, markersize=7)
    ax.axhline(y=256, color="red", linestyle="--", alpha=0.5, label="max_seq_len=256")
    ax.set_xticks(range(len(vocab_sizes)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("Mean tokens per file")
    ax.set_title("Mean Sequence Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 3. % vocab actually used ──
    ax = axes[0, 2]
    pct_used = [m["pct_vocab_used"] for m in all_metrics]
    bars = ax.bar(range(len(vocab_sizes)), pct_used, color="#FF9800", alpha=0.8)
    ax.set_xticks(range(len(vocab_sizes)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("% of vocab used (≥1 occurrence)")
    ax.set_title("Vocabulary Utilization")
    ax.set_ylim(0, 105)
    for bar, pct in zip(bars, pct_used):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── 4. Tokens below frequency thresholds ──
    ax = axes[1, 0]
    thresholds = [10, 50, 100, 500]
    colors_t = ["#E91E63", "#9C27B0", "#3F51B5", "#009688"]
    for threshold, color in zip(thresholds, colors_t):
        fracs = [100.0 * m["tokens_below_threshold"][threshold] / m["vocab_size"]
                 for m in all_metrics]
        ax.plot(range(len(vocab_sizes)), fracs, "o-", color=color,
                linewidth=2, markersize=6, label=f"< {threshold} occ.")
    ax.set_xticks(range(len(vocab_sizes)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("% of vocab below threshold")
    ax.set_title("Underused Tokens")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # ── 5. Entropy ratio ──
    ax = axes[1, 1]
    ent_ratios = [m["entropy_ratio"] for m in all_metrics]
    ax.plot(range(len(vocab_sizes)), ent_ratios, "D-", color="#673AB7",
            linewidth=2, markersize=7)
    ax.set_xticks(range(len(vocab_sizes)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("Entropy / Max entropy")
    ax.set_title("Token Distribution Uniformity")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # ── 6. Embedding params vs model budget ──
    ax = axes[1, 2]
    emb_M = [m["embedding_params_M"] for m in all_metrics]
    ax.bar(range(len(vocab_sizes)), emb_M, color="#607D8B", alpha=0.8)
    # Rough estimate of non-embedding params for reference
    # 3 layers × ~(4h² + MoE overhead) ≈ 10-15M for hidden=384
    model_params_M = 12.0  # approximate non-embedding params
    ax.axhline(y=model_params_M, color="red", linestyle="--", alpha=0.6,
               label=f"~non-embedding params ({model_params_M:.0f}M)")
    ax.set_xticks(range(len(vocab_sizes)))
    ax.set_xticklabels(x_labels, rotation=45)
    ax.set_xlabel("Vocab size")
    ax.set_ylabel("Embedding params (M)")
    ax.set_title("Embedding Parameter Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_dir / "vocab_analysis_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Separate: Token frequency rank plot (log-log) for each vocab size ──
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_metrics)))
    for m, color in zip(all_metrics, cmap):
        freqs = sorted(m["token_counter"].values(), reverse=True)
        ranks = np.arange(1, len(freqs) + 1)
        ax2.loglog(ranks, freqs, color=color, alpha=0.8,
                   label=f"V={m['vocab_size']:,}")
    ax2.set_xlabel("Token rank (log)")
    ax2.set_ylabel("Frequency (log)")
    ax2.set_title("Token Frequency vs. Rank (Zipf plot)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")
    fig2.savefig(output_dir / "zipf_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Separate: Sequence length distributions ──
    fig3, axes3 = plt.subplots(1, len(all_metrics), figsize=(4 * len(all_metrics), 4),
                               sharey=True)
    if len(all_metrics) == 1:
        axes3 = [axes3]
    for ax, m in zip(axes3, all_metrics):
        ax.hist(m["seq_lengths"], bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(x=256, color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"V={m['vocab_size']:,}")
        ax.set_xlabel("Tokens per file")
    axes3[0].set_ylabel("Count")
    fig3.suptitle("Sequence Length Distributions", fontsize=12)
    fig3.tight_layout()
    fig3.savefig(output_dir / "seq_length_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    print(f"\nPlots saved to {output_dir}/")


# ──────────────────────────────────────────────────────────────────────
# Step 6: Summary report
# ──────────────────────────────────────────────────────────────────────

def print_report(all_metrics: list[dict], base_info: dict):
    sep = "=" * 90
    print(f"\n{sep}")
    print("VOCAB SIZE ANALYSIS REPORT")
    print(sep)

    print(f"\nBase vocabulary (no BPE): {all_metrics[0]['base_vocab_size']} tokens")
    if base_info["total_tokens"] > 0:
        print(f"Base tokenizer — mean seq len: {base_info['mean_seq_len']:.1f} "
              f"tokens/file over {base_info['num_files']} files")

    print(f"\n{'Vocab':>8} {'BPE':>7} {'Compr':>7} {'Mean':>8} {'Med':>8} "
          f"{'P95':>8} {'Used%':>7} {'<100':>7} {'Entropy':>8} {'Emb(M)':>8}")
    print("-" * 90)

    for m in all_metrics:
        comp = (base_info["total_tokens"] / m["total_tokens"]
                if base_info["total_tokens"] > 0 else float("nan"))
        below_100_pct = 100.0 * m["tokens_below_threshold"][100] / m["vocab_size"]
        print(f"{m['vocab_size']:>8,} {m['num_bpe_merges']:>7,} {comp:>7.2f}x "
              f"{m['mean_seq_len']:>8.1f} {m['median_seq_len']:>8.1f} "
              f"{m['p95_seq_len']:>8.1f} {m['pct_vocab_used']:>6.1f}% "
              f"{below_100_pct:>6.1f}% {m['entropy_ratio']:>8.3f} "
              f"{m['embedding_params_M']:>8.2f}")

    # ── Recommendation heuristic ──
    print(f"\n{sep}")
    print("RECOMMENDATIONS")
    print(sep)
    print("""
Consider the following when choosing your vocab size:

  1. COMPRESSION ELBOW: Pick the vocab size where the compression ratio
     curve starts to flatten. Beyond this point, additional merges yield
     diminishing returns in sequence length reduction.

  2. VOCABULARY UTILIZATION: Aim for >90% of the vocab being used at
     least once in the sample, and <20% of tokens appearing fewer than
     100 times. Unused or rare tokens waste embedding capacity.

  3. ENTROPY RATIO: Higher is better — it means the token distribution
     is more uniform. A sharp drop in entropy ratio indicates many tokens
     are absorbing very little probability mass.

  4. EMBEDDING BUDGET: For hidden_dim=384, the embedding table should
     ideally be 10-30% of total model parameters. If embeddings dominate,
     the model spends capacity on lookup tables rather than computation.

  5. SEQUENCE LENGTH: Given max_seq_len=256, check what fraction of files
     fit within the context window at each vocab size. More compression
     means more musical context per training example.

  6. DATASET SIZE RULE OF THUMB: With ~642k training files, you can
     support a larger vocab than a small dataset. But each BPE merge
     token should still appear at least ~100 times in the corpus for
     the model to learn a reliable embedding.
""")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vocab size analysis for REMI+BPE")
    parser.add_argument("--midi-dir", type=str,
                        default="C:/Users/Proprietario/repo/midi_data/giga_midi_guitars_train",
                        help="Directory containing training MIDI files")
    parser.add_argument("--sample-size", type=int, default=10000,
                        help="Number of files to sample for tokenization metrics "
                             "(default: 10000). Set to 0 to use all files.")
    parser.add_argument("--train-sample-size", type=int, default=50000,
                        help="Number of files to use for BPE training "
                             "(default: 50000). Set to 0 to use all files.")
    parser.add_argument("--vocab-sizes", type=str, default="1024,2048,10000",
                        help="Comma-separated list of vocab sizes to test")
    parser.add_argument("--output-dir", type=str, default="plots/vocab_analysis",
                        help="Directory for output plots")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizers",
                        help="Directory to save trained tokenizers "
                             "(default: tokenizers/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    midi_dir = Path(args.midi_dir)
    output_dir = Path(args.output_dir)
    tokenizer_dir = Path(args.tokenizer_dir)
    candidate_sizes = [int(v.strip()) for v in args.vocab_sizes.split(",")]

    print(f"MIDI directory: {midi_dir}")
    all_files = sorted(midi_dir.glob("*.mid"))
    print(f"Total MIDI files found: {len(all_files):,}")

    if len(all_files) == 0:
        print("ERROR: No .mid files found. Check --midi-dir path.", file=sys.stderr)
        sys.exit(1)

    # ── Sample files for BPE training ──
    if args.train_sample_size > 0 and args.train_sample_size < len(all_files):
        train_files = random.sample(all_files, args.train_sample_size)
        print(f"Using {len(train_files):,} files for BPE training")
    else:
        train_files = all_files
        print(f"Using all {len(train_files):,} files for BPE training")

    # ── Sample files for metrics evaluation ──
    if args.sample_size > 0 and args.sample_size < len(all_files):
        eval_files = random.sample(all_files, args.sample_size)
        print(f"Using {len(eval_files):,} files for metric evaluation")
    else:
        eval_files = all_files
        print(f"Using all {len(eval_files):,} files for metric evaluation")

    # ── Step 1: Base vocabulary ──
    config = make_tokenizer_config()
    base_vocab = count_base_vocab(config)
    print(f"\nBase REMI vocabulary (no BPE): {base_vocab} tokens")

    # Filter out candidate sizes smaller than base vocab
    valid_sizes = [v for v in candidate_sizes if v > base_vocab]
    skipped = [v for v in candidate_sizes if v <= base_vocab]
    if skipped:
        print(f"Skipping vocab sizes <= base vocab ({base_vocab}): {skipped}")
    if not valid_sizes:
        print("ERROR: All candidate sizes are <= base vocab size.", file=sys.stderr)
        sys.exit(1)

    # ── Step 2: Base tokenizer metrics (compression baseline) ──
    print("\nComputing base tokenizer metrics (no BPE)...")
    base_info = compute_base_token_counts(config, eval_files)

    # ── Step 3: Train and evaluate each candidate ──
    all_metrics = []
    for vocab_size in valid_sizes:
        print(f"\n--- Vocab size: {vocab_size:,} ---")
        print("  Training BPE tokenizer...")
        tok = train_tokenizer(config, train_files, vocab_size, save_dir=tokenizer_dir)

        print(f"  Actual vocab size after training: {tok.vocab_size}")
        print("  Computing metrics...")
        metrics = compute_metrics(tok, eval_files, base_vocab)
        if metrics:
            all_metrics.append(metrics)
            print(f"  Mean seq len: {metrics['mean_seq_len']:.1f}, "
                  f"Vocab used: {metrics['pct_vocab_used']:.1f}%, "
                  f"Entropy ratio: {metrics['entropy_ratio']:.3f}")
        else:
            print("  WARNING: No valid files tokenized at this size.")

    if not all_metrics:
        print("ERROR: No metrics computed. Check data.", file=sys.stderr)
        sys.exit(1)

    # ── Step 4: Report and plots ──
    print_report(all_metrics, base_info)
    plot_results(all_metrics, base_info, output_dir)

    # Save raw metrics to JSON (excluding non-serializable fields)
    json_metrics = []
    for m in all_metrics:
        m_copy = {k: v for k, v in m.items()
                  if k not in ("token_counter", "seq_lengths")}
        json_metrics.append(m_copy)
    json_path = output_dir / "vocab_analysis_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(json_metrics, f, indent=2)
    print(f"Raw metrics saved to {json_path}")


if __name__ == "__main__":
    main()
