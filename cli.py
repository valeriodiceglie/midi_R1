import random
from copy import deepcopy
from dataclasses import fields
from enum import Enum
from pathlib import Path
from typing import List, Optional

import torch
import typer
from miditok import REMI
from miditok.pytorch_data import DataCollator
from torch.utils.data import DataLoader

from src.midiR1.config import ModelConfig, TrainConfig
from src.midiR1.data.cache import build_token_cache
from src.midiR1.data.dataset import PieceCropDataset
from src.midiR1.data.split import split_directory
from src.midiR1.engine.checkpointing import load_checkpoint
from src.midiR1.engine.evaluate import evaluate
from src.midiR1.engine.inference import GenerationConfig, MidiGenerator
from src.midiR1.engine.trainer import train
from src.midiR1.model.model import MidiR1
from src.midiR1.utils.diagnostics import sequence_diagnostics
from src.midiR1.utils.labeling import classify_section, label_directory
from src.midiR1.utils.metrics import compute_generation_metrics
from src.midiR1.utils.redirect import _silence_worker
from src.midiR1.utils.vocab_resize import resize_model_vocab

app = typer.Typer(
    name="midiR1",
    help="MidiR1: MIDI sequence modeling with DeepSeek-V3 transformer architecture.",
    add_completion=False,
)

# Conditioning token offsets (added on top of tokenizer.vocab_size)
CONDITIONING_TOKENS = {
    "SECTION_SOLO": 0,
    "SECTION_RHYTHM": 1,
    # 2-15 reserved for future conditioning tokens
}


# ─── helpers ────────────────────────────────────────────────────────────────


def _cast_value(v: str):
    """Auto-cast a CLI override string to int / float / bool / str."""
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _apply_overrides(config, overrides: List[str]):
    """Apply --override key=value pairs to a dataclass config."""
    field_names = {f.name for f in fields(config)}
    for item in overrides:
        if "=" not in item:
            typer.echo(f"Warning: ignoring malformed override '{item}' (expected key=value)")
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in field_names:
            typer.echo(f"Warning: unknown config key '{key}', ignoring")
            continue
        setattr(config, key, _cast_value(value.strip()))
    return config


def _load_tokenizer(config: TrainConfig) -> REMI:
    tok_path = Path(config.tokenizer_config_path)
    return REMI.from_pretrained(tok_path, local_files_only=True)


def _build_model(model_config: ModelConfig) -> MidiR1:
    return MidiR1(model_config.to_dict())


def _build_dataloader(
    cache_dir: str,
    model_config: ModelConfig,
    collator: DataCollator,
    batch_size: int,
    pad_token_id: int,
    length_exponent: float = 0.0,
    shuffle: bool = False,
    num_workers: int = 6,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
) -> DataLoader:
    dataset = PieceCropDataset(
        cache_dir=cache_dir,
        max_seq_len=model_config.max_seq_len,
        min_seq_len=model_config.min_seq_len,
        pad_token_id=pad_token_id,
        length_exponent=length_exponent,
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_silence_worker,
        pin_memory=True,
    )
    if persistent_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def _init_wandb(
    project: str,
    run_name: Optional[str],
    config: dict,
    disabled: bool,
    job_type: str = "train",
    run_id: Optional[str] = None,
    resume: Optional[bool] = None,
):
    """Initialize a wandb run. Returns the run object or None if disabled."""
    if disabled:
        return None
    try:
        import wandb
    except ImportError:
        typer.echo("Warning: wandb not installed. Logging disabled. Install with: pip install wandb")
        return None

    init_kwargs = {
        "project": project,
        "name": run_name,
        "config": config,
        "job_type": job_type,
    }
    if run_id is not None:
        init_kwargs["id"] = run_id
    if resume is not None:
        init_kwargs["resume"] = resume

    run = wandb.init(**init_kwargs)
    return run


def _finish_wandb(run):
    if run is not None:
        run.finish()


# ─── split-data command ──────────────────────────────────────────────────


@app.command("split-data")
def split_data_cmd(
    src_dir: str = typer.Argument(..., help="Directory containing .mid files to split."),
    dst_dir: Optional[str] = typer.Option(None, help="Output parent dir (default: src_dir + '_split')."),
    train_ratio: float = typer.Option(0.8, help="Fraction for training."),
    val_ratio: float = typer.Option(0.1, help="Fraction for validation."),
    test_ratio: float = typer.Option(0.1, help="Fraction for test."),
    seed: int = typer.Option(42, help="Random seed for shuffling."),
):
    """Split a flat directory of .mid files into train/val/test sub-directories."""
    src = Path(src_dir)
    dst = Path(dst_dir) if dst_dir else src.parent / (src.name + "_split")

    counts = split_directory(
        src_dir=src,
        dst_dir=dst,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    typer.echo(f"Split complete: train={counts['train']}, val={counts['val']}, test={counts['test']}")
    typer.echo(f"Output: {dst}")


# ─── plot-lengths command ─────────────────────────────────────────────────


@app.command("plot-lengths")
def plot_lengths_cmd(
    cache_dir: Optional[str] = typer.Option(None, help="Token cache directory (default: config token_cache_dir)."),
    output: str = typer.Option("plots/length_distribution.png", help="Output plot file."),
    override: Optional[List[str]] = typer.Option(None, help="Config overrides as key=value pairs."),
):
    """Plot token length distribution from cached manifests."""
    import json

    t_cfg = TrainConfig()
    if override:
        t_cfg = _apply_overrides(t_cfg, override)

    base = Path(cache_dir or t_cfg.token_cache_dir)

    all_lengths = []
    split_lengths = {}
    for split_name in ["train", "val", "test"]:
        manifest_path = base / split_name / "manifest.json"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        lengths = [entry["length"] for entry in manifest]
        all_lengths.extend(lengths)
        split_lengths[split_name] = lengths
        typer.echo(f"  {split_name}: {len(lengths)} pieces, "
                   f"min={min(lengths)}, max={max(lengths)}, "
                   f"mean={sum(lengths)/len(lengths):.0f}")

    if not all_lengths:
        typer.echo("No manifests found. Run cache-tokens first.")
        raise typer.Exit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        typer.echo("matplotlib not installed. Install with: pip install matplotlib")
        typer.echo(f"Total: {len(all_lengths)} pieces, "
                   f"min={min(all_lengths)}, max={max(all_lengths)}, "
                   f"mean={sum(all_lengths)/len(all_lengths):.0f}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for split_name, lengths in split_lengths.items():
        ax.hist(lengths, bins=50, alpha=0.6, label=split_name)
    ax.set_xlabel("Token sequence length")
    ax.set_ylabel("Number of pieces")
    ax.set_title("Token Length Distribution")
    ax.legend()

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved plot to {output}")


# ─── cache-tokens command ──────────────────────────────────────────────────


@app.command("cache-tokens")
def cache_tokens_cmd(
    condition: bool = typer.Option(False, help="Auto-label solo/rhythm and insert conditioning tokens (legacy)."),
    attribute_controls: bool = typer.Option(False, help="Use MIDITok attribute controls for conditioning."),
    override: Optional[List[str]] = typer.Option(None, help="Config overrides as key=value pairs."),
):
    """Tokenize full-length MIDI files and save token caches."""
    t_cfg = TrainConfig()
    m_cfg = ModelConfig()
    if override:
        t_cfg = _apply_overrides(t_cfg, override)
        m_cfg = _apply_overrides(m_cfg, override)

    use_ac = attribute_controls or t_cfg.use_attribute_controls

    tokenizer = _load_tokenizer(t_cfg)
    base_vocab = tokenizer.vocab_size

    conditioning_fn = None
    if condition and not use_ac:
        from symusic import Score as SyScore

        solo_id = base_vocab + CONDITIONING_TOKENS["SECTION_SOLO"]
        rhythm_id = base_vocab + CONDITIONING_TOKENS["SECTION_RHYTHM"]

        def conditioning_fn(path):
            try:
                score = SyScore(str(path))
            except Exception:
                return None
            label = classify_section(score)
            cond_id = solo_id if label == "solo" else rhythm_id
            return cond_id, label

    if use_ac:
        typer.echo(f"Using attribute controls (AC tokens in vocab: "
                   f"{len(tokenizer.attribute_controls)} ACs configured)")

    for split_name, midi_dir in [("train", t_cfg.train), ("val", t_cfg.val), ("test", t_cfg.test)]:
        midi_path = Path(midi_dir)
        if not midi_path.exists():
            typer.echo(f"Skipping {split_name}: {midi_dir} not found")
            continue
        cache_dir = Path(t_cfg.token_cache_dir) / split_name
        typer.echo(f"Caching {split_name}: {midi_dir} -> {cache_dir}")
        build_token_cache(
            midi_dir=midi_path,
            cache_dir=cache_dir,
            tokenizer=tokenizer,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            min_seq_len=m_cfg.min_seq_len,
            conditioning_fn=conditioning_fn,
            use_attribute_controls=use_ac,
            ac_tracks_ratio_range=t_cfg.ac_tracks_ratio_range,
            ac_bars_ratio_range=t_cfg.ac_bars_ratio_range,
        )

    typer.echo("Token caching complete.")


# ─── train command ──────────────────────────────────────────────────────────


@app.command("train")
def train_cmd(
    resume: Optional[str] = typer.Option(None, help="Checkpoint path to resume training from."),
    wandb_project: Optional[str] = typer.Option(None, help="W&B project name (overrides config)."),
    wandb_run_name: Optional[str] = typer.Option(None, help="W&B run name (auto-generated if omitted)."),
    wandb_run_id: Optional[str] = typer.Option(None, help="W&B run id to resume (optional)."),
    wandb_resume: Optional[bool] = typer.Option(None, help="Resume W&B run if run id is provided."),
    wandb_disabled: bool = typer.Option(False, help="Disable W&B logging entirely."),
    override: Optional[List[str]] = typer.Option(None, help="Config overrides as key=value pairs."),
    seed: int = typer.Option(42, help="Random seed."),
):
    """Train the MidiR1 model."""
    random.seed(seed)
    torch.manual_seed(seed)

    t_cfg = TrainConfig()
    m_cfg = ModelConfig()

    if override:
        t_cfg = _apply_overrides(t_cfg, override)
        m_cfg = _apply_overrides(m_cfg, override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_cfg.device = device

    typer.echo(f"Training: eval_steps={t_cfg.eval_steps}, "
               f"save_steps={t_cfg.save_steps}, "
               f"patience={t_cfg.patience}, "
               f"total_steps={t_cfg.total_steps}, "
               f"gradient_accumulation_steps={t_cfg.gradient_accumulation_steps}, "
               f"eval_max_batches={t_cfg.eval_max_batches}")

    tokenizer = _load_tokenizer(t_cfg)
    m_cfg.vocab_size = tokenizer.vocab_size
    t_cfg.pad_token_id = tokenizer.pad_token_id

    model = _build_model(m_cfg)

    collator = DataCollator(
        tokenizer.pad_token_id,
        shift_labels=True,
        labels_pad_idx=tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
    )

    batch_size = t_cfg.batch_size
    cache_base = Path(t_cfg.token_cache_dir)

    train_loader = _build_dataloader(
        str(cache_base / "train"), m_cfg, collator,
        batch_size=batch_size, pad_token_id=tokenizer.pad_token_id,
        length_exponent=t_cfg.length_exponent,
        shuffle=True,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = _build_dataloader(
        str(cache_base / "val"), m_cfg, collator,
        batch_size=batch_size, pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )
    test_loader = _build_dataloader(
        str(cache_base / "test"), m_cfg, collator,
        batch_size=batch_size, pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        num_workers=6,
    )

    typer.echo(f"Train batches: {len(train_loader)}")
    typer.echo(f"Val batches:   {len(val_loader)}")
    typer.echo(f"Test batches:  {len(test_loader)}")

    # wandb
    wb_project = wandb_project or t_cfg.wandb_project
    wb_disabled = wandb_disabled or t_cfg.wandb_disabled

    merged_config = {**t_cfg.to_dict(), **m_cfg.to_dict()}
    if wandb_run_id is not None and wandb_resume is None:
        wandb_resume = True

    run = _init_wandb(
        wb_project,
        wandb_run_name,
        merged_config,
        wb_disabled,
        job_type="train",
        run_id=wandb_run_id,
        resume=wandb_resume,
    )
    t_cfg.wandb_run = run

    model.to(device)

    train(model, resume, train_loader, val_loader, test_loader, t_cfg)

    _finish_wandb(run)
    typer.echo("Training complete.")


# ─── evaluate command ───────────────────────────────────────────────────────


class SplitChoice(str, Enum):
    val = "val"
    test = "test"
    both = "both"


@app.command("evaluate")
def evaluate_cmd(
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path (default: best_model.pt)."),
    split: SplitChoice = typer.Option(SplitChoice.both, help="Which split to evaluate."),
    reference_dir: Optional[str] = typer.Option(None, help="Directory of reference .mid files for MIDI-level metrics (KL/OA/FMD)."),
    generated_dir: Optional[str] = typer.Option(None, help="Directory of generated .mid files for MIDI-level metrics (KL/OA/FMD)."),
    wandb_project: Optional[str] = typer.Option(None, help="W&B project name."),
    wandb_disabled: bool = typer.Option(False, help="Disable W&B logging."),
    override: Optional[List[str]] = typer.Option(None, help="Config overrides as key=value pairs."),
):
    """Evaluate the MidiR1 model on validation and/or test splits.

    Optionally compute MIDI-level generation metrics (KL divergence,
    overlapping area, Frechet Music Distance) when --reference-dir and
    --generated-dir are both provided.
    """
    t_cfg = TrainConfig()
    m_cfg = ModelConfig()

    if override:
        t_cfg = _apply_overrides(t_cfg, override)
        m_cfg = _apply_overrides(m_cfg, override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _load_tokenizer(t_cfg)
    m_cfg.vocab_size = tokenizer.vocab_size

    model = _build_model(m_cfg)

    ckpt_path = checkpoint or str(Path(t_cfg.checkpoint_dir) / "best_model.pt")
    load_checkpoint(ckpt_path, model, load_tokenizer=False)
    model.to(device)

    collator = DataCollator(
        tokenizer.pad_token_id,
        shift_labels=True,
        labels_pad_idx=tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
    )

    # wandb
    wb_project = wandb_project or t_cfg.wandb_project
    wb_disabled = wandb_disabled or t_cfg.wandb_disabled
    run = _init_wandb(wb_project, None, {**t_cfg.to_dict(), **m_cfg.to_dict()}, wb_disabled, job_type="evaluate")

    splits_to_run = []
    if split in (SplitChoice.val, SplitChoice.both):
        splits_to_run.append("val")
    if split in (SplitChoice.test, SplitChoice.both):
        splits_to_run.append("test")

    cache_base = Path(t_cfg.token_cache_dir)

    for split_name in splits_to_run:
        cache_dir = str(cache_base / split_name)
        if not Path(cache_dir).exists():
            typer.echo(f"Skipping {split_name}: cache not found ({cache_dir})")
            continue

        loader = _build_dataloader(
            cache_dir, m_cfg, collator,
            batch_size=t_cfg.batch_size, pad_token_id=tokenizer.pad_token_id,
            shuffle=False, num_workers=4,
        )

        metrics = evaluate(
            model, loader, tokenizer.pad_token_id, device,
            label_smoothing=t_cfg.label_smoothing,
        )

        typer.echo(
            f"[{split_name.upper()}] Loss: {metrics['loss']:.4f}, "
            f"PPL: {metrics['perplexity']:.2f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"KL: {metrics['kl_divergence']:.4f}, "
            f"OA: {metrics['overlapping_area']:.4f}"
        )

        if run is not None:
            run.log({
                f"{split_name}/loss": metrics["loss"],
                f"{split_name}/perplexity": metrics["perplexity"],
                f"{split_name}/accuracy": metrics["accuracy"],
                f"{split_name}/kl_divergence": metrics["kl_divergence"],
                f"{split_name}/overlapping_area": metrics["overlapping_area"],
            })

    # ── MIDI-level generation metrics (KL / OA / FMD) ──
    if reference_dir is not None and generated_dir is not None:
        typer.echo(f"\nComputing MIDI-level metrics: {reference_dir} vs {generated_dir}")
        midi_metrics = compute_generation_metrics(Path(reference_dir), Path(generated_dir))
        typer.echo(
            f"  Pitch  KL: {midi_metrics['pitch_kl']:.4f}  OA: {midi_metrics['pitch_oa']:.4f}\n"
            f"  Vel    KL: {midi_metrics['velocity_kl']:.4f}  OA: {midi_metrics['velocity_oa']:.4f}\n"
            f"  Dur    KL: {midi_metrics['duration_kl']:.4f}  OA: {midi_metrics['duration_oa']:.4f}\n"
            f"  Mean   KL: {midi_metrics['mean_kl']:.4f}  OA: {midi_metrics['mean_oa']:.4f}\n"
            f"  FMD:   {midi_metrics['fmd']:.4f}"
        )
        if run is not None:
            run.log({f"midi/{k}": v for k, v in midi_metrics.items()})

    _finish_wandb(run)


# ─── fine-tune command ────────────────────────────────────────────────────


@app.command("fine-tune")
def finetune_cmd(
    checkpoint: str = typer.Option(..., help="Pre-trained checkpoint path."),
    data_dir: Optional[str] = typer.Option(None, help="Single directory of .mid files (auto-splits into train/val/test)."),
    attribute_controls: bool = typer.Option(False, help="Use MIDITok attribute controls for conditioning."),
    wandb_project: Optional[str] = typer.Option(None, help="W&B project name (overrides config)."),
    wandb_run_name: Optional[str] = typer.Option(None, help="W&B run name (auto-generated if omitted)."),
    wandb_disabled: bool = typer.Option(False, help="Disable W&B logging entirely."),
    override: Optional[List[str]] = typer.Option(None, help="Config overrides as key=value pairs."),
    seed: int = typer.Option(42, help="Random seed."),
):
    """Fine-tune MidiR1 from a pre-trained checkpoint with conditioning tokens.

    If --data-dir is provided, the MIDI files are auto-split into train/val/test,
    tokenized with conditioning labels, and cached. Otherwise, expects pre-split
    train/val/test paths in the config (set via --override).

    Use --attribute-controls to condition on MIDITok attribute controls (density,
    polyphony, duration, pitch class, repetition) instead of the legacy solo/rhythm
    tokens.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    t_cfg = TrainConfig()
    m_cfg = ModelConfig()

    if override:
        t_cfg = _apply_overrides(t_cfg, override)
        m_cfg = _apply_overrides(m_cfg, override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_cfg.device = device

    use_ac = attribute_controls or t_cfg.use_attribute_controls

    tokenizer = _load_tokenizer(t_cfg)
    base_vocab = tokenizer.vocab_size
    t_cfg.pad_token_id = tokenizer.pad_token_id

    if use_ac:
        # AC tokens are already in the tokenizer's vocabulary, so the
        # tokenizer.vocab_size is the full vocab including AC tokens.
        # The pre-trained model likely has a smaller vocab, so we resize.
        new_vocab_size = base_vocab
        m_cfg.vocab_size = new_vocab_size
        typer.echo(f"Fine-tuning with attribute controls: vocab={new_vocab_size} "
                   f"(ACs: {len(tokenizer.attribute_controls)} configured)")
    else:
        new_vocab_size = base_vocab + t_cfg.conditioning_tokens
        m_cfg.vocab_size = new_vocab_size
        typer.echo(f"Fine-tuning: base_vocab={base_vocab}, new_vocab={new_vocab_size}, "
                   f"conditioning_tokens={t_cfg.conditioning_tokens}")

    # ── Auto-split + auto-cache if --data-dir is given ──
    if data_dir is not None:
        src = Path(data_dir)
        split_dst = src.parent / (src.name + "_split")
        if not (split_dst / "train").exists():
            typer.echo(f"Splitting {src} into {split_dst} ...")
            counts = split_directory(src, split_dst, seed=seed)
            typer.echo(f"  train={counts['train']}, val={counts['val']}, test={counts['test']}")
        else:
            typer.echo(f"Using existing split at {split_dst}")

        # Point config at split directories
        t_cfg.train = str(split_dst / "train")
        t_cfg.val = str(split_dst / "val")
        t_cfg.test = str(split_dst / "test")

        conditioning_fn = None
        if not use_ac:
            # Build legacy conditioning function
            from symusic import Score as SyScore

            solo_id = base_vocab + CONDITIONING_TOKENS["SECTION_SOLO"]
            rhythm_id = base_vocab + CONDITIONING_TOKENS["SECTION_RHYTHM"]

            def conditioning_fn(path):
                try:
                    score = SyScore(str(path))
                except Exception:
                    return None
                label = classify_section(score)
                cond_id = solo_id if label == "solo" else rhythm_id
                return cond_id, label

        # Cache tokens for each split
        cache_base = Path(t_cfg.token_cache_dir)
        for split_name, midi_dir in [("train", t_cfg.train), ("val", t_cfg.val), ("test", t_cfg.test)]:
            midi_path = Path(midi_dir)
            if not midi_path.exists() or not list(midi_path.glob("*.mid")):
                continue
            cache_dir = cache_base / split_name
            if not (cache_dir / "manifest.json").exists():
                typer.echo(f"Caching {split_name}: {midi_dir} -> {cache_dir}")
                build_token_cache(
                    midi_dir=midi_path,
                    cache_dir=cache_dir,
                    tokenizer=tokenizer,
                    bos_token_id=tokenizer["BOS_None"],
                    eos_token_id=tokenizer["EOS_None"],
                    min_seq_len=m_cfg.min_seq_len,
                    conditioning_fn=conditioning_fn,
                    use_attribute_controls=use_ac,
                    ac_tracks_ratio_range=t_cfg.ac_tracks_ratio_range,
                    ac_bars_ratio_range=t_cfg.ac_bars_ratio_range,
                )

    # ── Build model, load pre-trained weights, resize if needed ──
    # Load checkpoint to discover its vocab size
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_vocab = ckpt.get("vocab_size", new_vocab_size)
    del ckpt

    m_cfg.vocab_size = ckpt_vocab
    model = _build_model(m_cfg)
    load_checkpoint(checkpoint, model, load_tokenizer=False)

    if ckpt_vocab != new_vocab_size:
        typer.echo(f"Resizing model vocab: {ckpt_vocab} -> {new_vocab_size}")
        resize_model_vocab(model, new_vocab_size)
        m_cfg.vocab_size = new_vocab_size

    model.to(device)

    # Apply learning rate scaling
    base_lr = t_cfg.learning_rate
    effective_lr = base_lr * t_cfg.finetune_lr_scale
    t_cfg.learning_rate = effective_lr
    typer.echo(f"Effective learning rate: {effective_lr:.2e} "
               f"(base={base_lr:.2e} * scale={t_cfg.finetune_lr_scale})")

    # Freeze early layers if configured
    if t_cfg.finetune_freeze_layers > 0:
        for idx, layer in enumerate(model.layers):
            if idx < t_cfg.finetune_freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        typer.echo(f"Froze first {t_cfg.finetune_freeze_layers} transformer layers")

    collator = DataCollator(
        tokenizer.pad_token_id,
        shift_labels=True,
        labels_pad_idx=tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
    )

    batch_size = t_cfg.batch_size
    cache_base = Path(t_cfg.token_cache_dir)

    train_loader = _build_dataloader(
        str(cache_base / "train"), m_cfg, collator,
        batch_size=batch_size, pad_token_id=tokenizer.pad_token_id,
        length_exponent=t_cfg.length_exponent,
        shuffle=True,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )
    val_loader = _build_dataloader(
        str(cache_base / "val"), m_cfg, collator,
        batch_size=batch_size, pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        num_workers=6, persistent_workers=True, prefetch_factor=4,
    )
    test_loader = _build_dataloader(
        str(cache_base / "test"), m_cfg, collator,
        batch_size=batch_size, pad_token_id=tokenizer.pad_token_id,
        shuffle=False,
        num_workers=6,
    )

    typer.echo(f"Train batches: {len(train_loader)}")
    typer.echo(f"Val batches:   {len(val_loader)}")
    typer.echo(f"Test batches:  {len(test_loader)}")

    # wandb
    wb_project = wandb_project or t_cfg.wandb_project
    wb_disabled = wandb_disabled or t_cfg.wandb_disabled

    merged_config = {**t_cfg.to_dict(), **m_cfg.to_dict(), "finetune": True}
    run = _init_wandb(wb_project, wandb_run_name, merged_config, wb_disabled, job_type="fine-tune")
    t_cfg.wandb_run = run

    train(model, None, train_loader, val_loader, test_loader, t_cfg)

    _finish_wandb(run)
    typer.echo("Fine-tuning complete.")


# ─── inference command ──────────────────────────────────────────────────────


@app.command("inference")
def inference_cmd(
    checkpoint: Optional[str] = typer.Option(None, help="Checkpoint path (default: best_model.pt)."),
    input_dir: Optional[str] = typer.Option(None, help="Directory with prompt MIDI files (default: config 'gen')."),
    output_dir: Optional[str] = typer.Option(None, help="Directory for generated output (default: config 'generate_dir')."),
    reference_dir: Optional[str] = typer.Option(None, help="Reference MIDI dir for post-generation metrics (KL/OA/FMD)."),
    condition: Optional[str] = typer.Option(None, help="Conditioning: 'solo' or 'rhythm'. Requires fine-tuned model (legacy)."),
    attribute_controls: bool = typer.Option(False, help="Encode prompts with attribute controls derived from the input MIDI."),
    max_length: int = typer.Option(128, help="Maximum generation length in tokens."),
    temperature: float = typer.Option(0.8, help="Sampling temperature."),
    top_k: int = typer.Option(50, help="Top-k sampling."),
    top_p: float = typer.Option(0.9, help="Top-p (nucleus) sampling."),
    repetition_penalty: float = typer.Option(1.2, help="Repetition penalty."),
    no_repeat_ngram_size: int = typer.Option(3, help="N-gram repetition blocking size."),
    use_mtp: bool = typer.Option(True, help="Enable MTP speculative decoding."),
    num_beams: int = typer.Option(1, help="Beam search width (1 = greedy/sampling)."),
    wandb_disabled: bool = typer.Option(True, help="Disable W&B logging (default for inference)."),
    override: Optional[List[str]] = typer.Option(None, help="Config overrides as key=value pairs."),
):
    """Generate MIDI continuations from prompt files."""
    t_cfg = TrainConfig()
    m_cfg = ModelConfig()

    if override:
        t_cfg = _apply_overrides(t_cfg, override)
        m_cfg = _apply_overrides(m_cfg, override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = _load_tokenizer(t_cfg)
    base_vocab = tokenizer.vocab_size
    use_ac = attribute_controls or t_cfg.use_attribute_controls

    # Determine vocab size — AC tokens are part of the tokenizer vocab;
    # legacy conditioning adds extra tokens on top.
    if use_ac:
        m_cfg.vocab_size = base_vocab
    elif condition is not None:
        m_cfg.vocab_size = base_vocab + t_cfg.conditioning_tokens
    else:
        m_cfg.vocab_size = base_vocab

    model = _build_model(m_cfg)

    ckpt_path = checkpoint or str(Path(t_cfg.checkpoint_dir) / "best_model.pt")
    load_checkpoint(ckpt_path, model, load_tokenizer=False)

    gen_config = GenerationConfig(
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        do_sample=True,
        use_mtp=use_mtp,
        mtp_speculation_mode=use_mtp,
        num_beams=num_beams,
        length_penalty=1.5,
        early_stopping=False,
        eos_token_id=tokenizer["EOS_None"],
        pad_token_id=tokenizer.pad_token_id,
    )

    generator = MidiGenerator(model, tokenizer, device=str(device))

    in_dir = Path(input_dir or t_cfg.gen)
    out_dir = Path(output_dir or t_cfg.generate_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    midi_paths = sorted(in_dir.glob("*.mid"))
    if not midi_paths:
        typer.echo(f"No .mid files found in {in_dir}")
        raise typer.Exit(1)

    # Resolve conditioning token ID (legacy)
    cond_token_id = None
    if condition is not None and not use_ac:
        cond_key = f"SECTION_{condition.upper()}"
        if cond_key not in CONDITIONING_TOKENS:
            typer.echo(f"Unknown condition '{condition}'. Use 'solo' or 'rhythm'.")
            raise typer.Exit(1)
        cond_token_id = base_vocab + CONDITIONING_TOKENS[cond_key]
        typer.echo(f"Conditioning: {condition} (token_id={cond_token_id})")

    typer.echo(f"Generating from {len(midi_paths)} prompts...")

    # Encode prompts — with or without attribute controls
    if use_ac and tokenizer.attribute_controls:
        from symusic import Score as SyScore
        prompt_seqs = []
        for p in midi_paths:
            score = SyScore(str(p))
            score = tokenizer.preprocess_score(score)
            # Apply all ACs to all tracks/bars for inference conditioning
            ac_indexes = {}
            from miditok.attribute_controls import BarAttributeControl
            for ti in range(len(score.tracks)):
                track_acs = {}
                for ai, ac in enumerate(tokenizer.attribute_controls):
                    if isinstance(ac, BarAttributeControl):
                        tpq = score.ticks_per_quarter or 480
                        n_bars = max(int(score.end() / (tpq * 4)), 1)
                        track_acs[ai] = list(range(n_bars))
                    else:
                        track_acs[ai] = True
                ac_indexes[ti] = track_acs
            tok_result = tokenizer.encode(
                score,
                no_preprocess_score=True,
                attribute_controls_indexes=ac_indexes,
            )
            prompt_seqs.append(tok_result)
        typer.echo(f"Encoded prompts with attribute controls ({len(tokenizer.attribute_controls)} ACs)")
    else:
        prompt_seqs = [tokenizer(p) for p in midi_paths]

    # Inject legacy conditioning token after BOS in each prompt
    if cond_token_id is not None and not use_ac:
        for seq in prompt_seqs:
            ids = seq[0].ids
            if ids and ids[0] == tokenizer["BOS_None"]:
                seq[0].ids = [ids[0], cond_token_id] + ids[1:]
            else:
                seq[0].ids = [cond_token_id] + ids

    continuations = generator.generate(prompt_seqs, gen_config)

    if not isinstance(continuations, list):
        continuations = [continuations]

    all_diags = []
    for idx, (in_seq, full_seq) in enumerate(zip(prompt_seqs, continuations)):
        gen_ids = full_seq.ids[len(in_seq[0].ids):]
        tokens_list = [gen_ids, in_seq[0].ids, full_seq.ids]

        midi_main = tokenizer.decode([deepcopy(gen_ids)])
        for tok_ids in tokens_list[1:]:
            track_midi = tokenizer.decode([deepcopy(tok_ids)])
            midi_main.tracks.append(track_midi.tracks[0])

        midi_main.tracks[0].name = f"Continuation ({len(gen_ids)} tokens)"
        midi_main.tracks[1].name = f"Prompt ({len(in_seq[0].ids)} tokens)"
        midi_main.tracks[2].name = "Prompt + Continuation"

        midi_main.dump_midi(out_dir / f"{idx}.mid")
        tokenizer.save_tokens(tokens_list, out_dir / f"{idx}.json")

        # Diagnostics on the generated continuation
        gen_score = tokenizer.decode([deepcopy(gen_ids)])
        diag = sequence_diagnostics(gen_ids, score=gen_score)
        all_diags.append(diag)
        typer.echo(
            f"  [{idx}] tokens={len(gen_ids)}  "
            f"rep3={diag['rep_3gram']:.2f}  "
            f"entropy={diag['token_entropy']:.2f}  "
            f"density={diag.get('note_density', 0):.2f}  "
            f"pitch={diag.get('pitch_min', 0)}-{diag.get('pitch_max', 0)}"
        )

    # Print aggregate diagnostics
    if all_diags:
        n = len(all_diags)
        avg = {k: sum(d[k] for d in all_diags) / n for k in all_diags[0]}
        typer.echo(
            f"Avg over {n}: "
            f"rep2={avg['rep_2gram']:.3f}  rep3={avg['rep_3gram']:.3f}  "
            f"rep4={avg['rep_4gram']:.3f}  entropy={avg['token_entropy']:.2f}  "
            f"density={avg.get('note_density', 0):.2f}  "
            f"pitch_range={avg.get('pitch_range', 0):.1f}"
        )

    typer.echo(f"Saved {len(continuations)} generations to {out_dir}")

    # ── MIDI-level generation metrics ──
    midi_metrics = None
    if reference_dir is not None:
        typer.echo(f"\nComputing MIDI-level metrics vs reference: {reference_dir}")
        try:
            midi_metrics = compute_generation_metrics(Path(reference_dir), out_dir)
            typer.echo(
                f"  Pitch  KL: {midi_metrics['pitch_kl']:.4f}  OA: {midi_metrics['pitch_oa']:.4f}\n"
                f"  Vel    KL: {midi_metrics['velocity_kl']:.4f}  OA: {midi_metrics['velocity_oa']:.4f}\n"
                f"  Dur    KL: {midi_metrics['duration_kl']:.4f}  OA: {midi_metrics['duration_oa']:.4f}\n"
                f"  Mean   KL: {midi_metrics['mean_kl']:.4f}  OA: {midi_metrics['mean_oa']:.4f}\n"
                f"  FMD:   {midi_metrics['fmd']:.4f}"
            )
        except Exception as e:
            typer.echo(f"Warning: could not compute MIDI-level metrics: {e}")

    # wandb artifact logging
    if not wandb_disabled:
        try:
            import wandb

            wb_project = t_cfg.wandb_project
            run = wandb.init(project=wb_project, job_type="inference")
            artifact = wandb.Artifact("generated-midi", type="generation")
            artifact.add_dir(str(out_dir))
            run.log_artifact(artifact)
            if all_diags:
                n = len(all_diags)
                avg = {k: sum(d[k] for d in all_diags) / n for k in all_diags[0]}
                run.log({f"gen/{k}": v for k, v in avg.items()})
            if midi_metrics is not None:
                run.log({f"midi/{k}": v for k, v in midi_metrics.items()})
            run.finish()
        except ImportError:
            pass


# ─── entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    app()
