import torch
import os
import json
import shutil
import logging
from datetime import datetime
from pathlib import Path
from miditok import REMI

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, global_step,
                    checkpoint_dir='checkpoints', metrics=None,
                    is_best=False, tokenizer_config_path=None):
    """
    Save model checkpoint with metadata.

    Args:
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        global_step: Current optimizer step
        checkpoint_dir: Directory to save checkpoints
        metrics: Dictionary of metrics to save
        is_best: Whether this is the best model so far
        tokenizer_config_path: Path to the tokenizer JSON config
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'vocab_size': model.config["vocab_size"],
    }

    if metrics:
        checkpoint['metrics'] = metrics

    checkpoint_filename = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}.pt')
    torch.save(checkpoint, checkpoint_filename)
    logger.info(f"Saved checkpoint: {checkpoint_filename}")

    metadata = {
        'global_step': global_step,
        'timestamp': checkpoint['timestamp'],
        'vocab_size': checkpoint['vocab_size'],
    }
    if metrics:
        metadata['metrics'] = metrics
    if tokenizer_config_path:
        metadata['tokenizer_config_path'] = str(tokenizer_config_path)

    metadata_filename = os.path.join(checkpoint_dir, f'checkpoint_step_{global_step}_metadata.json')
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)

    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
        shutil.copyfile(checkpoint_filename, best_model_path)
        logger.info(f"Saved as best model: {best_model_path}")

        best_metadata_path = os.path.join(checkpoint_dir, 'best_model_metadata.json')
        with open(best_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None,
                    device='cuda', load_tokenizer=True):
    """
    Load model and optimizer state from checkpoint.

    Returns:
        Tuple of (checkpoint_dict, tokenizer_or_None)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    vocab_mismatch = (
        'vocab_size' in checkpoint
        and checkpoint['vocab_size'] != model.config["vocab_size"]
    )
    if vocab_mismatch:
        logger.warning(
            "Vocab size mismatch: checkpoint=%d, model=%d. Loading with strict=False.",
            checkpoint['vocab_size'], model.config["vocab_size"],
        )
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    tokenizer = None
    if load_tokenizer:
        tokenizer_config_path = None

        metadata_path = str(checkpoint_path).replace('.pt', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            tokenizer_config_path = metadata.get('tokenizer_config_path')

        if tokenizer_config_path is None:
            best_meta = os.path.join(os.path.dirname(checkpoint_path), 'best_model_metadata.json')
            if os.path.exists(best_meta):
                with open(best_meta) as f:
                    metadata = json.load(f)
                tokenizer_config_path = metadata.get('tokenizer_config_path')

        if tokenizer_config_path and os.path.exists(tokenizer_config_path):
            tokenizer = REMI.from_pretrained(Path(tokenizer_config_path), local_files_only=True)
            logger.info(f"Loaded tokenizer from {tokenizer_config_path}")
        else:
            logger.warning("No tokenizer path found in checkpoint metadata. "
                           "Pass load_tokenizer=False or provide a valid tokenizer_config_path.")

    logger.info(f"Loaded checkpoint from step {checkpoint.get('global_step', 'unknown')}")
    return checkpoint, tokenizer


def get_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint in the given directory."""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [f for f in os.listdir(checkpoint_dir)
                        if f.endswith('.pt') and not f.startswith('best_model')]

    if not checkpoint_files:
        return None

    checkpoint_files.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )

    return os.path.join(checkpoint_dir, checkpoint_files[0])
