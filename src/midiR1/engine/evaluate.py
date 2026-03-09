"""
Standalone evaluation for the MidiR1 model.
Loads the best checkpoint, runs validation/test, and reports loss + token accuracy.
"""
import torch
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from miditok import REMI
from miditok.pytorch_data import DataCollator
import logging
import math
import os

from src.midiR1.config import ModelConfig, TrainConfig
from src.midiR1.data.dataset import PieceCropDataset
from src.midiR1.model.model import MidiR1
from src.midiR1.engine.checkpointing import load_checkpoint
from src.midiR1.engine.trainer import validate

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/evaluate.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EVALUATOR")


def evaluate(model, data_loader, pad_token_id, device, label_smoothing=0.0):
    """
    Run full evaluation and return metrics dict.

    Returns:
        Dictionary with 'loss', 'accuracy', 'perplexity', 'kl_divergence',
        and 'overlapping_area' keys.
    """
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=label_smoothing,
    )
    val_loss, val_accuracy, _mtp_loss, val_kl, val_oa = validate(
        model, data_loader, criterion, pad_token_id, device,
    )
    val_ppl = math.exp(min(val_loss, 100.0))
    return {
        "loss": val_loss,
        "accuracy": val_accuracy,
        "perplexity": val_ppl,
        "kl_divergence": val_kl,
        "overlapping_area": val_oa,
    }


def main():
    """Standalone evaluation entry point."""
    t_cfg = TrainConfig()
    m_cfg = ModelConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = REMI.from_pretrained(
        Path(t_cfg.tokenizer_config_path),
        local_files_only=True,
    )
    m_cfg.vocab_size = tokenizer.vocab_size

    model = MidiR1(m_cfg.to_dict())
    best_path = os.path.join(t_cfg.checkpoint_dir, "best_model.pt")
    load_checkpoint(best_path, model, load_tokenizer=False)
    model.to(device)

    collator = DataCollator(
        tokenizer.pad_token_id,
        shift_labels=True,
        labels_pad_idx=tokenizer.pad_token_id,
        copy_inputs_as_labels=True,
    )

    for split_name in ["val", "test"]:
        cache_dir = Path(t_cfg.token_cache_dir) / split_name
        if not cache_dir.exists():
            logger.warning(f"Skipping {split_name}: cache not found ({cache_dir})")
            continue

        dataset = PieceCropDataset(
            cache_dir=cache_dir,
            max_seq_len=m_cfg.max_seq_len,
            min_seq_len=m_cfg.min_seq_len,
            pad_token_id=tokenizer.pad_token_id,
        )
        loader = DataLoader(
            dataset,
            batch_size=t_cfg.batch_size,
            collate_fn=collator,
            shuffle=False,
            num_workers=4,
        )

        metrics = evaluate(model, loader, tokenizer.pad_token_id, device,
                           label_smoothing=t_cfg.label_smoothing)

        logger.info(
            f"[{split_name.upper()}] Loss: {metrics['loss']:.4f}, "
            f"PPL: {metrics['perplexity']:.2f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"KL: {metrics['kl_divergence']:.4f}, "
            f"OA: {metrics['overlapping_area']:.4f}"
        )


if __name__ == '__main__':
    main()
