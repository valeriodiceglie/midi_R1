import os
import torch
import logging
from torch import nn, optim
from miditok import MIDILike
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from midiR1.transformer_decoder.trainer import train_one_epoch
from midiR1.transformer_decoder.MidiTransformer import MidiTransformer
from tqdm.auto import tqdm
from torchmetrics import Accuracy
from torch.amp import GradScaler
from transformers import get_cosine_schedule_with_warmup

from src.midiR1.transformer_decoder.trainer import validate_one_epoch

logging.basicConfig(
    filename='../logs/training.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

CONFIG = {
    "tokenizer_config_path": "tokenizerconfig20k.json",
    "train": "C:/Users/Proprietario/repo/midi_data/chunks_midi_train",
    "val": "C:/Users/Proprietario/repo/midi_data/chunks_midi_val",
    "seq_len": 128,
    "batch_size": 256,
    "epochs": 500,
    "lr": 5e-4,
    "dropout": 0.2,
    "weight_decay": 1e-2,
    "grad_accum_steps": 4,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.05,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "C:/Users/Proprietario/repo/midiR1/checkpoints"
}

def main():
    if not torch.cuda.is_available():
        exit()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = Path('./tokenizerconfig20k.json')
    tokenizer = MIDILike.from_pretrained(p, local_files_only=True)

    train_accuracy = Accuracy(
        task='multiclass',
        num_classes=tokenizer.vocab_size,
        ignore_index=tokenizer.pad_token_id,
    ).to(CONFIG['device'])
    val_accuracy = Accuracy(
        task='multiclass',
        num_classes=tokenizer.vocab_size,
        ignore_index=tokenizer.pad_token_id,
    ).to(CONFIG['device'])

    train_dataset = DatasetMIDI(
        files_paths=list(Path(CONFIG['train']).resolve().glob("*.mid")),
        tokenizer=tokenizer,
        max_seq_len=128,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
        pre_tokenize=False
    )
    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=collator,
        shuffle=True,
        num_workers=12)

    val_dataset = DatasetMIDI(
        files_paths=list(Path(CONFIG['val']).glob("*.mid")),
        tokenizer=tokenizer,
        max_seq_len=128,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer["BOS_None"],
        pre_tokenize=False
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        collate_fn=collator,
        shuffle=True,
        num_workers=12
    )

    model = MidiTransformer(vocab_size=tokenizer.vocab_size, dropout=CONFIG['dropout']).to(CONFIG['device'])

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    #optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    optimizer = optim.AdamW(model.parameters(), lr = CONFIG['lr'])

    total_steps = (len(train_dataloader) // CONFIG['grad_accum_steps']) * CONFIG['epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler()

    os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

    best_acc = 0.0
    early_stop_counter = 0

    for epoch in range(1, CONFIG['epochs'] + 1):
        loss, acc = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum_steps=CONFIG['grad_accum_steps'],
            max_grad_norm=CONFIG['max_grad_norm'],
            scaler=scaler,
            device=CONFIG['device'],
            pad_token_id=tokenizer.pad_token_id,
            logger= logger,
            accuracy=train_accuracy
        )
        tqdm.write(f"[Epoch {epoch}] TRAIN - Loss: {loss:.4f} Accuracy: {acc:.4f}")
        logger.info(f"[Epoch {epoch}] TRAIN - Loss: {loss:.4f} Accuracy: {acc:.4f}")
        ckpt_path = os.path.join(CONFIG['checkpoint_dir'], f"epoch_{epoch}.pt")
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)
        logger.info(f"Saved checkpoint {ckpt_path}")

        val_loss, val_acc = validate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=CONFIG['device'],
            pad_token_id=tokenizer.pad_token_id,
            logger=logger,
            accuracy_metric=val_accuracy
        )

        tqdm.write(f"[EPOCH {epoch}] Val - Loss: {val_loss:.4f} Accuracy: {val_acc:.4f}")
        logger.info(f"[EPOCH {epoch}] Val - Loss: {val_loss:.4f} Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                print("Early stopping: no improvement for 5 epochs.")
                break

if __name__ == '__main__':
    main()