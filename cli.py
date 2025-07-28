import logging
import torch
from miditok import MIDILike
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from src.midiR1.model.model import MidiR1
from src.midiR1.engine.trainer import train
import random
import yaml
from src.midiR1.utils.redirect import _silence_worker


def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    seed = 42
    random.seed(seed)

    train_config = load_config("config/train.yaml")
    model_config = load_config("config/model.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config["device"] = device

    tok_path = Path(train_config["tokenizer_config_path"])
    tokenizer = MIDILike.from_pretrained(tok_path, local_files_only=True)
    model_config["vocab_size"] = tokenizer.vocab_size
    train_dataset = DatasetMIDI(
        files_paths=list(Path(train_config["train"]).resolve().glob("*.mid")),
        tokenizer=tokenizer,
        max_seq_len=model_config["max_seq_len"],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        pre_tokenize=False
    )
    val_dataset = DatasetMIDI(
        files_paths=list(Path(train_config["val"]).resolve().glob("*.mid")),
        tokenizer=tokenizer,
        max_seq_len=model_config["max_seq_len"],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        pre_tokenize=False
    )
    test_dataset = DatasetMIDI(
        files_paths=list(Path(train_config["test"]).resolve().glob("*.mid")),
        tokenizer=tokenizer,
        max_seq_len=model_config["max_seq_len"],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        pre_tokenize=False
    )
    train_config['pad_token_id'] = tokenizer.pad_token_id
    collator = DataCollator(tokenizer.pad_token_id, shift_labels=True, labels_pad_idx=tokenizer.pad_token_id,
                            copy_inputs_as_labels=True)
    train_dataloader = DataLoader(train_dataset, batch_size=train_config["batch_size"],
                                  collate_fn=collator, shuffle=True, num_workers=6, persistent_workers=True,
                                  prefetch_factor=4, worker_init_fn=_silence_worker)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config["batch_size"],
                                collate_fn=collator, shuffle=False, num_workers=6, persistent_workers=True,
                                prefetch_factor=4, worker_init_fn=_silence_worker)
    test_dataloader = DataLoader(test_dataset, batch_size=train_config["batch_size"],
                                 collate_fn=collator, shuffle=False, num_workers=6, worker_init_fn=_silence_worker)
    model = MidiR1(model_config)

    print(f"Number of batches in train_loader: {len(train_dataloader)}")
    print(f"Number of batches in val_loader: {len(val_dataloader)}")
    print(f"Number of batches in test_loader: {len(test_dataloader)}")
    model.to(device)

    train(model, train_dataloader, val_dataloader, test_dataloader, {**train_config})

    # else:  # generate
    #     model = MidiTransformer(
    #         n_layers=12,
    #         moe_experts=0,
    #         top_moe_k=0,
    #         vocab_size=tokenizer.vocab_size,
    #         n_heads=8,
    #         mtp_num_heads=2,
    #         hidden_dim=512,
    #         latent_dim=64,
    #         rotary_dim=32,
    #         base_seq_len=512,
    #         stage1_seq_len=1024,
    #         max_seq_len=args.max_seq_len,
    #         dropout=0.1,
    #     )
    #     ckpt_path = Path(args.checkpoint_dir) / f"model_epoch{args.epochs}.pt"
    #     model.load_state_dict(torch.load(ckpt_path, map_location=device))
    #     inference = Inference(model, tokenizer, device)
    #     inference.generate(args.generate_midi, args.output_midi)

if __name__ == "__main__":
    main()
