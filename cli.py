import argparse
import torch
from dotenv import load_dotenv
from miditok import MIDILike
from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from midiR1.model.model import R1Model
from midiR1.engine.trainer import Trainer
from midiR1.engine.inference import Inference
from midiR1.utils.logging import init_logger


def main():
    parser = argparse.ArgumentParser(description="Train or generate with R1Model")
    parser.add_argument("--mode", choices=["train", "generate"], default="train")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizerconfig20k.json")
    parser.add_argument("--midi_dir", type=str, default="giga_midi_guitars")
    parser.add_argument("--chunks_dir", type=str, default="chunks_midi")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.1)
    parser.add_argument("--lr_step", type=int, default=1)
    parser.add_argument("--lr_gamma", type=float, default=0.95)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--num_overlap_bars", type=int, default=2)
    parser.add_argument("--generate_midi", type=str, help="Input MIDI file for generation")
    parser.add_argument("--output_midi", type=str, help="Output MIDI file path")
    args = parser.parse_args()

    load_dotenv()
    init_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = Path(args.tokenizer_path)
    tokenizer = MIDILike.from_pretrained(p, local_files_only=True)

    if args.mode == "train":

        dataset = DatasetMIDI(
            files_paths=list(Path(args.chunks_dir).resolve().glob("*.mid")),
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pre_tokenize=False
        )
        collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=True, num_workers=12)

        model = R1Model(
            n_layers=6,
            moe_experts=0,
            top_moe_k=0,
            vocab_size=tokenizer.vocab_size,
            n_heads=4,
            mtp_num_heads=1,
            hidden_dim=256,
            latent_dim=32,
            rotary_dim=16,
            base_seq_len=512,
            stage1_seq_len=1024,
            max_seq_len=args.max_seq_len,
            dropout=0.1,
        )

        trainer = Trainer(
                model,
                dataloader,
                pad_token_id=tokenizer.pad_token_id,
                lr=args.lr,
                weight_decay=args.wd,
                epochs=args.epochs,
                grad_accum_steps=args.grad_accum_steps,
                max_grad_norm=1.0,
                log_every=100,
                lr_step=args.lr_step,
                lr_gamma=args.lr_gamma,
                ckpt_dir=args.checkpoint_dir
        )
        trainer.train()

    else:  # generate
        model = R1Model(
            n_layers=12,
            moe_experts=0,
            top_moe_k=0,
            vocab_size=tokenizer.vocab_size,
            n_heads=8,
            mtp_num_heads=2,
            hidden_dim=512,
            latent_dim=64,
            rotary_dim=32,
            base_seq_len=512,
            stage1_seq_len=1024,
            max_seq_len=args.max_seq_len,
            dropout=0.1,
        )
        ckpt_path = Path(args.checkpoint_dir) / f"model_epoch{args.epochs}.pt"
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        inference = Inference(model, tokenizer, device)
        inference.generate(args.generate_midi, args.output_midi)

if __name__ == "__main__":
    main()
