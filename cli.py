import torch
from miditok import MIDILike, REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from src.midiR1.model.model import MidiR1
from src.midiR1.engine.trainer import train
from src.midiR1.engine.checkpointing import load_checkpoint
from src.midiR1.engine.inference import MidiGenerator, GenerationConfig
import random
import yaml
from src.midiR1.utils.redirect import _silence_worker
from copy import deepcopy

mode = "train"

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
    #tokenizer = MIDILike.from_pretrained(tok_path, local_files_only=True)
    tokenizer = REMI.from_pretrained(tok_path, local_files_only=True)
    model_config["vocab_size"] = tokenizer.vocab_size

    model = MidiR1(model_config)

    if mode == "train":
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


        print(f"Number of batches in train_loader: {len(train_dataloader)}")
        print(f"Number of batches in val_loader: {len(val_dataloader)}")
        print(f"Number of batches in test_loader: {len(test_dataloader)}")
        model.to(device)


        train(model, None, train_dataloader, val_dataloader, test_dataloader,
            {**train_config})

    elif mode == "inference":
        config = GenerationConfig(
            max_length= 128,
            temperature= 0.8,
            top_k = 50,
            top_p = 0.9,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 3,
            do_sample = True,
            use_mtp = False,
            mtp_speculation_mode = False,
            num_beams = 1,
            length_penalty = 1.5,
            early_stopping = False,
            pad_token_id=tokenizer.pad_token_id
        )

        load_checkpoint("checkpoints/best_model.pt", model, load_tokenizer=False)
        generator = MidiGenerator(model, tokenizer)

        prompt_seqs = []
        midi_paths = sorted(Path(train_config["gen"]).glob("*.mid"))

        for p in midi_paths:
            seqs = tokenizer(p)
            prompt_seqs.append(seqs)



        continuations = generator.generate(prompt_seqs, config)
        for idx, (in_seq, full_seq) in enumerate(zip(prompt_seqs, continuations)):
            # split out continuation
            gen_ids = full_seq.ids[len(in_seq[0].ids):]
            tokens_list = [gen_ids, in_seq[0].ids, full_seq.ids]

            # decode MIDI and append three tracks
            midi_main = tokenizer.decode([deepcopy(gen_ids)])
            for tok_ids in tokens_list[1:]:
                track_midi = tokenizer.decode([deepcopy(tok_ids)])
                midi_main.tracks.append(track_midi.tracks[0])

            midi_main.tracks[0].name = f"Continuation ({len(gen_ids)} tokens)"
            midi_main.tracks[1].name = f"Prompt ({len(in_seq[0].ids)} tokens)"
            midi_main.tracks[2].name = "Prompt + Continuation"

            midi_main.dump_midi(Path(train_config["generate_dir"]) / f"{idx}.mid")
            tokenizer.save_tokens(tokens_list, Path(train_config["generate_dir"]) / f"{idx}.json")

        print(f"Saved {len(continuations)} generations to {train_config["generate_dir"]}")


        # continuations = generator.generate(prompt_seqs, config)
        #
        # out_dir = Path(train_config["generate_dir"])
        # out_dir.mkdir(exist_ok=True, parents=True)
        #
        # for idx, (in_seq, full_seq) in enumerate(zip(prompt_seqs, continuations)):
        #     # only take the newly generated tokens
        #     prompt_len = len(in_seq[0].ids)
        #     gen_ids = full_seq.ids[prompt_len:]
        #
        #     # decode to a single-track MIDI
        #     midi_cont = tokenizer.decode([deepcopy(gen_ids)])
        #     #midi_cont.tracks[0].name = f"Generated continuation ({len(gen_ids)} tokens)"
        #
        #     # save the MIDI
        #     midi_cont.dump_midi(out_dir / f"continuation_{idx}.mid")
        #
        #     # if you still want to persist tokens:
        #     tokenizer.save_tokens([gen_ids], out_dir / f"continuation_{idx}.json")
        #
        # print(f"Saved {len(continuations)} generated-only MIDIs to {out_dir}")


if __name__ == "__main__":
    main()
