import numpy as np
import torch
from miditok import MIDILike, REMI
from miditok.pytorch_data import DatasetMIDI, DataCollator
from torch.utils.data import DataLoader
from pathlib import Path
from src.midiR1.engine.inference import MidiGenerator, GenerationConfig
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3Model
from transformers import Trainer, TrainingArguments, get_linear_schedule_with_warmup, EvalPrediction
import random
import yaml
from src.midiR1.utils.redirect import _silence_worker
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

mode = "train"

def _compute_metrics(eval_pred: EvalPrediction, pad_id: int) -> Dict[str, float]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    mask = labels != pad_id
    correct = (preds == labels) & mask
    accuracy = correct.sum() / mask.sum()
    return {"accuracy": float(accuracy)}

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    seed = 42
    random.seed(seed)

    train_config = load_config("config/train.yaml")
    model_config = load_config("config/model_v3_huggingfaces.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_config["device"] = device

    tok_path = Path(train_config["tokenizer_config_path"])
    #tokenizer = MIDILike.from_pretrained(tok_path, local_files_only=True)
    tokenizer = REMI.from_pretrained(tok_path, local_files_only=True)
    model_config["vocab_size"] = tokenizer.vocab_size

    config = DeepseekV3Config(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        moe_intermediate_size=model_config["moe_intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        num_key_value_heads=model_config["num_key_value_heads"],
        n_shared_experts=model_config["n_shared_experts"],
        n_routed_experts=model_config["n_routed_experts"],
        routed_scaling_factor=model_config["routed_scaling_factor"],
        kv_lora_rank=model_config["kv_lora_rank"],
        q_lora_rank=model_config["q_lora_rank"],
        qk_rope_head_dim=model_config["qk_rope_head_dim"],
        v_head_dim=model_config["v_head_dim"],
        qk_nope_head_dim=model_config["qk_nope_head_dim"],
        n_group=model_config["n_group"],
        topk_group=model_config["topk_group"],
        num_experts_per_tok=model_config["num_experts_per_tok"],
        first_k_dense_replace=model_config["first_k_dense_replace"],
        norm_topk_prob=model_config["norm_topk_prob"],
        hidden_act=model_config["hidden_act"],
        max_position_embeddings=model_config["max_position_embeddings"],
        initializer_range=model_config["initializer_range"],
        rms_norm_eps=model_config["rms_norm_eps"],
        use_cache=model_config["use_cache"],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
        pretraining_tp=model_config["pretraining_tp"],
        tie_word_embeddings=model_config["tie_word_embeddings"],
        rope_theta=model_config["rope_theta"],
        #rope_scaling=model_config["rope_scaling"],
        rope_interleave=model_config["rope_interleave"],
        attention_bias=model_config["attention_bias"],
        attention_dropout=model_config["attention_dropout"]
    )
    model = DeepseekV3Model(config)

    if mode == "train":
        train_dataset = DatasetMIDI(
            files_paths=list(Path(train_config["train"]).resolve().glob("*.mid")),
            tokenizer=tokenizer,
            max_seq_len=model_config["max_position_embeddings"],
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pre_tokenize=False
        )
        val_dataset = DatasetMIDI(
            files_paths=list(Path(train_config["val"]).resolve().glob("*.mid")),
            tokenizer=tokenizer,
            max_seq_len=model_config["max_position_embeddings"],
            bos_token_id=tokenizer["BOS_None"],
            eos_token_id=tokenizer["EOS_None"],
            pre_tokenize=False
        )
        test_dataset = DatasetMIDI(
            files_paths=list(Path(train_config["test"]).resolve().glob("*.mid")),
            tokenizer=tokenizer,
            max_seq_len=model_config["max_position_embeddings"],
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

        training_args = TrainingArguments(
            output_dir="checkpoints/deepseekv3",
            overwrite_output_dir=True,
            num_train_epochs=20,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=2,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=500,
            save_total_limit=3,
            logging_steps=100,
            learning_rate=5e-5,
            weight_decay=0.01,
            warmup_steps=1000,
            fp16=True,
            dataloader_num_workers=6,
            dataloader_persistent_workers=True,
            report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",

        )

        compute_metrics = partial(_compute_metrics, pad_id=tokenizer.pad_token_id)

        # Instantiate Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()


if __name__ == "__main__":
    main()
