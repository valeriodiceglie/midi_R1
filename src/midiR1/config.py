from dataclasses import dataclass, fields as dc_fields
from typing import Any, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    num_layers: int = 3
    hidden_dim: int = 384
    num_heads: int = 8

    # MLA head dimensions
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    kv_compression_dim: int = 96
    query_compression_dim: int = 96

    # MoE configuration
    num_routed_experts: int = 2
    segmentation_factor: int = 4
    num_shared_experts: int = 1
    activated_experts: int = 2

    first_k_dense_replace: int = 1
    moe_aux_loss_alpha: float = 1e-5

    # Multi-Token Prediction
    mtp_depth: int = 2
    mtp_use_attention: bool = True
    mtp_loss_weight: float = 0.1

    dropout_rate: float = 0.1
    vocab_size: int = 20000
    min_seq_len: int = 5
    max_seq_len: int = 256
    stride: int = 32

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dc_fields(self)}


@dataclass
class TrainConfig:
    """Training configuration."""

    batch_size: int = 32
    learning_rate: float = 5e-4

    # Training settings
    save_steps: int = 1000
    eval_steps: int = 100
    eval_max_batches: int = 200
    total_steps: int = 50000

    # Optimizer
    optimizer: str = "adamw"
    momentum: float = 0.9
    nesterov: bool = True

    # Scheduler
    scheduler: str = "cosine_warmup"
    min_lr: float = 1e-4
    warmup_ratio: float = 0.05

    # Training settings
    gradient_clip: float = 0.5
    label_smoothing: float = 0.1
    gradient_accumulation_steps: int = 64
    mtp_weight: float = 0.2

    # Early stopping
    patience: int = 50
    min_delta: float = 0.001
    early_stopping_on: str = "train_and_val"

    # Hardware
    use_amp: bool = True
    use_compile: bool = False

    # Weights & Biases
    wandb_project: str = "midiR1"
    wandb_disabled: bool = False

    # Data paths
    tokenizer_config_path: str = "./tokenizer10kREMI.json"
    train: str = "C:/Users/Proprietario/repo/midi_data/giga_midi_guitars_train"
    val: str = "C:/Users/Proprietario/repo/midi_data/giga_midi_guitars_val"
    test: str = "C:/Users/Proprietario/repo/midi_data/giga_midi_guitars_test"
    token_cache_dir: str = "C:/Users/Proprietario/repo/midi_data/token_cache"
    gen: str = "C:/Users/Proprietario/repo/midi_data/64/chunks_midi_fine_tuned"
    checkpoint_dir: str = "C:/Users/Proprietario/repo/midiR1/checkpoints"
    generate_dir: str = "C:/Users/Proprietario/repo/midiR1/generate"

    # Sampling
    length_exponent: float = 0.0

    # Fine-tuning
    finetune_checkpoint: str = ""
    conditioning_tokens: int = 16          # Legacy: reserved token slots for custom conditioning
    use_attribute_controls: bool = False    # Use MIDITok attribute controls instead
    ac_tracks_ratio_range: Tuple[float, float] = (0.4, 0.9)
    ac_bars_ratio_range: Tuple[float, float] = (0.4, 0.9)
    finetune_lr_scale: float = 0.1
    finetune_freeze_layers: int = 0

    # Runtime fields (set programmatically, not serialized)
    device: Any = None
    pad_token_id: int = -100
    wandb_run: Any = None

    def to_dict(self) -> dict:
        skip = {"device", "wandb_run"}
        return {f.name: getattr(self, f.name) for f in dc_fields(self) if f.name not in skip}
