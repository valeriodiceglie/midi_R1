import torch
from dataclasses import dataclass, field
from typing import Tuple
from miditok import REMI, TokenizerConfig
from miditok.constants import CHORD_MAPS

@dataclass
class DataConfig:
    path: str = "Metacreation/GigaMIDI"
    split: str = "train"
    seq_len: int = 256
    batch_size: int = 8
    pitch_range: Tuple[int, int] = (21, 109)
    beat_res: dict[Tuple[int, int], int] = field(default_factory=lambda: {(0,4):8, (4,12):4})
    num_velocities: int = 32
    special_tokens: list[str] = field(default_factory=lambda: ["PAD","BOS","EOS"])
    encode_ids_split: str = "bar"
    use_velocities: bool = True
    use_note_duration_programs: list[int] = field(default_factory=lambda: list(range(-1, 128)))
    use_chords: bool = False
    use_rests: bool = False
    use_tempos: bool = False
    use_time_signatures: bool = False
    use_sustain_pedals: bool = False
    use_pitch_bends: bool = False
    use_pitch_intervals: bool = False
    use_programs: bool = False
    use_pitchdrum_tokens: bool = True
    default_note_duration: float = 0.5
    beat_res_rest: dict[tuple[int,int], int] = field(default_factory=lambda: {(0, 1): 8, (1, 2): 4, (2, 12): 2})
    chord_maps: dict = field(default_factory=lambda: CHORD_MAPS)
    chord_tokens_with_root_note: bool = False
    chord_unknown: tuple[int,int] | None = None
    num_tempos: int = 32
    tempo_range: tuple[int,int] = (40, 250)
    log_tempos: bool = False
    remove_duplicated_notes: bool = False
    delete_equal_successive_tempo_changes: bool = False
    time_signature_range: dict[int, list[int] | tuple[int,int]] = field(default_factory=lambda: {8: [3, 12, 6], 4: [5, 6, 3, 2, 1, 4]})
    sustain_pedal_duration: bool = False
    pitch_bend_range: tuple[int,int,int] = (-8192, 8191, 32)
    delete_equal_successive_time_sig_changes: bool = False
    programs: list[int] = field(default_factory=lambda: list(range(-1, 128)))
    one_token_stream_for_programs: bool = True
    program_changes: bool = False
    max_pitch_interval: int = 16
    pitch_intervals_max_time_dist: float = 1.0
    drums_pitch_range: tuple[int,int] = (27, 88)
    ac_polyphony_track: bool = False
    ac_polyphony_bar: bool = False
    ac_polyphony_min: int = 1
    ac_polyphony_max: int = 6
    ac_pitch_class_bar: bool = False
    ac_note_density_track: bool = False
    ac_note_density_track_min: int = 0
    ac_note_density_track_max: int = 18
    ac_note_density_bar: bool = False
    ac_note_density_bar_max: int = 18
    ac_note_duration_bar: bool = False
    ac_note_duration_track: bool = False
    ac_repetition_track: bool = False
    ac_repetition_track_num_bins: int = 10
    ac_repetition_track_num_consec_bars: int = 4


@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 6
    num_heads: int = 4
    ffn_dim: int = 2048
    num_latents: int = 64
    latent_dim: int = 256
    latent_heads: int = 4


@dataclass
class TrainConfig:
    lr: float = 1e-4
    epochs: int = 1
    log_interval: int = 50
    mixed_precision: bool = True
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs"

#cs = ConfigStore.instance()
#cs.store(name="config", node=Config)


def init_tokenizer(cfg: Config) -> REMI:
    tok_cfg = TokenizerConfig(
        pitch_range=tuple(cfg.data.pitch_range),
        beat_res=cfg.data.beat_res,
        num_velocities=cfg.data.num_velocities,
        special_tokens=cfg.data.special_tokens
    )
    tokenizer = REMI(tokenizer_config=tok_cfg, max_bar_embedding=None)
    return tokenizer