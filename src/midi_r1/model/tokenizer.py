from miditok import REMI, TokenizerConfig
from midi_r1.config import Config

def init_tokenizer(cfg: Config):
    tok_cfg = TokenizerConfig(
        pitch_range=cfg.data.pitch_range,
        beat_resolutions=cfg.data.beat_resolutions,
        num_velocities=cfg.data.num_velocities,
        special_tokens=cfg.data.special_tokens
    )
    tokenizer = REMI(tokenizer_config=tok_cfg, max_bar_embedding=None)
    return tokenizer