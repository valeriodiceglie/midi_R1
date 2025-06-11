
from miditok import MIDILike, TokenizerConfig
from pathlib import Path
from midiR1.utils.logging import init_logger

init_logger()

config = TokenizerConfig(
    beat_res={(0, 4): 8, (4, 12): 4},
    beat_res_rest={(0, 1): 8, (1, 2): 4, (2, 12): 2},  
    encode_ids_split="bar",                    
    pitch_range=(40, 88),
    num_velocities=32,   
    use_rests=True,      
    use_chords=True,     
    use_tempos=True,     
    use_time_signatures=True,
    use_pitch_bends=True,    
    num_tempos=32,                             
    tempo_range=(40, 250),
    log_tempos=False,
    time_signature_range={4: [1, 2, 3, 4, 5, 6], 8: [3, 6, 12]},  
    use_programs=False,                        
    one_token_stream_for_programs=True,        
    special_tokens=["PAD", "BOS", "EOS", "MASK"]
)

tokenizer = MIDILike(config)

tokenizer.train(
    files_paths=list(Path("giga_midi_guitars").resolve().glob("*.mid")),
    vocab_size=20_000,
    model="BPE"    
)
tokenizer.save("tokenizer_config_1024.json")