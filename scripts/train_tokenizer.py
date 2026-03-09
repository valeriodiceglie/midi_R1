import miditok.constants
from miditok import MIDILike, TokenizerConfig, REMI, constants
from pathlib import Path

guitar_ranges = [
    inst["pitch_range"]
    for inst in constants.MIDI_INSTRUMENTS
    if "Guitar" in inst["name"]
]

pitch_min = min(r.start for r in guitar_ranges)
pitch_max = max(r.start for r in guitar_ranges)

config = TokenizerConfig(
    beat_res={(0, 4): 12, (4, 12): 4},
    beat_res_rest={(0, 1): 12, (1, 2): 4, (2, 12): 2},
    encode_ids_split="bar",
    pitch_range=(pitch_min, pitch_max),
    num_velocities=32,
    use_rests=True,
    chord_unknown=(3,7),
    use_chords=True,
    use_tempos=True,
    use_time_signatures=True,
    use_pitch_bends=True,
    time_signature_range={4: [1, 2, 3, 4, 5, 6, 7, 9], 8: [3, 6, 12]},
    # Attribute controls for conditioning
    ac_note_density_bar=True,
    ac_note_density_bar_max=20,
    ac_polyphony_bar=True,
    ac_note_duration_bar=True,
    ac_pitch_class_bar=True,
    ac_note_density_track=True,
    ac_polyphony_track=True,
    ac_repetition_track=True,
)

#tokenizer = MIDILike(config)
tokenizer = REMI(config)

tokenizer.train(
    files_paths=list(Path("C:/Users/Proprietario/repo/midi_data/giga_midi_guitars_train").resolve().glob("*.mid")),
    vocab_size=20_000,
    model="BPE"
)
tokenizer.save("tokenizer20kREMI.json")
