from pathlib import Path
from tqdm.auto import tqdm
from symusic import Score
from datasets import load_dataset
from miditok import REMI

OUTPUT_DIR = Path("giga_midi_guitars")
OUTPUT_DIR.mkdir(exist_ok=True)
GUITAR_PROGRAMS = set(range(24, 32))  # MIDI program numbers for guitars

ds = load_dataset("Metacreation/GigaMIDI", "v1.1.0", split="train")

for example in tqdm(ds, desc="Scanning MIDIs"):
    # get raw bytes and an ID for naming
    midi_bytes = example["music"]            
    fname = Path(example["md5"])
    song_id = fname.stem

    # parse midi
    try:
        score = Score.from_midi(midi_bytes)
    except Exception as e:
        tqdm.write(f"Skipping {song_id}: {e}")
        continue
    guitar_tracks = [t for t in score.tracks if t.program in GUITAR_PROGRAMS]        
    for i, t in enumerate(guitar_tracks):    
        new_score = Score()
        new_score.tracks.append(t)
        out_path = OUTPUT_DIR / f"{song_id}_guitar_{i}.mid"
        new_score.dump_midi(str(out_path))
        