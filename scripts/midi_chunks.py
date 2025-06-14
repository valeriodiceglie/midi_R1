from miditok.utils import split_files_for_training
from pathlib import Path
from miditok import MIDILike

p = Path("tokenizerconfig20k.json")
tokenizer = MIDILike.from_pretrained(p, local_files_only=True)

save_dir = Path("chunks_midi")
save_dir.mkdir(exist_ok=True)
split_files_for_training(
    files_paths=list(Path("giga_midi_guitars").resolve().glob("*.mid")),
    tokenizer=tokenizer,
    save_dir=save_dir,
    max_seq_len=1024,
    num_overlap_bars=2,
)