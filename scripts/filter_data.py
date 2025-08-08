from pathlib import Path
from tqdm.auto import tqdm
from symusic import Score
from datasets import load_dataset, DatasetDict, Dataset, IterableDataset, IterableDatasetDict
from typing import List, Union, Iterable, Mapping


OUTPUT_DIR = Path("C:/Users/Proprietario/repo/giga_midi_guitars_fine_tuned")
OUTPUT_DIR.mkdir(exist_ok=True)
GUITAR_PROGRAMS = set(range(24, 32))  # MIDI program numbers for guitars



def filter_by_guitar(ds):
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

def filter_by_author(ds, authors: Union[str, Iterable[str]]) -> (DatasetDict | Dataset |
                                                            IterableDatasetDict | IterableDataset):
    """
    Select the dataset entries given the list of authors
    """
    # normalize input authors to a set of lowercase stripped strings
    if isinstance(authors, str):
        author_set = {authors.strip().lower()}
    else:
        author_set = {a.strip().lower() for a in authors if isinstance(a, str) and a.strip()}
    if not author_set:
        return ds  # nothing to filter by
    def predicate(example: Mapping):
        # grab artist/author fields safely
        artist_field = example.get("artist") or example.get("author") or ""
        if artist_field is None:
            artist_field = ""
        # normalize to a string (could be list/tuple too)
        if isinstance(artist_field, (list, tuple, set)):
            names = [str(a).strip().lower() for a in artist_field if a]
            return bool(author_set & set(names))
        else:
            artist_str = str(artist_field).lower()
            # substring match; change to == if you want exact equality
            return any(a in artist_str for a in author_set)
    return ds.filter(predicate)


def main():
    ds = load_dataset("Metacreation/GigaMIDI", "v1.1.0", split="train")
    authors = "Allan Holdsworth"
    ds = filter_by_author(ds, authors)
    filter_by_guitar(ds)


if __name__ == "__main__":
    main()