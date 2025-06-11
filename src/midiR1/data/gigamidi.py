import torch
from pathlib import Path
from torch.utils.data import IterableDataset
from datasets import load_dataset
from miditok import MusicTokenizer
from symusic import Score

class GigaMIDIIterable(IterableDataset):
    def __init__(self, seq_len:int, path:str, split:str, dump_folder: str):
        self.seq_len = seq_len
        self.raw = load_dataset(path, split=split)
        self.dump_folder = Path(dump_folder)
        self.dump_folder.mkdir(parents=True, exist_ok=True)

    def __iter__(self):
        for ex in self.raw:
            score = Score.from_midi(ex['music'])
            tokens = self.tokenizer(score)
            if not tokens:
                continue
            token_list = tokens[0].tokens
            ids = self.tokenizer._tokens_to_ids(token_list)
            if len(ids) > self.seq_len:
                continue
            yield torch.tensor(ids, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0)
        inputs = padded[:, :-1]
        labels = padded[:, 1:]
        attention_mask = inputs.ne(0)
        return inputs, labels, attention_mask