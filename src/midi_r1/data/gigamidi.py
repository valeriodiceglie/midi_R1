import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
from symusic import Score
from midi_r1.config import Config

class GigaMIDIIterable(IterableDataset):
    def __init__(self, cfg: Config, tokenizer):
        #self.cfg = cfg
        self.seq_len = cfg.data.seq_len
        self.tokenizer = tokenizer
        self.raw = load_dataset(path=cfg.data.path, split=cfg.data.split)

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