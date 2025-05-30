from dotenv import load_dotenv
from datasets import load_dataset
import pytest
import torch
from midi_r1.config import Config, init_tokenizer
from midi_r1.data.gigamidi import GigaMIDIIterable
load_dotenv(dotenv_path="../.env") 


@pytest.fixture
def dataset():
    """
    Fixture to create a GigaMIDIIterable dataset instance using default Config and tokenizer.
    """
    cfg = Config()
    assert cfg.data.path != None
    tokenizer = init_tokenizer(cfg)
    return GigaMIDIIterable(cfg, tokenizer)


def test_iterable_returns_tensor(dataset):
    """
    Test that iterating the dataset yields a 1D torch.Tensor of token IDs.
    """
    it = iter(dataset)
    sample = next(it)
    assert isinstance(sample, torch.Tensor)
    assert sample.ndim == 1


def test_collate_fn_pads_and_masks():
    """
    Test that the collate_fn correctly pads sequences and produces inputs, labels, and attention masks.
    """
    seq1 = torch.tensor([1, 2, 3], dtype=torch.long)
    seq2 = torch.tensor([4, 5], dtype=torch.long)
    inputs, labels, mask = GigaMIDIIterable.collate_fn([seq1, seq2])

    # After padding, sequences become [[1,2,3], [4,5,0]]
    # inputs = padded[:, :-1] -> shape (2,2), labels = padded[:,1:] -> shape (2,2)
    assert inputs.shape == (2, 2)
    assert labels.shape == (2, 2)
    # Check mask is True where inputs != 0
    assert mask.shape == (2, 2)
    assert mask.tolist() == [[True, True], [True, True]]
