import torch
from midi_r1.config import Config
from midi_r1.data.gigamidi import GigaMIDIIterable
from midi_r1.config import init_tokenizer
from midi_r1.model.foundational import R1FoundationModel
from midi_r1.engine.trainer import Trainer
from torch.utils.data import DataLoader

def test_trainer_one_step(tmp_path):
    cfg = Config()
    cfg.train.epochs = 1
    cfg.data.batch_size = 2
    cfg.output_dir = str(tmp_path)
    tokenizer = init_tokenizer(cfg)
    dataset = GigaMIDIIterable(cfg, tokenizer)
    dl = DataLoader(dataset, batch_size=2, collate_fn=GigaMIDIIterable.collate_fn)
    model = R1FoundationModel(cfg, tokenizer.vocab_size).to(cfg.device)
    trainer = Trainer(model, dl, cfg)
    # run a single epoch
    trainer.train()
    # check checkpoint exists
    ckpt = tmp_path / "checkpoint_epoch0.pt"
    assert ckpt.exists()