from dotenv import load_dotenv
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from midi_r1.config import init_tokenizer
from midi_r1.data.gigamidi import GigaMIDIIterable
from midi_r1.model.foundational import R1FoundationModel
from midi_r1.engine.trainer import Trainer
from midi_r1.utils.logging import init_logger

@hydra.main(config_path="../../configs", config_name="train")
def train(cfg: DictConfig):
    load_dotenv(dotenv_path="../.env")  
    init_logger(cfg)

    tokenizer = init_tokenizer(cfg)
    dataset = GigaMIDIIterable(cfg, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        collate_fn=GigaMIDIIterable.collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model = R1FoundationModel(cfg, tokenizer.vocab_size).to(cfg.device)
    trainer = Trainer(model, dataloader, cfg)
    trainer.train()

if __name__ == "__main__":
    train()