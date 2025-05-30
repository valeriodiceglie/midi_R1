import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from midi_r1.config import Config
from tqdm import trange, tqdm

class Trainer:
    def __init__(self, model, dataloader: DataLoader, cfg: Config):
        self.model = model
        self.dl = dataloader
        self.cfg = cfg
        self.device = cfg.device
        self.epochs = cfg.train.epochs
        self.log_interval = cfg.train.log_interval
        self.output_dir = cfg.output_dir
        self.mixed_precision = cfg.train.mixed_precision
        self.optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr)
        self.scaler = GradScaler(device=self.device, enabled=cfg.train.mixed_precision)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def train(self):
        self.model.train()
        global_step = 0
        for epoch in trange(self.epochs):
            for _, (inputs, labels, attention_mask) in enumerate(tqdm(self.dl, leave=False, unit="batch")):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                attention_mask  = attention_mask.to(self.device)
                self.optimizer.zero_grad()
                with autocast(device_type=self.device, enabled=self.mixed_precision):
                    logits = self.model(inputs, attention_mask=attention_mask)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if global_step % self.log_interval == 0:
                    tqdm.write(f"Epoch {epoch} Step {global_step} Loss: {loss.item():.4f}")
                global_step += 1
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int):
        os.makedirs(self.output_dir, exist_ok=True)
        ckpt_path = os.path.join(self.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler_state': self.scaler.state_dict(),
        }, ckpt_path)