# trainer.py
import os
import math
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from transformers.optimization import get_linear_schedule_with_warmup
import torch.nn as nn
from midiR1.model.model import R1Model
from tqdm import trange, tqdm
torch.autograd.set_detect_anomaly(True)


class Trainer:
    def __init__(
        self,
        model: R1Model,
        dataloader: DataLoader,
        pad_token_id: int,
        lr: float,
        weight_decay: float,
        epochs: int,
        grad_accum_steps: int,
        max_grad_norm: float,
        log_every: int,
        lr_step: float,
        lr_gamma: float,
        ckpt_dir: str,
        compile_model: bool = False
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)

        self.dataloader = dataloader
        self.epochs = epochs
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.ckpt_dir = ckpt_dir

        # optimizer & scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        # Params for linear_scheduler
        #num_training_steps = len(dataloader) * epochs // grad_accum_steps
        #warmup_steps = math.ceil(total_steps * warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_step,
            gamma=lr_gamma            
        )

        # loss & AMP
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
        self.scaler = GradScaler()

    def train(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        for epoch in trange(1, self.epochs + 1, desc="Epoch"):
            self.model.train()
            running_loss = 0.0
            for step, batch in enumerate(tqdm(self.dataloader, desc="Batch"), 1):
                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                with autocast(device_type=self.device):
                    # model should return averaged loss over MTP heads when labels supplied
                    loss = self.model(input_ids, labels=labels)

                    # normalize for grad accumulation
                    loss = loss / self.grad_accum_steps

                self.scaler.scale(loss).backward()

                if step % self.grad_accum_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                running_loss += loss.item() * self.grad_accum_steps  # un-scaled for logging

                if step % self.log_every == 0:
                    avg = running_loss / self.log_every
                    tqdm.write(f"[Epoch {epoch}] Step {step}/{len(self.dataloader)} — Loss: {avg:.4f}")
                    running_loss = 0.0

            # checkpoint
            ckpt = os.path.join(self.ckpt_dir, f"r1_epoch{epoch}.pt")
            torch.save(self.model.state_dict(), ckpt)
            print(f"Saved checkpoint {ckpt}")
