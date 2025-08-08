from torch.amp import autocast
from torchmetrics.functional import accuracy
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch

def train_one_epoch(
    model,
    dataloader: DataLoader,
    criterion,
    optimizer,
    scheduler,
    grad_accum_steps: int,
    max_grad_norm: float,
    scaler,
    device: torch.device,
    pad_token_id,
    logger,
    accuracy,
    log_interval: int = 10,
):
    
    model.train()
    optimizer.zero_grad()
    
    total_loss = 0.0
    step_loss    = 0.0

    for step, batch in enumerate(tqdm(dataloader, desc="Training"), start=1):
        x = batch['input_ids'].transpose(0, 1).to(device)
        pad_mask = (x.transpose(0, 1) == pad_token_id)
        input_ids, labels = x[:-1], x[1:]
        pad_mask = pad_mask[:,:-1]
        
        with autocast(device_type="cuda"):
            logits = model(input_ids, src_key_padding_mask=pad_mask)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1)
            )
            loss = loss / grad_accum_steps
        
        scaler.scale(loss).backward()
        preds = logits.argmax(dim=-1)
        accuracy.update(preds.flatten(), labels.flatten())
        step_loss    += loss.item() * grad_accum_steps
        total_loss   += step_loss

        # gradient‐accumulation step
        if step % grad_accum_steps == 0 or step == len(dataloader):
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            if step % log_interval == 0:
                tqdm.write(f"[GradNorm] step {step}: {grad_norm:.4f}")
                logger.debug(f"Grad norm at step {step}: {grad_norm:.4f}")
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if step % log_interval == 0:
            avg_loss = step_loss / log_interval
            avg_acc = accuracy.compute().item()
            logger.info(
                f"Train step {step}/{len(dataloader)} --- "
                f"Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}"
            )
            step_loss    = 0.0

    # epoch metrics
    epoch_loss = total_loss / len(dataloader)
    epoch_acc  = accuracy.compute().item()

    return epoch_loss, epoch_acc

def validate_one_epoch(
    model,
    dataloader: DataLoader,
    criterion,
    device: torch.device,
    pad_token_id: int,
    logger,
    accuracy_metric,
    log_interval: int = 10,
):
    model.eval()
    total_loss = 0.0
    # reset metric at start of epoch
    accuracy_metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Validation"), start=1):
            x = batch['input_ids'].transpose(0, 1).to(device)
            pad_mask = (x.transpose(0, 1) == pad_token_id)
            input_ids, labels = x[:-1], x[1:]
            pad_mask = pad_mask[:,:-1]

            with autocast(device_type="cuda"):
                logits = model(input_ids, src_key_padding_mask=pad_mask)
                loss = criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.reshape(-1)
                )

            step_loss = loss.item()
            total_loss += step_loss
            preds = logits.argmax(dim=-1)
            accuracy_metric.update(preds.flatten(), labels.flatten())

            if step % log_interval == 0:
                avg_loss = step_loss / log_interval
                avg_acc = accuracy_metric.compute().item()
                logger.info(
                    f" Val Step {step}/{len(dataloader)} - "
                    f"Loss: {avg_loss:.4f}  Acc: {avg_acc:.4f}"
                )
                step_loss = 0.0

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = accuracy_metric.compute().item()
    return epoch_loss, epoch_acc

