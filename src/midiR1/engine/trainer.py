import math
import torch
from torch.optim import SGD, AdamW
from torch import nn
from tqdm import tqdm
import time
import os
import logging
from src.midiR1.config import TrainConfig
from src.midiR1.engine.checkpointing import save_checkpoint, load_checkpoint
from transformers import get_cosine_schedule_with_warmup

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TRAINER")


def _should_reset_patience(early_stopping_on, train_improved, val_improved):
    if early_stopping_on == "train":
        return train_improved
    if early_stopping_on in ("train_and_val", "both", "train_or_val", "either"):
        return train_improved or val_improved
    return val_improved


def perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss, clamped to avoid overflow."""
    return math.exp(min(loss, 100.0))


def compute_accuracy(logits, target_ids, pad_token_id):
    """Compute token-level prediction accuracy."""
    active_mask = target_ids != pad_token_id
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == target_ids) & active_mask
    total = active_mask.sum().item()
    if total > 0:
        return correct.sum().item() / total
    return 0.0


def _token_level_kl_oa(pred_dist: torch.Tensor, target_dist: torch.Tensor):
    """Compute KL divergence and overlapping area between two token distributions.

    Both tensors are 1-D probability distributions over the vocabulary.
    Returns ``(kl, oa)`` as Python floats.
    """
    eps = 1e-10
    p = target_dist + eps
    q = pred_dist + eps
    p = p / p.sum()
    q = q / q.sum()
    kl = float((p * torch.log(p / q)).sum().item())
    oa = float(torch.minimum(p, q).sum().item())
    return kl, oa


def validate(model, data_loader, criterion, pad_token_id, device, max_batches=None,
             mtp_weight=0.0):
    """Run validation on the provided data loader.

    Returns ``(avg_loss, avg_accuracy, avg_mtp_loss, kl_divergence, overlapping_area)``.
    """
    model.eval()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    epoch_mtp_loss = 0.0
    total_batches = len(data_loader)

    num_batches = min(max_batches, total_batches) if max_batches is not None else total_batches

    # Accumulators for token-level distribution metrics
    vocab_size = None
    pred_dist_acc = None
    target_freq_acc = None
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Validation",
                                       leave=False, total=num_batches)):
            if i >= num_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_ids = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask,
                            target_ids=target_ids)

            mtp_loss_val = 0.0
            if isinstance(outputs, tuple):
                logits, mtp_logits_list, _aux = outputs

                if mtp_logits_list:
                    mtp_loss = torch.tensor(0.0, device=device)
                    for k, mtp_logits_k in enumerate(mtp_logits_list):
                        mtp_target = target_ids[:, k + 2:k + 2 + mtp_logits_k.size(1)]
                        if mtp_target.numel() > 0:
                            depth_loss = criterion(
                                mtp_logits_k[:, :mtp_target.size(1)].contiguous().view(-1, mtp_logits_k.size(-1)),
                                mtp_target.contiguous().view(-1),
                            )
                            mtp_loss = mtp_loss + depth_loss
                    mtp_loss_val = (mtp_loss / len(mtp_logits_list)).item()
            else:
                logits = outputs

            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            epoch_loss += loss.item()
            epoch_mtp_loss += mtp_loss_val

            accuracy = compute_accuracy(logits, target_ids, pad_token_id)
            epoch_accuracy += accuracy

            # Accumulate token distributions for KL / OA
            active_mask = (target_ids != pad_token_id)
            if active_mask.any():
                if vocab_size is None:
                    vocab_size = logits.size(-1)
                    pred_dist_acc = torch.zeros(vocab_size, device=device, dtype=torch.float64)
                    target_freq_acc = torch.zeros(vocab_size, device=device, dtype=torch.float64)

                flat_logits = logits.view(-1, vocab_size)
                flat_mask = active_mask.view(-1)
                active_logits = flat_logits[flat_mask]
                probs = torch.softmax(active_logits.float(), dim=-1)
                pred_dist_acc += probs.sum(dim=0).double()

                active_targets = target_ids.view(-1)[flat_mask]
                target_freq_acc += torch.bincount(
                    active_targets, minlength=vocab_size
                ).double()
                total_tokens += int(flat_mask.sum().item())

    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches
    avg_mtp_loss = epoch_mtp_loss / num_batches

    # Compute token-level KL divergence and overlapping area
    val_kl = 0.0
    val_oa = 0.0
    if total_tokens > 0 and pred_dist_acc is not None:
        pred_dist_norm = pred_dist_acc / total_tokens
        target_dist_norm = target_freq_acc / total_tokens
        val_kl, val_oa = _token_level_kl_oa(pred_dist_norm, target_dist_norm)

    model.train()
    return avg_loss, avg_accuracy, avg_mtp_loss, val_kl, val_oa


def get_optimizer(model, config: TrainConfig):
    """Create the optimizer based on configuration."""
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_params = [
        {
            'params': [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.0001
        },
        {
            'params': [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    if config.optimizer == 'sgd':
        return SGD(
            optimizer_grouped_params,
            lr=config.learning_rate,
            momentum=config.momentum,
            nesterov=config.nesterov,
        )
    elif config.optimizer == 'adamw':
        return AdamW(
            optimizer_grouped_params,
            lr=config.learning_rate,
            eps=1e-8,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def get_scheduler(optimizer, total_steps, config: TrainConfig):
    """Create the learning rate scheduler."""
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )


def _forward_backward_step(model, batch, criterion, device, use_amp, amp_dtype,
                           gradient_accumulation_steps, mtp_weight):
    """Run forward and backward pass for a single batch."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_ids = batch["labels"].to(device)

    with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
        outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)

        if isinstance(outputs, tuple):
            logits, mtp_logits_list, aux_loss = outputs

            main_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            mtp_loss = torch.tensor(0.0, device=device)
            if mtp_logits_list:
                for k, mtp_logits_k in enumerate(mtp_logits_list):
                    mtp_target = target_ids[:, k + 2:k + 2 + mtp_logits_k.size(1)]
                    if mtp_target.numel() > 0:
                        depth_loss = criterion(
                            mtp_logits_k[:, :mtp_target.size(1)].contiguous().view(-1, mtp_logits_k.size(-1)),
                            mtp_target.contiguous().view(-1)
                        )
                        mtp_loss = mtp_loss + depth_loss
                mtp_loss = mtp_loss / len(mtp_logits_list)

            total_loss = (1.0 - mtp_weight) * main_loss + mtp_weight * mtp_loss + aux_loss
            mtp_loss_val = mtp_loss.item()
        else:
            logits = outputs
            total_loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            mtp_loss_val = 0.0

    scaled_loss = total_loss / gradient_accumulation_steps if gradient_accumulation_steps > 1 else total_loss
    scaled_loss.backward()

    return total_loss.item(), mtp_loss_val, logits, target_ids


def train(model, ckpt_path, train_loader, val_loader, test_loader, config: TrainConfig):
    """
    Step-based training loop with MTP, MoE aux loss, BF16 AMP, and torch.compile support.
    """
    device = config.device
    checkpoint_dir = config.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)

    if config.use_compile:
        logger.info("Compiling model with torch.compile (mode=reduce-overhead)...")
        model = torch.compile(model, mode="reduce-overhead")

    gradient_accumulation_steps = config.gradient_accumulation_steps
    pad_token_id = config.pad_token_id
    mtp_weight = config.mtp_weight
    patience = config.patience
    min_delta = config.min_delta
    wandb_run = config.wandb_run

    total_steps = config.total_steps
    eval_interval = config.eval_steps
    save_interval = config.save_steps
    eval_max_batches = config.eval_max_batches

    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, total_steps, config)

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_token_id,
        label_smoothing=config.label_smoothing,
    )

    use_amp = config.use_amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16

    # Metrics tracking
    train_losses = []
    train_accuracies = []
    train_perplexities = []
    val_losses = []
    val_accuracies = []
    val_perplexities = []

    # Early stopping (based on perplexity — lower is better)
    best_val_ppl = float('inf')
    best_val_step = 0
    best_train_ppl = float('inf')
    patience_counter = 0
    stop_reason = None
    early_stopping_on = config.early_stopping_on

    global_step = 0

    if ckpt_path is not None:
        ckpt, _ = load_checkpoint(ckpt_path, model, optimizer, scheduler, load_tokenizer=False)
        global_step = ckpt.get('global_step', 0)

    start_time = time.time()

    logger.info(f"Step-mode training: eval_steps={eval_interval}, "
                f"save_steps={save_interval}, patience={patience}, "
                f"min_delta={min_delta}, eval_max_batches={eval_max_batches}, "
                f"gradient_accumulation_steps={gradient_accumulation_steps}")

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0
    accumulated_mtp_loss = 0.0
    interval_train_loss = 0.0
    interval_train_mtp_loss = 0.0
    interval_train_accuracy = 0.0
    interval_optimizer_steps = 0
    micro_step = 0  # counts individual batches within an accumulation window
    total_batches = len(train_loader)
    data_pass = 0

    while True:
        data_pass += 1
        progress_bar = tqdm(total=total_batches, desc=f"Pass {data_pass}",
                            unit="batch", leave=True)

        for batch in train_loader:
            loss_val, mtp_loss_val, logits, target_ids = _forward_backward_step(
                model, batch, criterion, device, use_amp, amp_dtype,
                gradient_accumulation_steps, mtp_weight
            )
            accumulated_loss += loss_val
            accumulated_mtp_loss += mtp_loss_val
            micro_step += 1
            progress_bar.update(1)

            # Optimizer step
            if micro_step % gradient_accumulation_steps == 0:
                accuracy = compute_accuracy(logits, target_ids, pad_token_id)
                step_loss = accumulated_loss / gradient_accumulation_steps
                step_mtp_loss = accumulated_mtp_loss / gradient_accumulation_steps
                interval_train_accuracy += accuracy
                interval_train_loss += step_loss
                interval_train_mtp_loss += step_mtp_loss
                interval_optimizer_steps += 1

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.gradient_clip
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                step_ppl = perplexity(step_loss)
                progress_bar.set_postfix({
                    'step': global_step,
                    'loss': f'{step_loss:.4f}',
                    'ppl': f'{step_ppl:.2f}',
                    'acc': f'{accuracy:.4f}',
                    'lr': f'{current_lr:.2e}',
                })

                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss": step_loss,
                        "train/mtp_loss": step_mtp_loss,
                        "train/perplexity": step_ppl,
                        "train/accuracy": accuracy,
                        "train/learning_rate": current_lr,
                    }, step=global_step)

                accumulated_loss = 0.0
                accumulated_mtp_loss = 0.0

                # Validation
                if global_step % eval_interval == 0:
                    avg_interval_loss = interval_train_loss / interval_optimizer_steps if interval_optimizer_steps > 0 else 0
                    avg_interval_acc = interval_train_accuracy / interval_optimizer_steps if interval_optimizer_steps > 0 else 0
                    avg_interval_mtp = interval_train_mtp_loss / interval_optimizer_steps if interval_optimizer_steps > 0 else 0
                    train_ppl = perplexity(avg_interval_loss)
                    train_losses.append(avg_interval_loss)
                    train_accuracies.append(avg_interval_acc)
                    train_perplexities.append(train_ppl)

                    logger.info(
                        f"Step {global_step} - "
                        f"Train Loss: {avg_interval_loss:.4f}, Train PPL: {train_ppl:.2f}, "
                        f"Train Acc: {avg_interval_acc:.4f}, Train MTP Loss: {avg_interval_mtp:.4f}"
                    )

                    val_loss, val_accuracy, val_mtp_loss, val_kl, val_oa = validate(
                        model, val_loader, criterion, pad_token_id, device,
                        max_batches=eval_max_batches, mtp_weight=mtp_weight,
                    )
                    val_ppl = perplexity(val_loss)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    val_perplexities.append(val_ppl)
                    logger.info(
                        f"Validation - Loss: {val_loss:.4f}, PPL: {val_ppl:.2f}, "
                        f"Accuracy: {val_accuracy:.4f}, MTP Loss: {val_mtp_loss:.4f}, "
                        f"KL: {val_kl:.4f}, OA: {val_oa:.4f}"
                    )

                    if wandb_run is not None:
                        wandb_run.log({
                            "val/loss": val_loss,
                            "val/mtp_loss": val_mtp_loss,
                            "val/perplexity": val_ppl,
                            "val/accuracy": val_accuracy,
                            "val/kl_divergence": val_kl,
                            "val/overlapping_area": val_oa,
                            "val/best_perplexity": best_val_ppl,
                            "interval/train_loss": avg_interval_loss,
                            "interval/train_mtp_loss": avg_interval_mtp,
                            "interval/train_perplexity": train_ppl,
                            "interval/train_accuracy": avg_interval_acc,
                        }, step=global_step)

                    # Best model + early stopping (perplexity — lower is better)
                    train_improved = train_ppl < best_train_ppl - min_delta
                    if train_improved:
                        best_train_ppl = train_ppl

                    val_improved = val_ppl < best_val_ppl - min_delta
                    if val_improved:
                        best_val_ppl = val_ppl
                        best_val_step = global_step
                        save_checkpoint(
                            model, optimizer, scheduler,
                            global_step=global_step,
                            checkpoint_dir=checkpoint_dir,
                            metrics={
                                'train_loss': avg_interval_loss, 'val_loss': val_loss,
                                'train_perplexity': train_ppl, 'val_perplexity': val_ppl,
                                'train_accuracy': avg_interval_acc, 'val_accuracy': val_accuracy,
                                'val_kl_divergence': val_kl, 'val_overlapping_area': val_oa,
                            },
                            is_best=True
                        )

                    if _should_reset_patience(early_stopping_on, train_improved, val_improved):
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(
                                f"Early stopping at step {global_step} after {patience} "
                                f"evaluations without improvement (metric={early_stopping_on}, "
                                f"min_delta={min_delta})"
                            )
                            stop_reason = "converged"

                    # Reset interval accumulators
                    interval_train_loss = 0.0
                    interval_train_mtp_loss = 0.0
                    interval_train_accuracy = 0.0
                    interval_optimizer_steps = 0
                    model.train()

                # Periodic checkpoint
                if global_step % save_interval == 0:
                    save_checkpoint(
                        model, optimizer, scheduler,
                        global_step=global_step,
                        checkpoint_dir=checkpoint_dir,
                        metrics={'train_loss': step_loss}
                    )

                if global_step >= total_steps:
                    stop_reason = "total_steps_reached"

            if stop_reason is not None:
                break

        progress_bar.close()

        if stop_reason is not None:
            break

    # Teardown
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(
        f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s"
    )

    logger.info(
        f"Stop reason: {stop_reason} | "
        f"Best val_perplexity: {best_val_ppl:.4f} (at step {best_val_step}) | "
        f"Patience counter at stop: {patience_counter}/{patience} | "
        f"min_delta: {min_delta}"
    )

    # Test evaluation using best model
    best_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(best_path):
        load_checkpoint(best_path, model, optimizer, load_tokenizer=False)

        test_loss, test_accuracy, test_mtp_loss, test_kl, test_oa = validate(
            model, test_loader, criterion, pad_token_id, device,
            mtp_weight=mtp_weight,
        )
        test_ppl = perplexity(test_loss)
        logger.info(
            f"Test - Loss: {test_loss:.4f}, PPL: {test_ppl:.2f}, "
            f"Accuracy: {test_accuracy:.4f}, MTP Loss: {test_mtp_loss:.4f}, "
            f"KL: {test_kl:.4f}, OA: {test_oa:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log({
                "test/loss": test_loss,
                "test/mtp_loss": test_mtp_loss,
                "test/perplexity": test_ppl,
                "test/accuracy": test_accuracy,
                "test/kl_divergence": test_kl,
                "test/overlapping_area": test_oa,
            })


    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_perplexities': train_perplexities,
        'val_perplexities': val_perplexities,
        'stop_reason': stop_reason,
        'best_val_perplexity': best_val_ppl,
        'best_val_step': best_val_step,
    }
