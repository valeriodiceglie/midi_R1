"""Resize a MidiR1 model's vocabulary (embedding + output head).

Used when adding conditioning tokens for fine-tuning: the pre-trained
weights are copied into larger tensors and the new rows are initialised
from scratch.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

logger = logging.getLogger(__name__)


def resize_model_vocab(
    model,
    new_vocab_size: int,
    init_method: str = "normal",
) -> None:
    """Resize ``model.embedding`` and ``model.output_head`` in-place.

    Parameters
    ----------
    model:
        A ``MidiR1`` instance.
    new_vocab_size:
        Target vocabulary size (must be >= current vocab size).
    init_method:
        How to initialise new rows.

        * ``"normal"`` — ``N(0, 0.02)``
        * ``"mean"``   — mean of existing embedding vectors
    """
    old_vocab_size = model.config["vocab_size"]
    if new_vocab_size == old_vocab_size:
        return
    if new_vocab_size < old_vocab_size:
        raise ValueError(
            f"Cannot shrink vocab from {old_vocab_size} to {new_vocab_size}"
        )

    hidden_dim = model.config["hidden_dim"]

    # --- Embedding -----------------------------------------------------------
    old_emb_weight = model.embedding.weight.data  # [old_V, D]
    new_embedding = nn.Embedding(new_vocab_size, hidden_dim)

    if init_method == "mean":
        mean_vec = old_emb_weight.mean(dim=0)
        new_embedding.weight.data[:] = mean_vec
    else:  # "normal"
        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)

    new_embedding.weight.data[:old_vocab_size] = old_emb_weight

    # --- Output head ---------------------------------------------------------
    old_head_weight = model.output_head.weight.data  # [old_V, D]
    old_head_bias = model.output_head.bias  # may be None
    has_bias = old_head_bias is not None

    new_output_head = nn.Linear(hidden_dim, new_vocab_size, bias=has_bias)

    if init_method == "mean":
        mean_w = old_head_weight.mean(dim=0)
        new_output_head.weight.data[:] = mean_w
        if has_bias:
            mean_b = old_head_bias.data.mean()
            new_output_head.bias.data[:] = mean_b
    else:  # "normal"
        nn.init.normal_(new_output_head.weight, mean=0.0, std=0.02)
        if has_bias:
            nn.init.zeros_(new_output_head.bias)

    new_output_head.weight.data[:old_vocab_size] = old_head_weight
    if has_bias:
        new_output_head.bias.data[:old_vocab_size] = old_head_bias.data

    # --- Replace modules on model --------------------------------------------
    model.embedding = new_embedding
    model.output_head = new_output_head

    # --- Update MTP shared references ----------------------------------------
    if model.mtp is not None:
        model.mtp.shared_embedding = model.embedding
        model.mtp.shared_output_head = model.output_head

    model.config["vocab_size"] = new_vocab_size

    logger.info(
        "Resized vocab %d → %d (init=%s, mtp_updated=%s)",
        old_vocab_size, new_vocab_size, init_method, model.mtp is not None,
    )
