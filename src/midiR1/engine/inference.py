import time
import torch
import torch.nn.functional as fn
import logging
import os
from typing import List, Union, Optional, Dict, Tuple

from tqdm import tqdm
from miditok import MusicTokenizer, TokSequence
from symusic import Score

logger = logging.getLogger(__name__)

class GenerationConfig:
    """Configuration for MIDI generation."""

    def __init__(
            self,
            max_length: int = 128,
            temperature: float = 1.2,
            top_k: int = 50,
            top_p: float = 0.9,
            repetition_penalty: float = 1.2,
            no_repeat_ngram_size: int = 3,
            eos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            do_sample: bool = True,
            use_mtp: bool = True,
            mtp_speculation_mode: bool = True,
            num_beams: int = 1,
            length_penalty: float = 1.5,
            early_stopping: bool = False,
    ):
        self.max_length = max_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.do_sample = do_sample
        self.use_mtp = use_mtp
        self.mtp_speculation_mode = mtp_speculation_mode
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping


class MidiGenerator:
    """MIDI generator with various generation strategies."""

    def __init__(self, model, tokenizer: MusicTokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        os.makedirs('logs', exist_ok=True)

    def generate(
            self,
            prompts: Union[list, Score],
            config: GenerationConfig
    ) -> Union[TokSequence, List[TokSequence]]:
        """
        Generate MIDI from prompts using the specified configuration.

        Args:
            prompts: Single prompt or list of prompts (each is a tokenized sequence list)
            config: Generation configuration

        Returns:
            Generated TokSequence or list of TokSequences
        """
        if not isinstance(prompts, list):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False

        if config.num_beams > 1:
            generated_midis = self.generate_with_beam_search(prompts, config)
        elif config.use_mtp and config.mtp_speculation_mode and hasattr(self.model, 'mtp_depth') and self.model.mtp_depth > 0:
            generated_midis = self.generate_with_speculation(prompts, config)
        else:
            generated_midis = self.generate_standard(prompts, config)

        if return_single:
            return generated_midis[0]
        return generated_midis

    def _sample_token(self, logits: torch.Tensor, generated: torch.Tensor,
                      config: GenerationConfig) -> torch.Tensor:
        """
        Sample a single token from logits with temperature, repetition penalty,
        n-gram blocking, top-k, and top-p filtering.

        Args:
            logits: [1, V] logits for next token
            generated: [1, T] current generated sequence
            config: Generation configuration

        Returns:
            [1, 1] sampled token tensor
        """
        next_logits = logits.clone()

        # Temperature scaling
        if config.temperature != 1.0:
            next_logits = next_logits / config.temperature

        # Repetition penalty
        if config.repetition_penalty != 1.0:
            for token_id in generated[0].unique():
                next_logits[0, token_id] /= config.repetition_penalty

        # N-gram repetition prevention
        if 0 < config.no_repeat_ngram_size < generated.size(1):
            ngrams = self._get_ngrams(generated[0], config.no_repeat_ngram_size)
            banned_tokens = self._get_banned_tokens(generated[0], ngrams, config.no_repeat_ngram_size)
            for token_id in banned_tokens:
                next_logits[0, token_id] = -float('inf')

        # Top-k filtering
        if config.top_k > 0:
            next_logits = self._top_k_filtering(next_logits, config.top_k)

        # Top-p (nucleus) filtering
        if config.top_p < 1.0:
            next_logits = self._top_p_filtering(next_logits, config.top_p)

        # Sample or greedy
        if config.do_sample:
            probs = fn.softmax(next_logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        else:
            return torch.argmax(next_logits, dim=-1, keepdim=True)

    def generate_standard(
            self,
            prompts: List,
            config: GenerationConfig
    ) -> List[TokSequence]:
        """
        Standard autoregressive generation with KV cache.

        Args:
            prompts: List of tokenized prompt sequences
            config: Generation configuration

        Returns:
            List of generated TokSequences
        """
        self.model.eval()
        results = []

        for in_seq in prompts:
            input_ids = torch.tensor(in_seq[0].ids, device=self.device).unsqueeze(0)
            generated = input_ids.clone()
            past_key_values = None
            max_new = min(config.max_length - generated.size(1), config.max_length)

            for step in range(max_new):
                with torch.no_grad():
                    if past_key_values is None:
                        # First pass: process entire prompt
                        current_input = generated
                        attention_mask = torch.ones_like(generated, device=self.device)
                    else:
                        # Subsequent steps: only the new token (KV cache handles history)
                        current_input = generated[:, -1:]
                        attention_mask = None

                    out = self.model(
                        current_input,
                        attention_mask=attention_mask,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )

                    if isinstance(out, tuple):
                        logits = out[0]
                        past_key_values = out[-1]
                    else:
                        logits = out

                next_logits = logits[:, -1:, :]
                next_token = self._sample_token(next_logits.squeeze(1), generated, config)

                generated = torch.cat([generated, next_token], dim=1)

                if next_token.item() == config.eos_token_id:
                    break

            full_ids = generated[0].tolist()
            new_tok_seq = TokSequence(ids=full_ids, are_ids_encoded=True)
            self.tokenizer.decode_token_ids(new_tok_seq)
            self.tokenizer.complete_sequence(new_tok_seq)
            results.append(new_tok_seq)

        return results

    def generate_with_speculation(
                self,
                prompts: List,
                config: GenerationConfig
    ) -> List[TokSequence]:
        """
        Generate MIDI using speculative decoding with MTP draft predictions.

        The MTP heads produce draft tokens for D future positions.
        A verification pass accepts drafts that match the verifier's predictions.

        Args:
            prompts: List of tokenized prompt sequences
            config: Generation configuration

        Returns:
            List of generated TokSequences
        """
        generated_midis = []
        draft_depth = self.model.mtp_depth

        for idx, prompt in enumerate(prompts):
            input_ids = torch.tensor(prompt[0].ids, device=self.device).unsqueeze(0)
            generated = input_ids.clone()
            past_key_values = None

            tokens_generated = 0
            accepted_drafts = 0
            max_new = config.max_length - generated.size(1)

            with tqdm(total=max_new, desc=f"Generating MIDI {idx + 1}/{len(prompts)}") as pbar:
                while tokens_generated < max_new:
                    with torch.no_grad():
                        if past_key_values is None:
                            current_input = generated
                            attention_mask = torch.ones_like(generated, device=self.device)
                        else:
                            current_input = generated[:, -1:]
                            attention_mask = None

                        # Get main prediction + MTP drafts
                        out = self.model(
                            current_input,
                            attention_mask=attention_mask,
                            use_mtp_drafts=True,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )

                        main_logits, mtp_draft_logits, past_key_values = out

                    # Sample main token
                    main_token = self._sample_token(main_logits[:, -1, :], generated, config)
                    generated = torch.cat([generated, main_token], dim=1)
                    tokens_generated += 1
                    pbar.update(1)

                    if main_token.item() == config.eos_token_id:
                        break

                    # Collect draft tokens from MTP heads
                    draft_tokens = []
                    for draft_logits in mtp_draft_logits:
                        draft_token = self._sample_token(
                            draft_logits[:, -1, :], generated, config
                        )
                        draft_tokens.append(draft_token)

                    if not draft_tokens:
                        continue

                    # Verification: run all draft tokens through the model
                    all_drafts = torch.cat(draft_tokens, dim=1)  # [1, D]

                    with torch.no_grad():
                        verify_out = self.model(
                            all_drafts,
                            use_cache=True,
                            past_key_values=past_key_values,
                        )
                        if isinstance(verify_out, tuple):
                            verify_logits = verify_out[0]
                            verify_cache = verify_out[-1]
                        else:
                            verify_logits = verify_out
                            verify_cache = past_key_values

                    # Accept drafts that match verification
                    n_accepted = 0
                    for d in range(len(draft_tokens)):
                        verify_token = verify_logits[:, d, :].argmax(dim=-1)
                        if verify_token.item() == draft_tokens[d].item():
                            n_accepted += 1
                            accepted_drafts += 1
                        else:
                            # Use verifier's token instead and stop accepting
                            draft_tokens[d] = verify_token.unsqueeze(0)
                            n_accepted += 1
                            break

                    # Append accepted tokens
                    accepted = torch.cat(draft_tokens[:n_accepted], dim=1)
                    generated = torch.cat([generated, accepted], dim=1)
                    tokens_generated += n_accepted
                    pbar.update(n_accepted)

                    # Trim KV cache to match actual generated length
                    cache_target_len = generated.size(1)
                    trimmed_cache = []
                    for layer_cache in verify_cache:
                        if layer_cache is not None and layer_cache.c_kv.shape[1] > cache_target_len:
                            layer_cache.trim(cache_target_len)
                        trimmed_cache.append(layer_cache)
                    past_key_values = trimmed_cache

                    # Check for EOS in accepted tokens
                    eos_found = False
                    if config.eos_token_id is not None:
                        for t in draft_tokens[:n_accepted]:
                            if t.item() == config.eos_token_id:
                                eos_found = True
                                break
                    if eos_found:
                        break

                    if generated.size(1) >= config.max_length:
                        break

            # Decode to TokSequence
            full_ids = generated[0].tolist()
            new_tok_seq = TokSequence(ids=full_ids, are_ids_encoded=True)
            self.tokenizer.decode_token_ids(new_tok_seq)
            self.tokenizer.complete_sequence(new_tok_seq)
            generated_midis.append(new_tok_seq)

            if accepted_drafts > 0 and tokens_generated > 0:
                logger.info(
                    f"Speculation stats: {accepted_drafts}/{tokens_generated} tokens "
                    f"({100 * accepted_drafts / tokens_generated:.1f}%) generated speculatively."
                )

        return generated_midis

    def generate_with_beam_search(
                self,
                prompts: List,
                config: GenerationConfig
    ) -> List[TokSequence]:
        """
        Generate MIDI using beam search.

        Args:
            prompts: List of tokenized prompt sequences
            config: Generation configuration

        Returns:
            List of generated TokSequences
        """
        generated_midis = []

        for prompt_idx, prompt in enumerate(prompts):
            input_ids = torch.tensor(prompt[0].ids, device=self.device).unsqueeze(0)

            # Initialize beams: (sequence_tensor, score)
            beams = [(input_ids.clone(), 0.0)]
            finished_beams = []

            current_length = input_ids.size(1)
            max_gen_length = min(config.max_length - current_length, config.max_length)

            with tqdm(total=max_gen_length, desc=f"Beam search {prompt_idx + 1}/{len(prompts)}") as pbar:
                for _ in range(max_gen_length):
                    if not beams:
                        break

                    new_beams = []

                    for beam_sequence, beam_score in beams:
                        if beam_sequence[0, -1].item() == config.eos_token_id:
                            finished_beams.append((beam_sequence, beam_score))
                            continue

                        attention_mask = torch.ones_like(beam_sequence, device=self.device)

                        with torch.no_grad():
                            outputs = self.model(beam_sequence, attention_mask=attention_mask)

                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        next_token_logits = logits[:, -1, :].clone() / config.temperature

                        # Repetition penalty
                        for token_id in beam_sequence[0].unique():
                            next_token_logits[0, token_id] /= config.repetition_penalty

                        if config.top_k > 0:
                            next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                        if config.top_p < 1.0:
                            next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)

                        next_token_probs = fn.softmax(next_token_logits, dim=-1)

                        topk_probs, topk_tokens = torch.topk(
                            next_token_probs, k=config.num_beams, dim=-1
                        )

                        for token, prob in zip(topk_tokens[0], topk_probs[0]):
                            new_sequence = torch.cat(
                                [beam_sequence, token.unsqueeze(0).unsqueeze(0)],
                                dim=1
                            )
                            log_prob = torch.log(prob).item()
                            new_score = beam_score + log_prob

                            if config.length_penalty != 1.0:
                                length_factor = ((5.0 + new_sequence.size(1)) / 6.0) ** config.length_penalty
                                new_score = new_score / length_factor

                            new_beams.append((new_sequence, new_score))

                    new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:config.num_beams]

                    beams = [(seq, score) for seq, score in new_beams
                             if seq[0, -1].item() != config.eos_token_id]
                    finished_beams.extend(
                        [(seq, score) for seq, score in new_beams
                         if seq[0, -1].item() == config.eos_token_id]
                    )

                    pbar.update(1)

                    if not beams or (config.early_stopping and len(finished_beams) >= config.num_beams):
                        break

            if not finished_beams and beams:
                finished_beams = beams

            if finished_beams:
                finished_beams = sorted(finished_beams, key=lambda x: x[1], reverse=True)
                best_beam = finished_beams[0][0]
                full_ids = best_beam[0].tolist()
            else:
                full_ids = input_ids[0].tolist()

            new_tok_seq = TokSequence(ids=full_ids, are_ids_encoded=True)
            self.tokenizer.decode_token_ids(new_tok_seq)
            self.tokenizer.complete_sequence(new_tok_seq)
            generated_midis.append(new_tok_seq)

        return generated_midis

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')
        return filtered_logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(fn.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = -float('inf')
        return filtered_logits

    def _get_ngrams(self, token_ids: torch.Tensor, n: int) -> List[Tuple[int, ...]]:
        """Get all n-grams from a tensor of token IDs."""
        ngrams = []
        for i in range(len(token_ids) - n + 1):
            ngram = tuple(token_ids[i:i + n].tolist())
            ngrams.append(ngram)
        return ngrams

    def _get_banned_tokens(
            self,
            token_ids: torch.Tensor,
            ngrams: List[Tuple[int, ...]],
            n: int
    ) -> List[int]:
        """Get tokens that would form a repeated n-gram."""
        banned_tokens = []
        if len(token_ids) >= (n - 1):
            current_prefix = tuple(token_ids[-(n - 1):].tolist())
            for ngram in ngrams:
                if ngram[:-1] == current_prefix:
                    banned_tokens.append(ngram[-1])
        return banned_tokens
