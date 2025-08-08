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
    """Configuration for text generation."""

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
        """
        Initialize generation configuration.

        Args:
            max_length: Maximum length of generated text
            temperature: Temperature for sampling
            top_k: K for top-k sampling
            top_p: P for nucleus sampling
            repetition_penalty: Penalty for token repetition
            no_repeat_ngram_size: Size of n-grams to avoid repeating
            eos_token_id: End of sequence token ID
            pad_token_id: Padding token ID
            do_sample: Whether to sample (True) or use greedy decoding (False)
            use_mtp: Whether to use multi-token prediction
            mtp_speculation_mode: Whether to use speculative decoding with MTP
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
            early_stopping: Whether to stop early in beam search
        """
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
    """Text generator with various generation strategies."""

    def __init__(self, model, tokenizer: MusicTokenizer, device='cuda'):
        """
        Initialize TextGenerator.

        Args:
            model: model
            tokenizer: Tokenizer
            device: Device to run generation on
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Create directory for generation logs
        os.makedirs('logs', exist_ok=True)

    def generate(
            self,
            prompts: Union[TokSequence, List[TokSequence]],
            config: GenerationConfig
    ) -> Union[TokSequence, List[TokSequence]]:
        """
        Generate text from prompts using the specified configuration.

        Args:
            prompts: Single prompt or list of prompts
            config: Generation configuration

        Returns:
            Generated MIDI or list of generated MIDIs
        """
        # Convert single prompt to list
        if isinstance(prompts, Score):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False

        # Choose generation strategy based on config
        if config.num_beams > 1:
            # Beam search
            generated_midis = self.generate_with_beam_search(prompts, config)
        elif config.use_mtp and config.mtp_speculation_mode:
            # Speculative decoding with multi-token prediction
            generated_midis = self.generate_with_speculation(prompts, config)
        else:
            # Standard auto-regressive generation
            generated_midis = self.generate_standard(prompts, config)

        # Return single text or list based on input
        if return_single:
            return generated_midis[0]
        return generated_midis

    def generate_standard(
            self,
            prompts: List[TokSequence],
            config: GenerationConfig
    ) -> List[TokSequence]:

        """
        Standard autoregressive text generation.

        Args:
            prompts: List of prompts
            config: Generation configuration

        Returns:
            List of generated texts
        """

        self.model.eval()
        results = []

        for in_seq in prompts:
            input_ids = torch.tensor(in_seq[0].ids, device=self.device).unsqueeze(0)
            generated = input_ids.clone()
            attention = torch.ones_like(generated, device=self.device)
            max_new = min(config.max_length - generated.size(1), config.max_length)

            for _ in range(max_new):
                with torch.no_grad():
                    out = self.model(generated, attention_mask=attention)
                logits = out[0] if isinstance(out, tuple) else out
                next_logits = logits[:, -1, :]

                # temperature
                if config.temperature != 1.0:
                    next_logits = next_logits / config.temperature

                if config.repetition_penalty != 1.0:
                    for i in range(logits.size(0)):
                        for token_id in generated[i].unique():
                            next_logits[i, token_id] /= config.repetition_penalty

                    # Apply n-gram repetition prevention
                if 0 < config.no_repeat_ngram_size < generated.size(1):
                    for i in range(logits.size(0)):
                        ngrams = self._get_ngrams(generated[i], config.no_repeat_ngram_size)
                        banned_tokens = self._get_banned_tokens(generated[i], ngrams, config.no_repeat_ngram_size)
                        for token_id in banned_tokens:
                            next_logits[i, token_id] = -float('inf')

                # top-k / top-p
                if config.top_k > 0:
                    next_logits = self._top_k_filtering(next_logits, config.top_k)
                if config.top_p < 1.0:
                    next_logits = self._top_p_filtering(next_logits, config.top_p)

                if config.do_sample:
                    probs = fn.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    # Log token probabilities for analysis
                    token_prob = probs[0, next_token[0, 0]].item()

                    # Get top 5 candidates for logging
                    top_values, top_indices = torch.topk(probs[0], k=5)
                    top_candidates = [(idx.item(), val.item()) for idx, val in zip(top_indices, top_values)]
                else:
                    next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
                    token_prob = fn.softmax(next_logits, dim=-1)[0, next_token[0, 0]].item()
                    top_candidates = [(next_token[0, 0].item(), token_prob)]

                generated = torch.cat([generated, next_token], dim=1)
                attention = torch.cat([attention, torch.ones((1,1), device=self.device)], dim=1)

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
                prompts: List[TokSequence],
                config: GenerationConfig
    ) -> List[TokSequence]:
        """
        Generate MIDI using speculative decoding with MTP.

        Args:
            prompts: List of prompts
            config: Generation configuration

        Returns:
            List of generated MIDIs
        """
        generated_midis = []

        for idx, prompt in enumerate(prompts):
            # Encode prompt
            input_ids = torch.tensor(prompt[0].ids, device=self.device).unsqueeze(0)

            # Track generation statistics
            tokens_generated = 0
            speculative_tokens_accepted = 0

            # Initialize generation
            generated = input_ids.clone()
            attention_mask = torch.ones_like(generated, device=self.device)

            # Generation loop
            current_length = generated.size(1)
            max_gen_length = min(config.max_length - current_length, config.max_length)

            with tqdm(total=max_gen_length, desc=f"Generating MIDIs {idx + 1}/{len(prompts)}") as pbar:
                while current_length < config.max_length:
                    # Forward pass
                    with torch.no_grad():
                        outputs = self.model(generated, attention_mask=attention_mask)

                    # Process outputs - handle both main predictions and MTP
                    if isinstance(outputs, tuple) and len(outputs) > 1:
                        main_logits, mtp_logits = outputs
                    else:
                        # MTP not available, fall back to standard generation
                        main_logits = outputs
                        mtp_logits = None

                    # Sample next token from main logits
                    next_token_logits = main_logits[:, -1, :].clone() / config.temperature

                    # Apply repetition penalty
                    for token_id in generated[0].unique():
                        next_token_logits[0, token_id] /= config.repetition_penalty

                    # Filter logits
                    if config.top_k > 0:
                        next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                    if config.top_p < 1.0:
                        next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)

                    # Sample token
                    if config.do_sample:
                        probs = fn.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                    # Add to sequence
                    generated = torch.cat([generated, next_token], dim=1)
                    attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)], dim=1)
                    current_length += 1
                    tokens_generated += 1
                    pbar.update(1)

                    # Check for EOS
                    if next_token.item() == config.eos_token_id:
                        break

                    # Speculative decoding using MTP if available
                    if mtp_logits is not None and mtp_logits.size(1) > 0:
                        # Get MTP predictions for the next tokens
                        # We'll try to predict up to depth tokens ahead
                        depth = min(mtp_logits.size(1), 3)  # Limit depth to avoid too much speculation

                        speculation_successful = False
                        for d in range(depth):
                            # MTP logits for the current position
                            mtp_token_logits = mtp_logits[0, d, -1, :].clone() / config.temperature

                            # Apply repetition penalty
                            for token_id in generated[0].unique():
                                mtp_token_logits[token_id] /= config.repetition_penalty

                            # Filter logits
                            if config.top_k > 0:
                                mtp_token_logits = self._top_k_filtering(mtp_token_logits.unsqueeze(0),
                                                                         config.top_k).squeeze(0)
                            if config.top_p < 1.0:
                                mtp_token_logits = self._top_p_filtering(mtp_token_logits.unsqueeze(0),
                                                                         config.top_p).squeeze(0)

                            # Sample speculative token
                            if config.do_sample:
                                mtp_probs = fn.softmax(mtp_token_logits, dim=-1)
                                speculative_token = torch.multinomial(mtp_probs.unsqueeze(0), num_samples=1)
                            else:
                                speculative_token = torch.argmax(mtp_token_logits, dim=-1, keepdim=True).unsqueeze(
                                    0)

                            # In a real implementation, we'd verify this token
                            # but for simplicity, we'll accept it with high probability
                            if torch.rand(1).item() < 0.8:  # 80% chance to accept speculative token
                                generated = torch.cat([generated, speculative_token], dim=1)
                                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self.device)],
                                                           dim=1)
                                current_length += 1
                                tokens_generated += 1
                                speculative_tokens_accepted += 1
                                pbar.update(1)
                                speculation_successful = True

                                # Check for EOS
                                if speculative_token.item() == config.eos_token_id:
                                    break
                            else:
                                # Speculation failed, don't continue with deeper tokens
                                break

                        # If we failed to use speculation, continue standard generation
                        if not speculation_successful:
                            continue

                    # If we've reached max length, break
                    if current_length >= config.max_length:
                        break

            # Decode and add to results
            full_ids = generated[0].tolist()
            new_tok_seq = TokSequence(ids=full_ids, are_ids_encoded=True)
            self.tokenizer.decode_token_ids(new_tok_seq)
            self.tokenizer.complete_sequence(new_tok_seq)
            generated_midis.append(new_tok_seq)

            # Log speculation statistics
            if speculative_tokens_accepted > 0:
                logger.info(f"Speculation stats: {speculative_tokens_accepted}/{tokens_generated} tokens "
                            f"({100 * speculative_tokens_accepted / tokens_generated:.1f}%) generated speculatively.")

        return generated_midis

    def generate_with_beam_search(
                self,
                prompts: List[TokSequence],
                config: GenerationConfig
    ) -> List[TokSequence]:
        """
        Generate MIDI using beam search.

        Args:
            prompts: List of prompts
            config: Generation configuration

        Returns:
            List of generated MIDIs
        """
        generated_midis = []

        for prompt_idx, prompt in enumerate(prompts):
            # Encode prompt

            input_ids = torch.tensor(prompt[0].ids, device=self.device).unsqueeze(0)
            generated = input_ids.clone()

            # Initialize beams with the input sequence and score 0
            beams = [(input_ids.clone(), 0.0)]
            finished_beams = []

            # Generation loop
            current_length = input_ids.size(1)
            max_gen_length = min(config.max_length - current_length, config.max_length)

            with tqdm(total=max_gen_length, desc=f"Beam search {prompt_idx + 1}/{len(prompts)}") as pbar:
                for _ in range(max_gen_length):
                    if not beams:
                        break

                    new_beams = []

                    for beam_idx, (beam_sequence, beam_score) in enumerate(beams):
                        # Skip if this beam is done
                        if beam_sequence[0, -1].item() == config.eos_token_id:
                            finished_beams.append((beam_sequence, beam_score))
                            continue

                        # Forward pass
                        attention_mask = torch.ones_like(beam_sequence, device=self.device)

                        with torch.no_grad():
                            outputs = self.model(beam_sequence, attention_mask=attention_mask)

                        # Handle outputs
                        if isinstance(outputs, tuple):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        # Get next token logits
                        next_token_logits = logits[:, -1, :].clone() / config.temperature

                        # Apply repetition penalty
                        for token_id in beam_sequence[0].unique():
                            next_token_logits[0, token_id] /= config.repetition_penalty

                        # Apply top-k and top-p filtering
                        if config.top_k > 0:
                            next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
                        if config.top_p < 1.0:
                            next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)

                        # Convert logits to probabilities
                        next_token_probs = fn.softmax(next_token_logits, dim=-1)

                        # Get top candidates for this beam
                        topk_probs, topk_tokens = torch.topk(
                            next_token_probs, k=config.num_beams, dim=-1
                        )

                        # Expand beams
                        for token_idx, (token, prob) in enumerate(zip(topk_tokens[0], topk_probs[0])):
                            # Create new beam by appending token
                            new_sequence = torch.cat(
                                [beam_sequence, token.unsqueeze(0).unsqueeze(0)],
                                dim=1
                            )

                            # Update score with log probability
                            log_prob = torch.log(prob).item()
                            new_score = beam_score + log_prob

                            # Apply length penalty
                            if config.length_penalty != 1.0:
                                length_factor = ((5.0 + new_sequence.size(1)) / 6.0) ** config.length_penalty
                                new_score = new_score / length_factor

                            new_beams.append((new_sequence, new_score))

                    # Sort and keep top beams
                    new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:config.num_beams]

                    # Update beams
                    beams = [(seq, score) for seq, score in new_beams if seq[0, -1].item() != config.eos_token_id]

                    # Add completed beams
                    finished_beams.extend(
                        [(seq, score) for seq, score in new_beams if seq[0, -1].item() == config.eos_token_id])

                    # Update progress
                    pbar.update(1)

                    # Early stopping if all beams are finished
                    if not beams or (config.early_stopping and len(finished_beams) >= config.num_beams):
                        break

            # If no beams finished, use the best unfinished ones
            if not finished_beams and beams:
                finished_beams = beams

            # If we have some finished beams, select best one
            if finished_beams:
                finished_beams = sorted(finished_beams, key=lambda x: x[1], reverse=True)
                best_beam = finished_beams[0][0]
                full_ids = best_beam[0].tolist()
                new_tok_seq = TokSequence(ids=full_ids, are_ids_encoded=True)

            else:
                # Fallback to the input
                full_ids = input_ids[0].tolist()
                new_tok_seq = TokSequence(ids=full_ids)
                self.tokenizer.decode_token_ids(new_tok_seq)
                self.tokenizer.complete_sequence(new_tok_seq)

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

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
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
        """
        Get tokens that would form a repeated n-gram.

        Args:
            token_ids: Current sequence of tokens
            ngrams: List of existing n-grams
            n: Size of n-grams

        Returns:
            List of banned tokens
        """
        banned_tokens = []

        # Check if current (n-1)-gram exists and would form a banned n-gram
        if len(token_ids) >= (n - 1):
            current_prefix = tuple(token_ids[-(n - 1):].tolist())
            for ngram in ngrams:
                if ngram[:-1] == current_prefix:
                    banned_tokens.append(ngram[-1])

        return banned_tokens

    # Utility functions
# def sample_text(model, tokenizer, prompts, num_samples=5, max_length=100, device='cuda'):
#     """
#     Generate multiple text samples from each prompt for analysis.
#
#     Args:
#         model: DeepSeek model
#         tokenizer: Tokenizer
#         prompts: List of prompts
#         num_samples: Number of samples per prompt
#         max_length: Maximum generation length
#         device: Device to run on
#
#     Returns:
#         Dictionary mapping prompts to lists of generated samples
#     """
#     generator = MidiGenerator(model, tokenizer, device)
#
#     results = {}
#     for prompt in prompts:
#         samples = []
#
#         # Define configurations with different parameters
#         configs = [
#             # Standard sampling
#             GenerationConfig(
#                 max_length=max_length,
#                 temperature=0.8,
#                 top_p=0.9,
#                 repetition_penalty=1.2,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id
#             ),
#             # Creative sampling
#             GenerationConfig(
#                 max_length=max_length,
#                 temperature=1.2,
#                 top_p=0.95,
#                 repetition_penalty=1.05,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id
#             ),
#             # More focused
#             GenerationConfig(
#                 max_length=max_length,
#                 temperature=0.6,
#                 top_p=0.85,
#                 repetition_penalty=1.3,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id
#             ),
#             # Beam search
#             GenerationConfig(
#                 max_length=max_length,
#                 do_sample=False,
#                 num_beams=4,
#                 length_penalty=1.0,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id
#             ),
#             # With MTP
#             GenerationConfig(
#                 max_length=max_length,
#                 temperature=0.7,
#                 top_p=0.9,
#                 repetition_penalty=1.2,
#                 use_mtp=True,
#                 mtp_speculation_mode=True,
#                 eos_token_id=tokenizer.eos_token_id,
#                 pad_token_id=tokenizer.pad_token_id
#             )
#         ]
#
#         for i, config in enumerate(configs):
#             generated = generator.generate(prompt, config)
#             samples.append({
#                 'config': f"Config {i + 1}",
#                 'parameters': {k: v for k, v in config.__dict__.items() if not k.startswith('_')},
#                 'text': generated
#             })
#
#         results[prompt] = samples
#
#     return results

# def evaluate_model_generation(model, tokenizer, device='cuda'):
#     """
#     Evaluate model text generation on a set of standard prompts.
#
#     Args:
#         model: DeepSeek model
#         tokenizer: Tokenizer
#         device: Device to run on
#
#     Returns:
#         Dictionary with evaluation results
#     """
#     # Standard evaluation prompts
#     eval_prompts = [
#         "The history of artificial intelligence began",
#         "The three most important factors in real estate are",
#         "In recent years, climate change has",
#         "The solution to the equation x² + 5x + 6 = 0 is",
#         "The best way to learn a new language is to"
#     ]
#
#     # Create generator
#     generator = MidiGenerator(model, tokenizer, device)
#
#     # Define generation configs to test
#     configs = {
#         'greedy': GenerationConfig(
#             max_length=100,
#             do_sample=False,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         ),
#         'sampling': GenerationConfig(
#             max_length=100,
#             temperature=0.8,
#             top_p=0.9,
#             repetition_penalty=1.2,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         ),
#         'beam_search': GenerationConfig(
#             max_length=100,
#             do_sample=False,
#             num_beams=4,
#             length_penalty=1.0,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         ),
#         'mtp_speculation': GenerationConfig(
#             max_length=100,
#             temperature=0.7,
#             top_p=0.9,
#             repetition_penalty=1.2,
#             use_mtp=True,
#             mtp_speculation_mode=True,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )
#     }
#
#     # Generate text and collect results
#     results = {}
#
#     for config_name, config in configs.items():
#         config_results = {}
#
#         # Generate text for each prompt
#         for prompt in eval_prompts:
#             start_time = time.time()
#             generated = generator.generate(prompt, config)
#             generation_time = time.time() - start_time
#
#             # Analyze result
#             tokens_generated = len(tokenizer.encode(generated)) - len(tokenizer.encode(prompt))
#
#             config_results[prompt] = {
#                 'prompt': prompt,
#                 'generated_text': generated,
#                 'tokens_generated': tokens_generated,
#                 'generation_time': generation_time,
#                 'tokens_per_second': tokens_generated / generation_time if generation_time > 0 else 0
#             }
#
#         results[config_name] = config_results
#
#     # Create summary
#     summary = {
#         'generation_settings': {name: {k: v for k, v in config.__dict__.items()
#                                        if not k.startswith('_')} for name, config in configs.items()},
#         'performance': {
#             name: {
#                 'average_tokens_per_second': sum(r['tokens_per_second'] for r in config_result.values()) / len(
#                     config_result),
#                 'average_generation_time': sum(r['generation_time'] for r in config_result.values()) / len(
#                     config_result),
#                 'average_tokens_generated': sum(r['tokens_generated'] for r in config_result.values()) / len(
#                     config_result)
#             }
#             for name, config_result in results.items()
#         },
#         'detailed_results': results
#     }
#
#     return summary
