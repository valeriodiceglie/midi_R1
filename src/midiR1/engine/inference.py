# inference.py
import torch
import torch.nn.functional as F
from midiR1.model.model import R1Model
from symusic import Score

def top_k_top_p_filtering(logits, top_k: int = 0, top_p: float = 1.0, filter_value: float = -1e10):
    """ From HuggingFace: keep only top_k and/or nucleus top_p """
    vocab_size = logits.size(-1)
    # Top‐k
    if top_k > 0:
        kth_vals = torch.topk(logits, top_k)[0][..., -1, None]
        logits = torch.where(logits < kth_vals, filter_value, logits)
    # Top‐p
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # mask out tokens above threshold (but keep at least one)
        mask = cumulative > top_p
        mask[..., 1:] = mask[..., :-1]
        mask[..., 0] = False
        sorted_logits[mask] = filter_value
        logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    return logits

class Inference:
    def __init__(
        self,
        model: R1Model,
        tokenizer,
        device: torch.device = None,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: int = None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.eos_token_id = eos_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self, prompt_score=None):
        # 1) tokenize prompt (or empty)
        if prompt_score is None:
            prompt_score = Score()
        enc = self.tokenizer(prompt_score)[0]
        input_ids = torch.tensor(enc.tokens_to_ids, device=self.device).unsqueeze(0)  # (1, L)

        key_cache = [None] * len(self.model.transformer.layers)
        value_cache = [None] * len(self.model.transformer.layers)

        # 2) autoregressive sampling
        for _ in range(self.max_new_tokens):
            # forward returns (logits, key_cache, value_cache) when no labels
            logits, new_keys, new_values = self.model(
                input_ids,
                key_cache=key_cache,
                value_cache=value_cache
            )
            key_cache = new_keys
            value_cache = new_values
            
            next_logits = logits[:, -1, :] / max(self.temperature, 1e-5)

            filtered = top_k_top_p_filtering(
                next_logits,
                top_k=self.top_k,
                top_p=self.top_p
            )
            probs = F.softmax(filtered, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (1,1)

            input_ids = torch.cat([input_ids, next_id], dim=1)
            if next_id.item() == self.eos_token_id:
                break

        # 3) decode to Score
        generated_ids = input_ids.squeeze().tolist()
        token_strs = self.tokenizer.ids_to_tokens(generated_ids)
        return self.tokenizer.tokens_to_score(token_strs)
