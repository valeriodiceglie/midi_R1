from torch import nn
import math
from src.midiR1.transformer_decoder.positional_encoding import PositionalEncoding

class MidiTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6,
                 dim_feedforward=2048, dropout=0.1, max_seq_len=5000):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        # Decoder-only
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward, dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, src_key_padding_mask=None):
        # x: (seq_len, batch_size)
        seq_len, batch_size = x.size()
        # Embedding + positional
        x = self.token_emb(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        # Decoder: memory=None since no encoder
        # Use same x as both tgt and memory to achieve autoregressive behavior
        output = self.transformer(x, mask=mask, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)