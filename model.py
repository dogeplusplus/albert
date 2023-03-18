import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

class FactorizedEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_size,
            hidden_size,
            padding_idx=None,
            max_norm=None,
            norm_type=2.,
            scale_grad_by_freq=False,
            sparse=False,
        ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.max_norm = max_norm
        self.padding_idx = padding_idx
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse


        self.embed_vocab = nn.Parameter(torch.empty(vocab_size, embedding_size))
        self.embed_hidden = nn.Parameter(torch.empty(embedding_size, hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(self.embed_hidden)
        torch.nn.init.normal_(self.embed_vocab)
        self._fill_padding_idx_with_zero()

    def _fill_padding_idx_with_zero(self) -> None:
        if self.padding_idx is not None:
            with torch.no_grad():
                self.embed_vocab[self.padding_idx].fill_(0)
                self.embed_hidden[self.padding_idx].fill_(0)

    def forward(self, x):
        x = F.embedding(
            x,
            self.embed_vocab @ self.embed_hidden,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class EncoderBlock(nn.Module):
    def __init__(self, attn, ff, hidden_size):
        super().__init__()
        self.attn = attn
        self.ff = ff
        self.hidden_size = hidden_size

        # Normalisation params not shared in paper it seems
        self.ln = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
        x = self.attn(x)
        x = self.ln(x)
        x = self.ff(x)
        x = self.ln2(x)
        return x
    
class MHSA(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = nn.MultiheadAttention(*args, **kwargs)

    def forward(self, x, attn_mask=None):
        x, _ = self.attn(x, x, x, attn_mask=attn_mask)
        return x


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            vocab_size,
            hidden_size,
            embed_dim,
            layers,
            max_seq_len,
            attn_sharing = True,
            ff_sharing = True,
            ):
        super().__init__()
        self.vocab_size = vocab_size
        self.layers = layers
        self.hidden_size = hidden_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_heads = self.hidden_size // 64

        attn = Residual(MHSA(self.hidden_size, num_heads=self.num_heads, batch_first=True))
        ff = Residual(nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ))

        self.blocks = nn.Sequential()

        for _ in range(self.layers):
            if not attn_sharing:
                attn = deepcopy(attn)
            if not ff_sharing:
                ff = deepcopy(ff)
            
            self.blocks.append(EncoderBlock(attn, ff, self.hidden_size))

        self.embed = FactorizedEmbedding(self.vocab_size, self.embed_dim, self.hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn((self.max_seq_len, self.hidden_size)))

    def forward(self, x):
        l = x.shape[1]
        x = self.embed(x)
        x = x + self.pos_encoding[:l]
        x = self.blocks(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self, masked_attn, attn, ff, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.masked_attn = masked_attn
        self.attn = attn
        self.ff = ff

        # Normalisation params not shared in paper it seems
        self.ln = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self.ln3 = nn.LayerNorm(self.hidden_size)


    def forward(self, x, y):
        _, l, _ = x.shape
        attn_mask = torch.tril(torch.ones((l, l)))
        x = self.masked_attn(x, attn_mask=attn_mask)
        x = self.ln(x)
        # Add input from encoder for MHA
        x = self.attn(x + y)
        x = self.ln2(x)
        x = self.ff(x)
        x = self.ln3(x)

        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        embed_dim,
        layers,
        max_seq_len,
        attn_sharing = True,
        ff_sharing = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = self.hidden_size // 64
        self.embed_dim = embed_dim
        self.layers = layers
        self.max_seq_len = max_seq_len

        attn = Residual(MHSA(self.hidden_size, num_heads=self.num_heads, batch_first=True))
        masked_attn = Residual(MHSA(self.hidden_size, num_heads=self.num_heads, batch_first=True))
        self.blocks = nn.Sequential()

        ff = Residual(nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        ))

        for _ in range(self.layers):
            if not attn_sharing:
                attn = deepcopy(attn)
                masked_attn = deepcopy(masked_attn)
            if not ff_sharing:
                ff = deepcopy(ff)
            
            self.blocks.append(DecoderBlock(masked_attn, attn, ff, self.hidden_size))

    
        self.embed = FactorizedEmbedding(self.vocab_size, self.embed_dim, self.hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn((self.max_seq_len, self.hidden_size)))

    def forward(self, encoder_state, out_seq):
        l = out_seq.shape[1]
        out_emb = self.embed(out_seq)
        x = out_emb + self.pos_encoding[:l]

        for block in self.blocks:
            x = block(x, encoder_state)

        return x


# vocab_size = 10000
# embed_size = 128
# hidden_size = 768

# reg_embed = nn.Embedding(vocab_size, hidden_size)
# fact_embed = FactorizedEmbedding(vocab_size, embed_size, hidden_size)


def test_encoder_block():
    hidden_size = 512
    heads = 12
    attn = Residual(MHSA(hidden_size, num_heads=heads, batch_first=True))
    ff = Residual(nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    ))
    block = EncoderBlock(attn, ff, hidden_size)

    tokens = torch.randn((8, 78, 512))

    out = block(tokens)
    print(out.shape)


class ALBERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        embed_dim,
        layers,
        max_seq_len,
        out_features,
        attn_sharing = True,
        ff_sharing = True,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size,
            hidden_size,
            embed_dim,
            layers,
            max_seq_len,
            attn_sharing,
            ff_sharing,
        )
        self.decoder = TransformerDecoder(
            vocab_size,
            hidden_size,
            embed_dim,
            layers,
            max_seq_len,
            attn_sharing,
            ff_sharing,
        )

        self.linear = nn.Linear(hidden_size, out_features)

    def forward(self, x, y):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out, y)
        dec_out = self.linear(dec_out)
        return dec_out