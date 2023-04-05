import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import repeat
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
        weight = self.embed_vocab @ self.embed_hidden
        x = F.embedding(
            x,
            weight,
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
            attn_sharing=True,
            ff_sharing=True,
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
        self.pos_embed = nn.Embedding(max_seq_len, hidden_size)
        self.register_buffer("pos_tokens", torch.arange(0, self.max_seq_len))

    def forward(self, x):
        b, l = x.shape[:2]
        pos_tokens = repeat(self.pos_tokens, "... -> b ...", b=b)
        pos_embedding = self.pos_embed(pos_tokens)[:, :l]

        x = self.embed(x)
        x = x + pos_embedding
        x = self.blocks(x)

        return x


class DecoderBlock(nn.Module):
    def __init__(self, masked_attn, attn, ff, hidden_size, max_seq_len):
        super().__init__()
        self.hidden_size = hidden_size
        self.masked_attn = masked_attn
        self.attn = attn
        self.ff = ff

        # Normalisation params not shared in paper it seems
        self.ln = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        self.ln3 = nn.LayerNorm(self.hidden_size)

        self.register_buffer("attn_mask", torch.tril(torch.ones(max_seq_len, max_seq_len)).to(torch.bool))

    def forward(self, x):
        _, l, _ = x.shape
        x = self.masked_attn(x, attn_mask=self.attn_mask[:l, :l])
        x = self.ln(x)
        # Add input from encoder for MHA
        x = self.attn(x)
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
        attn_sharing=True,
        ff_sharing=True,
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

            self.blocks.append(DecoderBlock(masked_attn, attn, ff, self.hidden_size, self.max_seq_len))

        self.pos_encoding = nn.Parameter(torch.randn((self.max_seq_len, self.hidden_size)))
        self.decode_head = nn.Sequential(
            nn.LayerNorm((hidden_size,), eps=1e-12, elementwise_affine=True),
            nn.Linear(self.hidden_size, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, self.vocab_size),
            nn.GELU()
        )

    def forward(self, x):
        l = x.shape[1]
        x = x + self.pos_encoding[:l]

        for block in self.blocks:
            x = block(x)

        x = self.decode_head(x)
        return x


class ALBERT(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        embed_dim,
        layers,
        max_seq_len,
        attn_sharing=True,
        ff_sharing=True,
        simple_decoder=False,
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

        if simple_decoder:
            self.decoder = nn.Sequential(
                nn.LayerNorm((hidden_size,), eps=1e-12, elementwise_affine=True),
                nn.Linear(hidden_size, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, vocab_size),
                nn.GELU(),
            )
        else:
            self.decoder = TransformerDecoder(
                vocab_size,
                hidden_size,
                embed_dim,
                layers,
                max_seq_len,
                attn_sharing,
                ff_sharing,
            )

        # Separate head for sentence order, not part of original implementation
        self.sentence_order = nn.Linear(vocab_size, 1)
        self.device_indicator = nn.Parameter(torch.empty(0))

    def forward(self, x):
        enc_out = self.encoder(x)
        dec_out = self.decoder(enc_out)

        sentence_order = self.sentence_order(dec_out)
        sentence_order = F.sigmoid(torch.mean(sentence_order, dim=(1, 2)))

        return dec_out, sentence_order
