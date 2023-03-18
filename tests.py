import torch
import pytest
import torch.nn as nn

from model import TransformerEncoder, Residual, MHSA, DecoderBlock, EncoderBlock


@pytest.fixture
def sample():
    tokens = torch.randint(0, 100, (8, 78))
    return tokens


def test_encoder(sample):
    vocab_size = 30000
    hidden_size = 512
    embed_dim = 768
    layers = 2
    max_seq_len = 1000
    encoder = TransformerEncoder(vocab_size, hidden_size, embed_dim, layers, max_seq_len)

    out = encoder(sample)
    assert out.shape == torch.Size([8, 78, 512])


def test_decoder_block():
    hidden_size = 768
    num_heads = hidden_size // 64
    attn = Residual(MHSA(hidden_size, num_heads=num_heads, batch_first=True))
    masked_attn = Residual(MHSA(hidden_size, num_heads=num_heads, batch_first=True))
    ff = Residual(nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    ))
    block = DecoderBlock(masked_attn, attn, ff, hidden_size)
    output_embedding = torch.randn([8, 52, 768])
    encoder_state = torch.randn([8, 52, 768])
    out = block(output_embedding, encoder_state)
    assert out.shape == torch.Size([8, 52, 768])


def test_encoder_block():
    hidden_size = 512
    heads = 4
    attn = Residual(MHSA(hidden_size, num_heads=heads, batch_first=True))
    ff = Residual(nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    ))
    block = EncoderBlock(attn, ff, hidden_size)

    input_embedding = torch.randn([4, 28, 512])
    out = block(input_embedding)
    assert out.shape == torch.Size([4, 28, 512])

