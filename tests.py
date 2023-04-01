import torch
import pytest
import torch.nn as nn

from itertools import chain
from model import TransformerEncoder, Residual, MHSA, DecoderBlock, EncoderBlock, TransformerDecoder


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


def test_decoder(sample):
    vocab_size = 30000
    hidden_size = 512
    embed_dim = 768
    layers = 2
    max_seq_len = 1000
    decoder = TransformerDecoder(vocab_size, hidden_size, embed_dim, layers, max_seq_len)

    out = decoder(sample)
    assert out.shape == torch.Size([8, 78, 512])


def group_texts(examples, max_seq_length=1024):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
        for k, t in concatenated_examples.items()
    }
    return result


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
