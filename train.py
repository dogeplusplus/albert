import torch.nn as nn
import numpy as np
import sentencepiece as spm
import torch

from einops import rearrange

from torchdata.dataloader2 import DataLoader2
from torchdata.datapipes.iter import IterableWrapper, Collator


def mask_probability(n, N):
    return (1 / n) / sum(1 / i for i in range(1, N+1))
 

def ngram_masking(sample, percentage: 0.15, max_gram_length: 3):
    #TODO: Delete if masking on after tokenization makes more sense
    assert 0 < percentage < 1

    words = sample.split(" ")
    masked = np.zeros(len(words))
    vocab_size = sp.piece_size()

    while sum(masked) < percentage * len(words):
        span_lengths = list(range(1, max_gram_length+1))
        probs = [mask_probability(n, max_gram_length) for n in span_lengths]
        gram_length = np.random.choice(span_lengths, p=probs)
        gram_length = min(len(words), gram_length)
        position = np.random.randint(0, len(words) - gram_length + 1)
        mask_type = np.random.choice(["mask", "random", "original"], p=[0.8, 0.1, 0.1])

        if mask_type == "mask":
            mask = ["[MASK]"] * gram_length
            words[position: position+gram_length] = mask
            masked[position: position+gram_length] = 1
        elif mask_type == "random":
            rand_tokens = np.random.randint(0, vocab_size, gram_length).tolist()
            mask = sp.decode(rand_tokens).split(" ")
            words[position: position+gram_length] = mask
            masked[position: position+gram_length] = 1
        else:
            masked[position: position+gram_length] = 1

    return words, masked


def token_ngram_masking(sample, percentage: 0.15, max_gram_length: 3):
    assert 0 < percentage < 1

    masked = np.zeros(len(sample))
    vocab_size = sp.piece_size()

    while sum(masked) < percentage * len(sample):
        span_lengths = list(range(1, max_gram_length+1))
        probs = [mask_probability(n, max_gram_length) for n in span_lengths]
        gram_length = np.random.choice(span_lengths, p=probs)
        gram_length = min(len(sample), gram_length)
        position = np.random.randint(0, len(sample) - gram_length + 1)
        mask_type = np.random.choice(["mask", "random", "original"], p=[0.8, 0.1, 0.1])

        if mask_type == "mask":
            mask = [mask_token] * gram_length
            sample[position: position+gram_length] = mask
            masked[position: position+gram_length] = 1
        elif mask_type == "random":
            rand_tokens = np.random.randint(0, vocab_size, gram_length).tolist()
            sample[position: position+gram_length] = rand_tokens
            masked[position: position+gram_length] = 1
        # TODO: check if the original token should appear in the masked loss calculation
        else:
            masked[position: position+gram_length] = 1

    return sample, masked


sp = spm.SentencePieceProcessor(model_file="pubmed.model")

# special tokens are length 2
cls_token = sp.encode("[CLS]")[1]
sep_token = sp.encode("[SEP]")[1]
mask_token = sp.encode("[MASK]")[1]
pad_token = sp.encode("[PAD]")[1]

def tokenize(sample):
    first_sentence = sample[0]
    second_sentence = sample[1]

    return {
        "first_tokens": sp.encode(first_sentence),
        "second_tokens": sp.encode(second_sentence),
    }

def masking(sample):
    percentage = 0.15
    max_gram_length = 3

    first_tokens = sample["first_tokens"]
    second_tokens = sample["second_tokens"]

    first_masked, first_locs = token_ngram_masking(first_tokens, percentage, max_gram_length)
    second_masked, second_locs = token_ngram_masking(second_tokens, percentage, max_gram_length)

    swap = np.random.randint(0, 2)
    # add 2 for cls and sep for first segment, 1 for sep
    if swap:
        original_sequence = [cls_token] + second_tokens + [sep_token] + first_tokens + [sep_token]
        masked_sequence = [cls_token] + second_masked + [sep_token] + first_masked + [sep_token]
        segment_mask = (len(second_masked) + 2) * [0] + (len(first_masked) + 1) * [1]
        mask_locations = np.concatenate([(0,), second_locs, (0,), first_locs])
    else:
        original_sequence = [cls_token] + first_tokens + [sep_token] + second_tokens + [sep_token]
        masked_sequence = [cls_token] + first_masked + [sep_token] + second_masked + [sep_token]
        segment_mask = (len(first_masked) + 2) * [0] + (len(second_masked) + 1) * [1]
        mask_locations = np.concatenate([(0,), first_locs, (0,), second_locs])
    
    result = {
        "order": swap,
        "original_sequence": original_sequence,
        "masked_sequence": masked_sequence,
        "segment_mask": segment_mask,
        "mask_locations": mask_locations,
    }
 
    return result

def pad_longest(batch):
    longest_sequence = max([len(b["original_sequence"]) for b in batch])
    pad = lambda x, v: np.pad(x, (0, longest_sequence - len(x)), constant_values=v)
    batch = {
        "order": np.stack(b["order"] for b in batch),
        "original_sequence": np.stack([pad(b["original_sequence"], pad_token) for b in batch]),
        "masked_sequence": np.stack([pad(b["masked_sequence"], pad_token) for b in batch]),
        "segment_mask": np.stack([pad(b["segment_mask"], pad_token) for b in batch]),
        # Any pad tokens are not masked
        "mask_locations": np.stack([pad(b["mask_locations"], 0) for b in batch]),
    }

    batch = {
        k: torch.from_numpy(v) for k, v in batch.items()
    }
    return batch


datapipe = IterableWrapper(["pubmed.csv"])
datapipe = datapipe.open_files(encoding="utf-8").parse_csv()
datapipe = datapipe.shuffle().sharding_filter()
datapipe = datapipe.map(tokenize)
datapipe = datapipe.map(masking)
datapipe = datapipe.batch(8)
datapipe = Collator(datapipe, collate_fn=pad_longest)

from model import ALBERT

vocab_size = 30000

net = ALBERT(
    vocab_size=vocab_size,
    hidden_size=1024,
    embed_dim=128,
    layers=24,
    max_seq_len=512,
    out_features=vocab_size,
)

dl = DataLoader2(datapipe)
for sample in dl:
    pred = net(sample["masked_sequence"], sample["original_sequence"])

    # TODO: Figure out how to select to only the tokens that were masked for the loss calculation

    loss = nn.CrossEntropyLoss()(
        rearrange(pred, "b l v -> b v l"),
        sample["original_sequence"],
    )
    import pdb; pdb.set_trace()