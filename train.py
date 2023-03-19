import torch
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchmetrics
import sentencepiece as spm

from safetensors.torch import save_file
from collections import defaultdict
from tqdm import tqdm
from einops import rearrange
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService
from torchdata.datapipes.iter import IterableWrapper, Collator

from model import ALBERT


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def mask_probability(n, N):
    return (1 / n) / sum(1 / i for i in range(1, N+1))


def token_ngram_masking(sample, percentage: 0.15, max_gram_length: 3):
    masked = np.zeros(len(sample))
    vocab_size = sp.piece_size()

    while sum(masked) < percentage * len(sample):
        span_lengths = list(range(1, max_gram_length+1))
        probs = [mask_probability(n, max_gram_length) for n in span_lengths]
        gram_length = np.random.choice(span_lengths, p=probs)
        gram_length = min(len(sample), gram_length)
        position = np.random.randint(0, len(sample) - gram_length + 1)
        mask_type = np.random.choice(
            ["mask", "random", "original"], p=[0.8, 0.1, 0.1])

        if mask_type == "mask":
            mask = [MASK_TOKEN] * gram_length
            sample[position: position+gram_length] = mask
            masked[position: position+gram_length] = 1
        elif mask_type == "random":
            rand_tokens = np.random.randint(
                0, vocab_size, gram_length).tolist()
            sample[position: position+gram_length] = rand_tokens
            masked[position: position+gram_length] = 1
        # TODO: check if the original token should appear in the masked loss calculation
        else:
            masked[position: position+gram_length] = 1

    return sample, masked


sp = spm.SentencePieceProcessor(model_file="pubmed.model")

# special tokens are length 2
CLS_TOKEN = sp.encode("[CLS]")[1]
SEP_TOKEN = sp.encode("[SEP]")[1]
MASK_TOKEN = sp.encode("[MASK]")[1]
PAD_TOKEN = sp.encode("[PAD]")[1]


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

    first_masked, first_locs = token_ngram_masking(
        first_tokens, percentage, max_gram_length)
    second_masked, second_locs = token_ngram_masking(
        second_tokens, percentage, max_gram_length)

    swap = np.random.choice([0., 1.])
    # add 2 for cls and sep for first segment, 1 for sep
    if swap:
        original_sequence = [CLS_TOKEN] + second_tokens + \
            [SEP_TOKEN] + first_tokens + [SEP_TOKEN]
        masked_sequence = [CLS_TOKEN] + second_masked + \
            [SEP_TOKEN] + first_masked + [SEP_TOKEN]
        segment_mask = (len(second_masked) + 2) * \
            [0] + (len(first_masked) + 1) * [1]
        mask_locations = np.concatenate([(0,), second_locs, (0,), first_locs])
    else:
        original_sequence = [CLS_TOKEN] + first_tokens + \
            [SEP_TOKEN] + second_tokens + [SEP_TOKEN]
        masked_sequence = [CLS_TOKEN] + first_masked + \
            [SEP_TOKEN] + second_masked + [SEP_TOKEN]
        segment_mask = (len(first_masked) + 2) * \
            [0] + (len(second_masked) + 1) * [1]
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
    def pad(x, v): return np.pad(
        x, (0, longest_sequence - len(x)), constant_values=v)
    batch = {
        "order": np.stack([b["order"] for b in batch], dtype=np.float32),
        "original_sequence": np.stack([pad(b["original_sequence"], PAD_TOKEN) for b in batch]),
        "masked_sequence": np.stack([pad(b["masked_sequence"], PAD_TOKEN) for b in batch]),
        "segment_mask": np.stack([pad(b["segment_mask"], PAD_TOKEN) for b in batch]),
        # Any pad tokens are not masked
        "mask_locations": np.stack([pad(b["mask_locations"], 0) for b in batch]),
    }

    batch = {
        k: torch.from_numpy(v) for k, v in batch.items()
    }
    return batch


def main():
    epochs = 1
    batch_size = 8
    accumulation_steps = 4
    show_every = 10
    max_seq_len = 512
    lr = 1e-4
    valid_size = 0.2
    seed = 42

    def truncate(batch):
        for k in ["original_sequence", "masked_sequence", "segment_mask", "mask_locations"]:
            batch[k] = batch[k][:max_seq_len]

        return batch

    total_length = len(pd.read_csv("pubmed.csv"))
    logger.info(f"Total Length: {total_length}")

    datapipe = IterableWrapper(["pubmed.csv"])
    datapipe = datapipe.open_files(encoding="utf-8").parse_csv()

    train_ds, valid_ds = datapipe.random_split(
        weights={"train": 1 - valid_size, "valid": valid_size},
        seed=seed,
        total_length=total_length,
    )

    def preprocess_dataset(ds):
        ds = ds.shuffle().sharding_filter()
        ds = ds.map(tokenize)
        ds = ds.map(masking)
        ds = ds.map(truncate)
        ds = ds.batch(batch_size)
        ds = Collator(ds, collate_fn=pad_longest)

        mpi_service = MultiProcessingReadingService(
            num_workers=8, prefetch_factor=4)
        dl = DataLoader2(ds, reading_service=mpi_service)

        return dl

    train_loader = preprocess_dataset(train_ds)
    valid_loader = preprocess_dataset(valid_ds)

    vocab_size = 30000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = ALBERT(
        vocab_size=vocab_size,
        hidden_size=1024,
        embed_dim=128,
        layers=24,
        max_seq_len=512,
    )
    net.to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)

    step = 0
    for epoch in range(epochs):
        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch:03}")

        metrics = defaultdict(lambda: torchmetrics.MeanMetric())

        net.train()
        for batch in train_bar:
            batch = {
                k: v.to(device) for k, v in batch.items()
            }
            pred = net(batch["masked_sequence"])
            pred = rearrange(pred, "b l v -> b v l")
            # Set ignore value to non-masked tokens when calculating cross entropy
            label_masked = torch.where(
                batch["mask_locations"] == 0, -100, batch["original_sequence"])

            mlm_loss = F.cross_entropy(
                pred,
                label_masked,
            )
            sop_loss = F.binary_cross_entropy(
                F.sigmoid(pred[:, 0, 0]), batch["order"])

            total_loss = sop_loss + mlm_loss
            total_loss.backward()

            metrics["train_total_loss"].update(total_loss.item())
            metrics["train_mlm_loss"].update(mlm_loss.item())
            metrics["train_sop_loss"].update(sop_loss.item())

            if step % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if step % show_every == 0:
                train_bar.set_postfix({k: float(v.compute())
                                      for k, v in metrics.items()})

            step += 1

        net.eval()
        valid_bar = tqdm(valid_loader, desc=f"Valid Epoch {epoch:03}")
        for batch in valid_bar:
            with torch.no_grad():
                batch = {
                    k: v.to(device) for k, v in batch.items()
                }
                pred = net(batch["masked_sequence"])
                pred = rearrange(pred, "b l v -> b v l")
                # Set ignore value to non-masked tokens when calculating cross entropy
                label_masked = torch.where(
                    batch["mask_locations"] == 0, -100, batch["original_sequence"])

                mlm_loss = F.cross_entropy()(
                    pred,
                    label_masked,
                )

                sop_loss = F.binary_cross_entropy(
                    F.sigmoid(pred[:, 0, 0]), batch["order"])
                total_loss = mlm_loss + sop_loss

            metrics["valid_total_loss"].update(total_loss.item())
            metrics["valid_mlm_loss"].update(mlm_loss.item())
            metrics["valid_sop_loss"].update(sop_loss.item())

            if step % show_every == 0:
                valid_bar.set_postfix({k: float(v.compute())
                                      for k, v in metrics.items()})

            step += 1

    save_file(net, "model.safetensors")


if __name__ == "__main__":
    main()
