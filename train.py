import torch
import logging
import torchmetrics
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
from collections import defaultdict
from safetensors.torch import save_file
from accelerate import infer_auto_device_map, dispatch_model

from model import ALBERT
from datasets import huggingface_datapipeline


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    epochs = 1
    batch_size = 16
    accumulation_steps = 4
    show_every = 10
    max_seq_len = 512
    lr = 1e-4
    valid_size = 0.2

    train_loader, valid_loader = huggingface_datapipeline(max_seq_len, batch_size, valid_size)
    vocab_size = 30000
    net = ALBERT(
        vocab_size=vocab_size,
        hidden_size=1024,
        embed_dim=128,
        layers=24,
        max_seq_len=512,
    )

    device_map = infer_auto_device_map(net)
    net = dispatch_model(net, device_map)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            pred = net(batch["input_ids"])
            pred = rearrange(pred, "b l v -> b v l")
            labels_masked = batch["labels"]

            mlm_loss = F.cross_entropy(
                pred,
                labels_masked,
            )
            sop_loss = F.binary_cross_entropy(
                F.sigmoid(pred[:, 0, 0]), batch["swap"])

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
                labels_masked = batch["labels"]

                mlm_loss = F.cross_entropy()(
                    pred,
                    labels_masked,
                )

                sop_loss = F.binary_cross_entropy(
                    F.sigmoid(pred[:, 0, 0]), batch["swap"])
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
