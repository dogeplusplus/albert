import torch
import logging
import lightning.pytorch as pl
import torch.nn.functional as F

from model import ALBERT
from einops import rearrange
from data_loader import huggingface_datapipeline
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LanguageModelingTrainer(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        embed_dim: int,
        layers: int,
        max_seq_len: int,
        lr: float,
        attn_sharing: bool = True,
        ff_sharing: bool = True,
        simple_decoder: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        self.model = ALBERT(
            vocab_size,
            hidden_size,
            embed_dim,
            layers,
            max_seq_len,
            attn_sharing,
            ff_sharing,
            simple_decoder,
        )

    def training_step(self, batch, batch_idx):
        pred, sentence_order = self.model(batch["input_ids"])
        pred = rearrange(pred, "b l v -> b v l")
        labels_masked = batch["labels"]

        mlm_loss = F.cross_entropy(pred, labels_masked)
        sop_loss = F.binary_cross_entropy(sentence_order, batch["swap"])
        total_loss = sop_loss + mlm_loss

        self.log("train_total_loss", total_loss.item(), prog_bar=True)
        self.log("train_mlm_loss", mlm_loss.item(), prog_bar=True)
        self.log("train_sop_loss", sop_loss.item(), prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pred, sentence_order = self.model(batch["input_ids"])
        pred = rearrange(pred, "b l v -> b v l")
        labels_masked = batch["labels"]

        mlm_loss = F.cross_entropy(pred, labels_masked)
        sop_loss = F.binary_cross_entropy(sentence_order, batch["swap"])
        total_loss = mlm_loss + sop_loss

        self.log("valid_total_loss", total_loss.item(), prog_bar=True)
        self.log("valid_mlm_loss", mlm_loss.item(), prog_bar=True)
        self.log("valid_sop_loss", sop_loss.item(), prog_bar=True)

        return total_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer


def main():
    epochs = 3
    batch_size = 2
    accumulation_steps = 4
    max_seq_len = 512
    lr = 1e-4
    valid_size = 0.2
    vocab_size = 30000

    torch.set_float32_matmul_precision("medium")

    train_loader, valid_loader = huggingface_datapipeline(max_seq_len, batch_size, valid_size)

    progress_bar = RichProgressBar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
        )
    )
    wandb_log = WandbLogger(project="albert-transformer")

    model_summary = RichModelSummary()

    trainer = pl.Trainer(
        default_root_dir="models",
        accumulate_grad_batches=accumulation_steps,
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[progress_bar, model_summary],
        logger=wandb_log,
    )

    vocab_size = vocab_size
    hidden_size = 1024
    embed_dim = 128
    layers = 24
    max_seq_len = 512

    model = LanguageModelingTrainer(
        vocab_size,
        hidden_size,
        embed_dim,
        layers,
        max_seq_len,
        lr,
        ff_sharing=False,
        attn_sharing=False,
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
