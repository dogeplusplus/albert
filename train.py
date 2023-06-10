import torch
import logging
import pytorch_lightning as pl
import torch.nn.functional as F

from pathlib import Path
from einops import rearrange
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary, ModelCheckpoint


from model import ALBERT, HFALBERT
from imdb_data import imdb_data_pipeline

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
        debugging: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        if not debugging:
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
        else:
            self.model = HFALBERT(vocab_size, embed_dim, hidden_size, layers)

    def training_step(self, batch, batch_idx):
        pred, sentence_order = self.model(batch["input_ids"])
        pred = rearrange(pred, "b l v -> b v l")
        labels_masked = batch["labels"]

        mlm_loss = F.cross_entropy(pred, labels_masked)
        sop_loss = F.binary_cross_entropy(sentence_order, batch["swap"])
        total_loss = sop_loss + mlm_loss

        self.log("train_loss", total_loss.item(), prog_bar=True, on_epoch=True)
        self.log("train_mlm_loss", mlm_loss.item(), prog_bar=True, on_epoch=True)
        self.log("train_sop_loss", sop_loss.item(), prog_bar=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        pred, sentence_order = self.model(batch["input_ids"])
        pred = rearrange(pred, "b l v -> b v l")
        labels_masked = batch["labels"]

        mlm_loss = F.cross_entropy(pred, labels_masked)
        sop_loss = F.binary_cross_entropy(sentence_order, batch["swap"])
        total_loss = mlm_loss + sop_loss

        self.log("valid_loss", total_loss.item(), prog_bar=True, on_epoch=True)
        self.log("valid_mlm_loss", mlm_loss.item(), prog_bar=True, on_epoch=True)
        self.log("valid_sop_loss", sop_loss.item(), prog_bar=True, on_epoch=True)

        return total_loss

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        return optimizer


def main():
    epochs = 100
    batch_size = 4
    accumulation_steps = 4
    max_seq_len = 512
    lr = 1e-4
    vocab_size = 30000
    simple_decoder = True
    quick_test = False
    log_wandb = False

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")

    train_loader, valid_loader = imdb_data_pipeline(batch_size, quick_test)

    progress_bar = RichProgressBar()

    wandb_log = None
    if log_wandb:
        id = "magic_bacon"
        wandb_log = WandbLogger(project="albert-transformer", version=0, id=id)

    checkpoint_path = "models"
    checkpoint = ModelCheckpoint(checkpoint_path, filename="{epoch}-{val_loss:.2f}", save_last=True)

    model_summary = RichModelSummary()

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulation_steps,
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[progress_bar, model_summary, checkpoint],
        logger=wandb_log,
    )

    vocab_size = vocab_size
    hidden_size = 1024
    embed_dim = 128
    layers = 12
    max_seq_len = 512

    model = LanguageModelingTrainer(
        vocab_size,
        hidden_size,
        embed_dim,
        layers,
        max_seq_len,
        lr,
        ff_sharing=True,
        attn_sharing=True,
        simple_decoder=simple_decoder,
        debugging=True,
    )

    # Resume from last checkpoint
    ckpt_path = None
    previous_checkpoints = list(Path(checkpoint_path).rglob("*.ckpt"))
    if len(previous_checkpoints) > 0:
        ckpt_path = max(previous_checkpoints, key=lambda x: x.stat().st_ctime)
        model.load_from_checkpoint(ckpt_path)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
