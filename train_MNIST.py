# import encodex
# from encodex.utils.imports import *

### DO NOT CHANGE ###
### This version is created to test the conda environment and CUDA compatibility.
### Current Status: CUDA works.
### Feature addition: Custom Checkpoint
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from IPython.core.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger

from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST

PATH_DATASETS = os.environ.get("PATH_DATASETS", "../DATA/")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64


class MNISTModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

def run(epoch=3):
    # Create WandB Logger
    wandb_logger = WandbLogger(name="test_model", project="pd", log_model=False)

    # Add Model Checkpointing
    logger = wandb_logger
    # model_checkpoint = pl.callbacks.ModelCheckpoint(dirpath=logger.experiment.project + '/checkpoints/',
    #                                filename='{epoch}-{val_loss:.2f}',
    #                                monitor='val_loss',
    #                                save_top_k=3,
    #                                verbose=True)

    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    train_ds = MNIST(
        PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor()
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    # Initialize a trainer

    trainer = Trainer(
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=epoch,
        callbacks=[TQDMProgressBar(refresh_rate=20)],  # model_checkpoint],
        logger=wandb_logger,
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)

    # Best Model Path
    # print(f"Best Model was saved at: {model_checkpoint.best_model_path}")
    # Close W&B Logger
    wandb_logger.watch(mnist_model, log="all")
    wandb.finish()

if __name__ == "__main__":
    run(30)