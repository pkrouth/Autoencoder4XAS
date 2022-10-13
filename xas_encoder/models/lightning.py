from ..utils import *
from ..xasdata import *
from ..utils.transforms import *


class XasLinEncoders(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(hparams)
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 10 * 10),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        transformed_dataset = XDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([Normalize(), ToTensor()]),
        )  ##off
        self.train_size = int(0.98 * len(transformed_dataset))
        self.val_size = len(transformed_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            transformed_dataset, [self.train_size, self.val_size]
        )
        # self.train_ds, self.val_ds = self.train_ds.float(), self.val_ds.float()

    # #@pl.data_loader
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.cpus,
    #     )

    # #@pl.data_loader
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.cpus,
    #     )

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed = self.forward(x)
        loss = nn.MSELoss()(
            reconstructed, x
        )  # , reduction='mean')##Removing reduce here...
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_losses = torch.stack([x["train_loss"] for x in outputs])
        train_avg_loss = train_losses.mean()
        train_std_loss = train_losses.std()
        tensorboard_logs = {
            "train_epoch_loss": train_avg_loss,
            "training_loss_std": train_std_loss,
            "normalized_train_loss_std": train_std_loss / train_avg_loss,
            "step": self.current_epoch + 1,
        }
        return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed = self(x)
        loss = F.mse_loss(reconstructed, x, reduction="mean")
        # add logs
        logs = {"loss": {"val_loss": loss}}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # pdb.set_trace()
        val_losses = torch.stack([x["val_loss"] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        tensorboard_logs = {
            "val_loss": val_avg_loss,
            "val_loss_std": val_std_loss,
            "normalized_val_loss_std": val_std_loss / val_avg_loss,
            "step": self.current_epoch + 1,
        }
        return {"avg_val_loss": val_avg_loss, "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        # parser = ArgumentParser(parents=[parent_parser])
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        # parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


## Flatten and UnFlatten Layers
class Flatten(nn.Module):
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 32, -1)


class XasConvAutoEncoder(XasLinEncoders):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters()
        self.hparams.update(hparams)
        # self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=2
            ),  # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=1),  # Output: #bs, 8, 98
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1
            ),  # Output: b, 16, 50
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1
            ),  # Output: b, 32, 11
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=3),  # Output: b, 32, 3
            Flatten(full=False),
            nn.Linear(96, self.hparams.latent_size),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 32 * 3),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=2),  # b, 16, 10
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class XasConvAutoEncoder2(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(hparams)
        # self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=3, stride=2, padding=1
            ),  # Input: (bs,nc=1,l=100) Output: (bs,8,50)
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=1),  # Output: #bs, 8, 48
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1
            ),  # Output: b, 16, 25
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 12
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1
            ),  # Output: b, 32, 5
            nn.ReLU(True),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 32, 2
            Flatten(full=False),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            nn.Linear(64, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 32 * 2),
            nn.ReLU(True),
            nn.Linear(64, 64),
            nn.ReLU(True),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, kernel_size=6, stride=1, padding=0),  # b, 16, 10
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 10, stride=3, padding=0),  # b, 1, 100
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        transformed_dataset = XDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([Normalize(), ToTensor()]),
        )  ##off
        self.train_size = int(0.98 * len(transformed_dataset))
        self.val_size = len(transformed_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            transformed_dataset, [self.train_size, self.val_size]
        )
        # self.train_ds, self.val_ds = self.train_ds.float(), self.val_ds.float()

    # @pl.data_loader
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.cpus,
    #     )

    # @pl.data_loader
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.cpus,
    #     )

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed = self.forward(x)
        loss = nn.L1Loss()(
            reconstructed, x
        )  # , reduction='mean')##Removing reduce here...
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_losses = torch.stack([x["train_loss"] for x in outputs])
        train_avg_loss = train_losses.mean()
        train_std_loss = train_losses.std()
        tensorboard_logs = {
            "train_epoch_loss": train_avg_loss,
            "training_loss_std": train_std_loss,
            "normalized_train_loss_std": train_std_loss / train_avg_loss,
            "step": self.current_epoch + 1,
        }
        return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed = self(x)
        loss = nn.L1Loss()(reconstructed, x)
        # add logs
        logs = {"loss": {"val_loss": loss}}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # pdb.set_trace()
        val_losses = torch.stack([x["val_loss"] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        tensorboard_logs = {
            "val_loss": val_avg_loss,
            "val_loss_std": val_std_loss,
            "normalized_val_loss_std": val_std_loss / val_avg_loss,
            "step": self.current_epoch + 1,
        }
        return {"avg_val_loss": val_avg_loss, "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class XasLinAutoEncoder2(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.update(hparams)
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hparams.latent_size, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 10 * 10),
        )  ## Removing tanH from the last layer

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        transformed_dataset = XDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([Normalize(), ToTensor()]),
        )  ##off
        self.train_size = int(0.98 * len(transformed_dataset))
        self.val_size = len(transformed_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            transformed_dataset, [self.train_size, self.val_size]
        )
        # self.train_ds, self.val_ds = self.train_ds.float(), self.val_ds.float()

    # @pl.data_loader
    # def train_dataloader(self):
    #     return DataLoader(
    #         self.train_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.cpus,
    #     )

    # @pl.data_loader
    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_ds,
    #         batch_size=self.hparams.batch_size,
    #         num_workers=self.hparams.cpus,
    #     )

    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed = self.forward(x)
        loss = nn.L1Loss()(
            reconstructed, x
        )  # , reduction='mean')##Removing reduce here...
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_losses = torch.stack([x["train_loss"] for x in outputs])
        train_avg_loss = train_losses.mean()
        train_std_loss = train_losses.std()
        tensorboard_logs = {
            "train_epoch_loss": train_avg_loss,
            "training_loss_std": train_std_loss,
            "normalized_train_loss_std": train_std_loss / train_avg_loss,
            "step": self.current_epoch + 1,
        }
        return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed = self(x)
        loss = nn.L1Loss()(reconstructed, x)
        # add logs
        logs = {"loss": {"val_loss": loss}}
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # pdb.set_trace()
        val_losses = torch.stack([x["val_loss"] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        tensorboard_logs = {
            "val_loss": val_avg_loss,
            "val_loss_std": val_std_loss,
            "normalized_val_loss_std": val_std_loss / val_avg_loss,
            "step": self.current_epoch + 1,
        }
        return {"avg_val_loss": val_avg_loss, "log": tensorboard_logs}

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser
