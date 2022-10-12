from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout


class SS2AE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_size, self.hparams.hidden_layer_2_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            nn.Linear(self.hparams.hidden_layer_1_dim, 2 * 2),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.hparams.input_dim)
        x = self.encoder(x)
        x = nn.Tanh()(x)
        x = self.decoder(x)
        return x

    def prepare_data(self):
        transformed_dataset = XDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([ToTensor()]),
        )
        self.train_size = int(0.98 * len(transformed_dataset))
        self.val_size = len(transformed_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            transformed_dataset, [self.train_size, self.val_size]
        )
        # self.train_ds, self.val_ds = self.train_ds.float(), self.val_ds.float()

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.cpus,
            drop_last=False,
        )

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        # reconstructed = self.forward(x)
        y_ = self.forward(x)
        # loss = nn.MSELoss()(reconstructed, x)#, reduction='mean')##Removing reduce here...
        loss = nn.MSELoss()(y_, y)  # , reduction='mean')##Removing reduce here...
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
        y = y.float()
        # reconstructed = self(x)
        # loss = nn.MSELoss()(reconstructed, x)
        y_ = self(x)
        loss = nn.MSELoss()(y_, y)
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
        parser.add_argument("--hidden_layer_1_dim", default=128, type=int)
        parser.add_argument("--hidden_layer_2_dim", default=64, type=int)
        parser.add_argument("--hidden_layer_3_dim", default=12, type=int)
        parser.add_argument("--input_dim", default=100, type=int)

        ## CONV MODEL
        parser.add_argument("--fc_layer_1_dim", default=64, type=int)
        parser.add_argument("--dropout", default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class SS1AE(SS2AE):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_size, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            nn.Linear(self.hparams.hidden_layer_1_dim, 2 * 2),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.hparams.input_dim)
        x = self.encoder(x)
        x = nn.Tanh()(x)
        x = self.decoder(x)
        return x


class SS1AE2(SS2AE):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_size, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.Linear(self.hparams.hidden_layer_1_dim, 2 * 2),
        )

    def forward(self, x):
        x = x.view(-1, 1, self.hparams.input_dim)
        x = self.encoder(x)
        x = nn.Tanh()(x)
        x = self.decoder(x)
        return x
