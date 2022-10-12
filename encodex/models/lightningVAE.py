from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
from .lightningAE import *

# def gaussian_kernel(a, b):
#    dim1_1, dim1_2 = a.shape[0], b.shape[0]
#    depth = a.shape[1]
#    a = a.view(dim1_1, 1, depth)
#    b = b.view(1, dim1_2, depth)
#    a_core = a.expand(dim1_1, dim1_2, depth)
#    b_core = b.expand(dim1_1, dim1_2, depth)
#    numerator = (a_core - b_core).pow(2).mean(2)/depth
#    return torch.exp(-numerator)
#
# def MMD(a, b):
#    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()
#
# def loss_function(pred, true, latent):
#    return (pred-true).pow(2).mean(), MMD(torch.randn(200, self.hparams.latent_size, requires_grad = False).to(DEVICE), latent)
#
#


class Reshape(nn.Module):
    """
    Not in Use Currently. Used in a nn.Sequential pipeline to reshape on the fly.
    """

    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class Flatten(nn.Module):
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full

    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), 32, -1)


class VaeMmdConv3Lin1(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
            ),  # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1),  # Output: #bs, 8, 98
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1
            ),  # Output: b, 16, 50
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1
            ),  # Output: b, 32, 10
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),  # Output: b, 32, 3
            Flatten(),
            nn.Linear(in_features=3 * 32, out_features=self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=self.hparams.hidden_layer_1_dim,
                out_features=hparams.latent_size,
            ),
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=hparams.latent_size,
                out_features=self.hparams.hidden_layer_1_dim,
            ),
            nn.ReLU(),
            nn.Linear(in_features=self.hparams.hidden_layer_1_dim, out_features=3 * 32),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, 6, stride=2),  # b, 16, 10
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
        )

    def forward(self, X):
        # if self.training:
        latent = self.encoder(X)

        return self.decoder(nn.Tanh()(latent)), latent  ## Added Tanh
        # else:
        #    return self.decoder( self.encoder(X) )

    def prepare_data(self):
        train_dataset = XDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([ToTensor()]),
        )
        self.train_size = int(0.98 * len(train_dataset))
        self.val_size = len(train_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            train_dataset, [self.train_size, self.val_size]
        )
        if self.hparams.test_json_file:
            self.test_ds = XDataset(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                transform=Compose([ToTensor()]),
            )

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

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
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
        reconstruction, mu = self.forward(x)
        reconstruction_error, mmd = self.loss_function(reconstruction, x, mu)
        # loss = nn.MSELoss()(reconstructed, x)#, reduction='mean')##Removing reduce here...
        # loss = nn.MSELoss()(y_, y)#, reduction='mean')##Removing reduce here...
        loss = reconstruction_error + mmd

        tensorboard_logs = {
            "train_loss": loss,
            "train_recon_loss": reconstruction_error,
            "train_mmd_loss": mmd,
        }
        # print(tensorboard_logs)
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        train_losses = torch.stack([x["train_loss"] for x in outputs])
        train_avg_loss = train_losses.detach().mean()
        train_std_loss = train_losses.detach().std()
        train_recon_loss_epoch = (
            torch.stack([x["train_recon_loss"] for x in outputs]).detach().mean()
        )
        train_mmd_loss_epoch = (
            torch.stack([x["train_mmd_loss"] for x in outputs]).detach().mean()
        )

        tensorboard_logs = {
            "train_epoch_loss": train_avg_loss,
            "training_loss_std": train_std_loss,
            "normalized_train_loss_std": train_std_loss / train_avg_loss,
            "train_recon_loss_per_epoch": train_recon_loss_epoch,
            "train_mmd_loss_per_epoch": train_mmd_loss_epoch,
            "step": self.current_epoch + 1,
        }
        return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        # reconstructed = self(x)
        # loss = nn.MSELoss()(reconstructed, x)

        reconstruction, mu = self.forward(x)
        val_reconstruction_error, mmd = self.loss_function(reconstruction, x, mu)
        loss = val_reconstruction_error + mmd
        # add logs
        logs = {"loss": {"val_loss": loss}}
        return {
            "val_loss": loss,
            "val_recon_loss": val_reconstruction_error,
            "val_mmd_loss": mmd,
            "log": logs,
        }

    def validation_epoch_end(self, outputs):
        # print(outputs)
        # pdb.set_trace()
        val_losses = torch.stack([x["val_loss"] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        val_recon_loss_epoch = torch.stack(
            [x["val_recon_loss"] for x in outputs]
        ).mean()
        val_mmd_loss_epoch = torch.stack([x["val_mmd_loss"] for x in outputs]).mean()
        tensorboard_logs = {
            "val_loss": val_avg_loss,
            "val_loss_std": val_std_loss,
            "normalized_val_loss_std": val_std_loss / val_avg_loss,
            "val_recon_loss_per_epoch": val_recon_loss_epoch,
            "val_mmd_loss_per_epoch": val_mmd_loss_epoch,
            "step": self.current_epoch + 1,
        }
        return {"avg_val_loss": val_avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        """
        Lightning calls this during testing, similar to `validation_step`,
        with the data from the test dataloader passed in as `batch`.
        """
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        # output['test_loss'] = output.pop('val_loss')

        return output

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_epoch_end(outputs)
        # rename some keys
        results["log"].update(
            {
                "test_loss": results["log"].pop("val_loss"),
                "test_loss_std": results["log"].pop("val_loss_std"),
                "normalized_test_loss_std": results["log"].pop(
                    "normalized_val_loss_std"
                ),
                "test_recon_loss_per_epoch": results["log"].pop(
                    "val_recon_loss_per_epoch"
                ),
                "test_mmd_loss_per_epoch": results["log"].pop("val_mmd_loss_per_epoch"),
            }
        )
        results["avg_test_loss"] = results.pop("avg_val_loss")

        return results

    def gaussian_kernel(self, a, b):

        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)  # .cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)  # .cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        # a_core = a_core.to(self.device)
        # b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2) / depth

        return torch.exp(-numerator)

    def MMD(self, a, b):
        return (
            self.gaussian_kernel(a, a).mean()
            + self.gaussian_kernel(b, b).mean()
            - 2 * self.gaussian_kernel(a, b).mean()
        )

    def L1_regularization(self, input):
        loss = 0
        # values = torch.from_numpy(input)#.float()
        values = input
        for l in self.encoder.children():
            values = nn.LeakyReLU()(l(values))
            loss += torch.mean(abs(values))
        values = nn.Tanh()(values)
        for l in self.decoder.children():
            values = nn.ReLU()(l(values))
            loss += torch.mean(abs(values))
        return loss

    def loss_function(self, pred, true, latent):
        if self.hparams.sparsity:
            return (pred - true).pow(2).mean(), (self.hparams.reg_para) * (
                self.L1_regularization(true)
            )
        else:
            return (pred - true).pow(2).mean(), self.hparams.reg_para * self.MMD(
                a=torch.randn(
                    self.hparams.batch_size,
                    self.hparams.latent_size,
                    requires_grad=False,
                ),
                b=latent,
            )
        # mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        # mmd_loss = self.compute_mmd(a=true, b=latent)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
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
        # parser.add_argument('--reg_para', default=0.01, type=float)
        # parser.add_argument('--sparsity', default=False, type=bool)

        return parser


class VaeMmdLin1(VaeMmdConv3Lin1):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_size, self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_layer_1_dim, 10 * 10),
        )

    def forward(self, X):
        # if self.training:
        X = X.view(-1, 100)
        # print('Input Shape:', X.shape)
        latent = self.encoder(X)
        return self.decoder(latent), latent

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
        # parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class VaeMmdLin2(VaeMmdConv3Lin1):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_size, self.hparams.hidden_layer_2_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_layer_2_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_layer_1_dim, 10 * 10),
        )

    def forward(self, X):
        # if self.training:
        X = X.view(-1, 100)
        # print('Input Shape:', X.shape)
        latent = self.encoder(X)
        return self.decoder(latent), latent

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
        # parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class VaeMmdConv3Lin1XAS(VaeMmdConv3Lin1):
    def __init__(self, hparams):
        super().__init__(hparams)

    def prepare_data(self):
        train_dataset = XasDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
        )  ##[
        self.train_size = int(0.98 * len(train_dataset))
        self.val_size = len(train_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            train_dataset, [self.train_size, self.val_size]
        )
        if self.hparams.test_json_file:
            self.test_ds = XasDataset(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
            )  ##[Normalize(),

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
        # parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class VaeMmdLin2XAS(VaeMmdConv3Lin1):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hparams.latent_size, self.hparams.hidden_layer_2_dim),
            nn.ReLU(),
            nn.Linear(hparams.hidden_layer_2_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_layer_1_dim, 10 * 10),
        )

    def forward(self, X):
        # if self.training:
        X = X.view(-1, 100)
        # print('Input Shape:', X.shape)
        latent = self.encoder(X)
        return self.decoder(nn.Tanh()(latent)), latent

    def prepare_data(self):
        train_dataset = XasDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([XasInterpolate(), XasToTensor()]),
        )  ##[Normalize(),
        self.train_size = int(0.98 * len(train_dataset))
        self.val_size = len(train_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            train_dataset, [self.train_size, self.val_size]
        )
        if self.hparams.test_json_file:
            self.test_ds = XasDataset(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                transform=Compose([XasInterpolate(), XasToTensor()]),
            )  ##[Normalize(),

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument("--hidden_layer_1_dim", default=128, type=int)
        parser.add_argument("--hidden_layer_2_dim", default=64, type=int)
        # parser.add_argument('--hidden_layer_3_dim', default=12, type=int)
        # parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        # parser.add_argument('--reg_para', default=0.01, type=float)
        # parser.add_argument('--sparsity', default=True, type=bool)
        return parser


class Lin2AEBNXas(VaeMmdConv3Lin1XAS):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            # nn.Dropout(self.hparams.dropout),
            # nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            # nn.ReLU(True),
            # nn.BatchNorm1d(1),
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.latent_size, self.hparams.hidden_layer_2_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            # nn.Dropout(self.hparams.dropout),
            # nn.Linear(self.hparams.hidden_layer_3_dim, self.hparams.hidden_layer_2_dim),
            # nn.ReLU(True),
            # nn.BatchNorm1d(1),
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(1),
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.input_dim),
        )

    def forward(self, X):
        # if self.training:
        # X=X.view(-1,100)
        X = X.view(-1, 1, self.hparams.input_dim)
        # print('Input Shape:', X.shape)
        latent = self.encoder(X)
        return self.decoder(latent), latent

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed, _ = self.forward(x)
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
        reconstructed, _ = self(x)
        loss = nn.MSELoss()(reconstructed, x)
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

    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_epoch_end(outputs)
        # rename some keys
        results["log"].update(
            {
                "test_loss": results["log"].pop("val_loss"),
                "test_loss_std": results["log"].pop("val_loss_std"),
                "normalized_test_loss_std": results["log"].pop(
                    "normalized_val_loss_std"
                ),
                #'test_recon_loss_per_epoch': results['log'].pop('val_recon_loss_per_epoch'),
                #'test_mmd_loss_per_epoch': results['log'].pop('val_mmd_loss_per_epoch'),
            }
        )
        results["avg_test_loss"] = results.pop("avg_val_loss")

        return results

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
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class VaeMmdConv3Lin0XAS(VaeMmdConv3Lin1):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
            ),  # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.ReLU(),  # nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1),  # Output: #bs, 8, 98
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1
            ),  # Output: b, 16, 50
            nn.ReLU(),  # nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1
            ),  # Output: b, 32, 10
            nn.ReLU(),  # nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),  # Output: b, 32, 3
            Flatten(),
            nn.ReLU(),  # nn.LeakyReLU(),
            nn.Linear(in_features=32 * 3, out_features=hparams.latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hparams.latent_size, out_features=32 * 3),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, 6, stride=2),  # b, 16, 10
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
        )

    def forward(self, X):
        # if self.training:
        latent = self.encoder(X)
        return self.decoder(nn.Tanh()(latent)), latent  ## Added Tanh

    def prepare_data(self):
        train_dataset = XasDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([XasInterpolate(), XasToTensor()]),
        )  ##[Normalize(),
        self.train_size = int(0.98 * len(train_dataset))
        self.val_size = len(train_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            train_dataset, [self.train_size, self.val_size]
        )
        if self.hparams.test_json_file:
            self.test_ds = XasDataset(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                transform=Compose([XasInterpolate(), XasToTensor()]),
            )  ##[Normalize(),

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
        # parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser


class VaeMmdConv3Lin2XAS(VaeMmdConv3Lin1XAS):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
            ),  # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1),  # Output: #bs, 8, 98
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1
            ),  # Output: b, 16, 50
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1
            ),  # Output: b, 32, 10
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),  # Output: b, 32, 3
            Flatten(),
            nn.Linear(in_features=3 * 32, out_features=self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=self.hparams.hidden_layer_1_dim,
                out_features=self.hparams.hidden_layer_2_dim,
            ),
            nn.LeakyReLU(),
            nn.Linear(
                in_features=self.hparams.hidden_layer_2_dim,
                out_features=hparams.latent_size,
            ),
        )
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=hparams.latent_size,
                out_features=self.hparams.hidden_layer_2_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hparams.hidden_layer_2_dim,
                out_features=self.hparams.hidden_layer_1_dim,
            ),
            nn.ReLU(),
            nn.Linear(in_features=self.hparams.hidden_layer_1_dim, out_features=3 * 32),
            nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, 6, stride=2),  # b, 16, 10
            nn.ReLU(True),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
        )

    def forward(self, X):
        # if self.training:
        latent = self.encoder(X)
        return self.decoder(nn.Tanh()(latent)), latent  ## Added Tanh

    def forward(self, X):
        # if self.training:
        latent = self.encoder(X)
        return self.decoder(nn.Tanh()(latent)), latent  ## Added Tanh

    def prepare_data(self):
        train_dataset = XasDataset(
            json_file=self.hparams.json_file,
            root_dir="../data/",
            transform=Compose([XasInterpolate(), XasToTensor()]),
        )  ##[Normalize(),
        self.train_size = int(0.98 * len(train_dataset))
        self.val_size = len(train_dataset) - self.train_size
        self.train_ds, self.val_ds = random_split(
            train_dataset, [self.train_size, self.val_size]
        )
        if self.hparams.test_json_file:
            self.test_ds = XasDataset(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                transform=Compose([XasInterpolate(), XasToTensor()]),
            )  ##[Normalize(),

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
        # parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL
        # parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        # parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        # parser.add_argument('--learning_rate', default=0.001, type=float)
        # parser.add_argument('--batch_size', default=32, type=int)
        # parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument("--latent_size", default=3, type=int)
        # training specific (for this model)
        return parser
