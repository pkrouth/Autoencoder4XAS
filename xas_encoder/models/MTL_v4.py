from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
from .lightningVAE import *
from .lightningAE import *
from .lightningXAS import *
from .MTL import *
from .MTLv3 import *

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

# class UnFlatten(nn.Module):
#    def forward(self,x):
#        return x.view(x.size(0),32,-1)


## Inspirations from https://www.kaggle.com/purplejester/pytorch-deep-time-series-classification


## Cosine Scheduler


def cosine(epoch, t_max, ampl):

    t = epoch % t_max
    return (1 + np.cos(np.pi * t / t_max)) * ampl / 2


def inv_cosine(epoch, t_max, ampl):

    return 1 - cosine(epoch, t_max, ampl)


def one_cycle(epoch, t_max, a1=0.6, a2=1.0, pivot=0.3):
    pct = epoch / t_max
    if pct < pivot:
        return inv_cosine(epoch, pivot * t_max, a1)
    return cosine(epoch - pivot * t_max, (1 - pivot) * t_max, a2)


class Scheduler:
    """Updates' optimizer's learning rates using provided scheduling function."""

    def __init__(self, opt, schedule):
        self.opt = opt
        self.schedule = schedule
        self.history = defaultdict(list)

    def step(self, t):
        for i, group in enumerate(self.opt.param_group):
            lr = opt.defaults["lr"] * self.schedule(t)
            group["lr"] = lr
            self.history[i].append(lr)


class MultiTaskLossWrapper_v9(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v9, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams

    def forward(self, preds, y, x):
        # [recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce="none"), nn.CrossEntropyLoss()

        # sages = (age*4.75).exp_()

        loss_CN = mse(preds[2], y[:, 0:6]).mean()
        loss_Dist = mse(preds[3], y[:, 6:12]).mean()
        loss_H_fraction = mse(preds[4], y[:, 13]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:, 12].long()).mean()

        precision_CN = torch.exp(-self.log_vars[0])
        # loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)
        loss_CN = torch.sum(
            precision_CN * loss_CN + torch.log(1 + self.log_vars[0].exp()), -1
        )

        precision_Dist = torch.exp(-self.log_vars[1])
        # loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)
        loss_Dist = torch.sum(
            precision_Dist * loss_Dist + torch.log(1 + self.log_vars[1].exp()), -1
        )

        precision_H_fraction = torch.exp(-self.log_vars[2])
        # loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        loss_H_fraction = torch.sum(
            precision_H_fraction * loss_H_fraction
            + torch.log(1 + self.log_vars[2].exp()),
            -1,
        )

        precision_Crystal = torch.exp(-self.log_vars[3])
        # loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        loss_Crystal = torch.sum(
            precision_Crystal * loss_Crystal + torch.log(1 + self.log_vars[3].exp()), -1
        )

        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_mmd = self.loss_function(
            pred=reconstruction, true=x, latent=mu
        )

        # precision_recon = torch.exp(-self.log_vars[4])
        # loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        # loss_reconstruction = torch.sum(precision_recon*loss_reconstruction + torch.log(1+self.log_vars[4].exp()),-1)

        ## MMD Loss ##
        # precision_mmd = torch.exp(-self.log_vars[5])
        # loss_mmd = torch.sum(precision_mmd*loss_mmd+self.log_vars[5],-1)
        # loss_mmd = torch.sum(precision_mmd*loss_mmd + torch.log(1+self.log_vars[5].exp()),-1)

        ## Reconstruction + 0.01 MMD Loss ##
        # wtloss_reconstruction_mmd = torch.sum(precision_recon*(loss_reconstruction+self.hparams.reg_para*loss_mmd) + torch.log(1+self.log_vars[4].exp()),-1)

        ## Main Task - [Reconstruction + 0.01 MMD] Loss ##
        loss_L1_mu = self.L1_regularization(preds[1])
        wtloss_reconstruction_mmd = 1000 * (
            loss_reconstruction
            + self.hparams.mmd_para * loss_mmd
            + self.hparams.reg_para * loss_L1_mu
        )

        ## loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal

        return (
            wtloss_reconstruction_mmd
            + loss_CN
            + loss_Dist
            + loss_H_fraction
            + loss_Crystal,
            {
                "loss_CN": loss_CN.detach(),
                "loss_Distances": loss_Dist.detach(),
                "loss_H_fraction": loss_H_fraction.detach(),
                "loss_Crystal": loss_Crystal.detach(),
                "loss_weights_CN": self.log_vars[0].detach(),
                "loss_weights_Distances": self.log_vars[1].detach(),
                "loss_weights_Crystal": self.log_vars[3].detach(),
                "loss_weights_H_fraction": self.log_vars[2].detach(),
                "train_recon_loss": loss_reconstruction.detach(),
                "loss_weight_recon": 1000,
                "train_mmd_loss": loss_mmd.detach(),
                "loss_weight_MMD": 10,  ### Modified
                "train_loss_recon_mmd_wt": wtloss_reconstruction_mmd.detach(),  ## Combined weighted Recon+MMD Loss
            },
        )

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

    def loss_function(self, pred, true, latent):
        return (pred - true).pow(2).mean(), self.MMD(
            a=torch.randn(
                self.hparams.batch_size, self.hparams.latent_size, requires_grad=False
            ),
            b=latent,
        )

    # mse_loss = nn.MSELoss(reduce='mean')(pred, true)
    # mmd_loss = self.compute_mmd(a=true, b=latent)

    def L1_regularization(self, mu):
        return abs(mu).mean()


class MultiTaskLossWrapper_v10(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v10, self).__init__()
        self.task_num = task_num
        # self.log_vars = nn.Parameter(torch.ones((task_num)))## Changed from 0 to 1
        self.hparams = hparams

    def forward(self, preds, y, x):
        # [recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce="none"), nn.CrossEntropyLoss()

        # sages = (age*4.75).exp_()

        loss_CN = mse(preds[2], y[:, 0:6]).mean()
        loss_Dist = mse(preds[3], y[:, 6:12]).mean()
        loss_H_fraction = mse(preds[4], y[:, 13]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:, 12].long()).mean()

        ## Main Task - [Reconstruction + 0.01 MMD] Loss ##

        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_mmd = self.loss_function(
            pred=reconstruction, true=x, latent=mu
        )

        loss_L1_mu = self.L1_regularization(preds[1])
        wtloss_reconstruction_mmd = 1000 * (
            loss_reconstruction
            + self.hparams.mmd_para * loss_mmd
            + self.hparams.reg_para * loss_L1_mu
        )

        ## loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal

        return (
            wtloss_reconstruction_mmd
            + loss_CN
            + loss_Dist
            + loss_H_fraction
            + loss_Crystal,
            {
                "loss_CN": loss_CN.detach(),
                "loss_Distances": loss_Dist.detach(),
                "loss_H_fraction": loss_H_fraction.detach(),
                "loss_Crystal": loss_Crystal.detach(),
                "train_recon_loss": loss_reconstruction.detach(),
                "loss_weight_recon": 1000,
                "train_mmd_loss": loss_mmd.detach(),
                "train_loss_recon_mmd_wt": wtloss_reconstruction_mmd.detach(),  ## Combined weighted Recon+MMD Loss
            },
        )

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

    def loss_function(self, pred, true, latent):
        return (pred - true).pow(2).mean(), self.MMD(
            a=torch.randn(
                self.hparams.batch_size, self.hparams.latent_size, requires_grad=False
            ),
            b=latent,
        )

    # mse_loss = nn.MSELoss(reduce='mean')(pred, true)
    # mmd_loss = self.compute_mmd(a=true, b=latent)

    def L1_regularization(self, mu):
        return abs(mu).mean()


class MTLModelv12(MTLModelv10):
    """
    Changes from v5:
    BatchNorm from encoder and decoder layers ARE BACK.
    SOFTMAX IN CRYSTAL
    LOSS_V3: ALL IN AUTO ..LEADS TO NEGATIVE VALS.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        # self.save_parameters()
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp(
            self.hparams.latent_size, "CN", hidden_layers=2
        ).create()
        self.task_Dist = create_mlp(
            self.hparams.latent_size, "Distances", hidden_layers=2
        ).create()
        self.task_H2 = create_mlp(
            self.hparams.latent_size, "H_fraction", hidden_layers=2
        ).create()
        self.task_Crystal = create_classifier(
            self.hparams.latent_size, "Crystal", hidden_layers=2, out_features=26
        ).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v9(4, self.hparams)

    def forward(self, X):
        X = X.view(X.size(0), 1, -1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_)
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent)
        H2_hat = self.task_H2(latent)
        Crystal_hat = self.task_Crystal(latent)

        # import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_dim", type=int, default=100)
        # parser.add_argument('--bias', default='store_true')
        parser.add_argument("--hidden_layer_1_dim", type=int, default=500)

        return parser


class MTLModelv13(MTLModelv10):
    """
    Changes from v5:
    BatchNorm from encoder and decoder layers ARE BACK.
    SOFTMAX IN CRYSTAL
    LOSS_V3: ALL IN AUTO ..LEADS TO NEGATIVE VALS.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        # self.save_parameters()
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp(
            self.hparams.latent_size, "CN", hidden_layers=2, dropout_first=0.2
        ).create()
        self.task_Dist = create_mlp(
            self.hparams.latent_size, "Distances", hidden_layers=2, dropout_first=0.2
        ).create()
        self.task_H2 = create_mlp(
            self.hparams.latent_size, "H_fraction", hidden_layers=2, dropout_first=0.2
        ).create()
        self.task_Crystal = create_classifier(
            self.hparams.latent_size,
            "Crystal",
            hidden_layers=2,
            out_features=26,
            dropout_first=0.2,
            dropout_last=0.2,
        ).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v10(4, self.hparams)

    def forward(self, X):
        X = X.view(X.size(0), 1, -1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_)
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent)
        H2_hat = self.task_H2(latent)
        Crystal_hat = self.task_Crystal(latent)

        # import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_dim", type=int, default=100)
        # parser.add_argument('--bias', default='store_true')
        parser.add_argument("--hidden_layer_1_dim", type=int, default=500)

        return parser


class MTLModelv14(MTLModelv10):
    """
    Changes from v5:
    BatchNorm from encoder and decoder layers ARE BACK.
    SOFTMAX IN CRYSTAL
    LOSS_V3: ALL IN AUTO ..LEADS TO NEGATIVE VALS.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        # self.save_parameters()
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp_v5(
            self.hparams.latent_size, "CN", hidden_layers=2, hidden_features=512
        ).create()
        self.task_Dist = create_mlp_v5(
            self.hparams.latent_size, "Distances", hidden_layers=2, hidden_features=512
        ).create()
        self.task_H2 = create_mlp_v5(
            self.hparams.latent_size, "H_fraction", hidden_layers=2, hidden_features=512
        ).create()
        self.task_Crystal = create_classifier_v5(
            self.hparams.latent_size,
            "Crystal",
            hidden_layers=2,
            hidden_features=512,
            out_features=26,
        ).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v10(4, self.hparams)

    def forward(self, X):
        X = X.view(X.size(0), 1, -1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_)
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent)
        H2_hat = self.task_H2(latent)
        Crystal_hat = self.task_Crystal(latent)

        # import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

    def setup(self, stage):
        self.train_ds = XasMultiTaskDatasetv2(
            json_file=self.hparams.train_json_file,
            root_dir="../data/",
            descriptor="Distances",
            MTL=True,
            scalers=None,
            scale_min_max=True,
            cat_list=None,
            transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
        )
        self.train_size = len(self.train_ds)

        if self.hparams.val_json_file is not None:
            self.val_ds = XasMultiTaskDatasetv2(
                json_file=self.hparams.val_json_file,
                root_dir="../data/",
                descriptor="Distances",
                MTL=True,
                scalers=self.train_ds.scalers,
                scale_min_max=True,
                cat_list=self.train_ds.categories,
                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
            )
        else:
            self.train_size = int(0.98 * len(self.train_ds))
            self.val_size = len(self.train_ds) - self.train_size
            self.train_ds, self.val_ds = random_split(
                self.train_ds, [self.train_size, self.val_size]
            )

        if self.hparams.test_json_file:
            self.test_ds = XasMultiTaskDatasetv2(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                descriptor="Distances",
                MTL=True,
                scalers=self.train_ds.scalers,
                scale_min_max=True,
                cat_list=self.train_ds.categories,
                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
            )

        exp_dataset = XasExpDataset(
            json_file=self.hparams.exp_json_file,
            root_dir="../data/",
            descriptor="Distances",
            transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
        )

        self.exp_ds = exp_dataset

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 10, 1
        )

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )

    @pl.data_loader
    def val_dataloader(self):
        theo_DL = DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )
        exp_DL = DataLoader(
            self.exp_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )
        return [theo_DL, exp_DL]

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_dim", type=int, default=100)
        # parser.add_argument('--bias', default='store_true')
        parser.add_argument("--hidden_layer_1_dim", type=int, default=500)

        return parser


class create_mlp_v5(nn.Module):
    def __init__(
        self,
        input_size,
        descriptor,
        hidden_layers=1,
        hidden_features=512,
        dropout_first=None,
        dropout_last=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.descriptor = descriptor
        self.hidden_layers = hidden_layers
        self.dropout_first = dropout_first
        self.dropout_last = dropout_last

        if self.descriptor in ["CN", "Distances"]:
            self.out_features = 6

        elif self.descriptor == "H_fraction":
            self.out_features = 1
        elif self.descriptor == "Crystal":
            self.out_features = 25

        if self.hidden_layers == 1:
            self.layers = nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=self.input_size, out_features=hidden_features
                    ),
                    nn.ReLU(),
                    torch.nn.Linear(
                        in_features=hidden_features, out_features=self.out_features
                    ),
                ]
            )
        elif self.hidden_layers == 2:
            self.layers = nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=self.input_size, out_features=hidden_features
                    ),
                    nn.ReLU(),
                    torch.nn.Linear(
                        in_features=hidden_features, out_features=hidden_features
                    ),
                    nn.ReLU(),
                    torch.nn.Linear(
                        in_features=hidden_features, out_features=self.out_features
                    ),
                ]
            )

        ## Add Dropout before the network
        if self.dropout_first is not None:
            self.layers.insert(0, nn.Dropout(self.dropout_first))

        ## Add Dropout before the network
        if self.dropout_last is not None:
            self.layers.append(nn.Dropout(self.dropout_last))

        self.task_network = nn.Sequential(*self.layers)

    def create(self):
        return self.task_network

    def forward(self, X):
        return self.task_network(X)


class create_classifier_v5(nn.Module):
    def __init__(
        self,
        input_size,
        descriptor,
        hidden_layers=1,
        hidden_features=512,
        out_features=25,
        dropout_first=None,
        dropout_last=None,
    ):
        super().__init__()
        self.input_size = input_size
        self.descriptor = descriptor
        self.hidden_layers = hidden_layers
        self.dropout_first = dropout_first
        self.dropout_last = dropout_last

        if self.descriptor == "Crystal":
            self.out_features = out_features

        if self.hidden_layers == 1:
            self.layers = nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=self.input_size, out_features=hidden_features
                    ),
                    nn.ReLU(),
                    torch.nn.Linear(
                        in_features=hidden_features, out_features=self.out_features
                    ),
                    nn.Softmax(),
                ]
            )
        elif self.hidden_layers == 2:
            self.layers = nn.ModuleList(
                [
                    torch.nn.Linear(
                        in_features=self.input_size, out_features=hidden_features
                    ),
                    nn.ReLU(),
                    torch.nn.Linear(
                        in_features=hidden_features, out_features=hidden_features
                    ),
                    nn.ReLU(),
                    torch.nn.Linear(
                        in_features=hidden_features, out_features=self.out_features
                    ),
                    nn.Softmax(),
                ]
            )

        ## Add Dropout before the network
        if self.dropout_first is not None:
            self.layers.insert(0, nn.Dropout(self.dropout_first))

        ## Add Dropout before the network
        if self.dropout_last is not None:
            self.layers.insert(len(self.layers) - 1, nn.Dropout(self.dropout_last))

        self.task_network = nn.Sequential(*self.layers)

    def create(self):
        return self.task_network

    def forward(self, X):
        return self.task_network(X)


class Encoderv3(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1
            ),  # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1),  # Output: #bs, 8, 98
            nn.Conv1d(
                in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1
            ),  # Output: b, 16, 50
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(
                in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1
            ),  # Output: b, 32, 10
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),  # Output: b, 32, 3
            Flatten(),
            # nn.Linear(in_features = 3*32, out_features = self.hparams.hidden_layer_1_dim),
            # nn.BatchNorm1d(self.hparams.hidden_layer_1_dim),
            # nn.LeakyReLU(),
            # nn.Linear(in_features = self.hparams.hidden_layer_1_dim, out_features = hparams.latent_size)
        )

    def forward(self, X):
        return self._encoder(X)


class Decoderv3(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._decoder = nn.Sequential(
            nn.Linear(
                in_features=hparams.latent_size, out_features=3 * 32
            ),  # , self.hparams.hidden_layer_1_dim),
            # nn.ReLU(),
            # nn.Linear(in_features = self.hparams.hidden_layer_1_dim, out_features = 3*32),
            # nn.ReLU(),
            UnFlatten(),
            nn.ConvTranspose1d(32, 16, 6, stride=2),  # b, 16, 10
            nn.ReLU(),
            nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
        )

    def forward(self, X):
        return self._decoder(X)


class MTLModelv15(MTLModelv10):
    """
    Changes from v5:
    BatchNorm from encoder and decoder layers ARE BACK.
    SOFTMAX IN CRYSTAL
    LOSS_V3: ALL IN AUTO ..LEADS TO NEGATIVE VALS.
    """

    def __init__(self, hparams):
        super().__init__(hparams)
        # self.save_parameters()
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv3(self.hparams)
            self.decoder = Decoderv3(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()

        self.fc_mu = nn.Linear(96, self.hparams.latent_size)
        self.fc_var = nn.Linear(96, self.hparams.latent_size)

        self.task_CN = create_mlp_v5(
            self.hparams.latent_size, "CN", hidden_layers=2, hidden_features=512
        ).create()
        self.task_Dist = create_mlp_v5(
            self.hparams.latent_size, "Distances", hidden_layers=2, hidden_features=512
        ).create()
        self.task_H2 = create_mlp_v5(
            self.hparams.latent_size, "H_fraction", hidden_layers=2, hidden_features=512
        ).create()
        self.task_Crystal = create_classifier_v5(
            self.hparams.latent_size,
            "Crystal",
            hidden_layers=2,
            hidden_features=512,
            out_features=26,
        ).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v10(4, self.hparams)

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)

        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

        latent = z
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent)
        H2_hat = self.task_H2(latent)
        Crystal_hat = self.task_Crystal(latent)

        # import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def setup(self, stage):
        self.train_ds = XasMultiTaskDatasetv2(
            json_file=self.hparams.train_json_file,
            root_dir="../data/",
            descriptor="Distances",
            MTL=True,
            scalers=None,
            scale_min_max=True,
            cat_list=None,
            transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
        )
        self.train_size = len(self.train_ds)

        if self.hparams.val_json_file is not None:
            self.val_ds = XasMultiTaskDatasetv2(
                json_file=self.hparams.val_json_file,
                root_dir="../data/",
                descriptor="Distances",
                MTL=True,
                scalers=self.train_ds.scalers,
                scale_min_max=True,
                cat_list=self.train_ds.categories,
                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
            )
        else:
            self.train_size = int(0.98 * len(self.train_ds))
            self.val_size = len(self.train_ds) - self.train_size
            self.train_ds, self.val_ds = random_split(
                self.train_ds, [self.train_size, self.val_size]
            )

        if self.hparams.test_json_file:
            self.test_ds = XasMultiTaskDatasetv2(
                json_file=self.hparams.test_json_file,
                root_dir="../data/",
                descriptor="Distances",
                MTL=True,
                scalers=self.train_ds.scalers,
                scale_min_max=True,
                cat_list=self.train_ds.categories,
                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
            )

        exp_dataset = XasExpDataset(
            json_file=self.hparams.exp_json_file,
            root_dir="../data/",
            descriptor="Distances",
            transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]),
        )

        self.exp_ds = exp_dataset

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 10, 1
        )

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )

    @pl.data_loader
    def val_dataloader(self):
        theo_DL = DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )
        exp_DL = DataLoader(
            self.exp_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )
        return [theo_DL, exp_DL]

    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.cpus,
            drop_last=False,
            pin_memory=True,
        )

    def _calc_loss(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction="mean")

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

        loss = self.kl_coeff * kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    def training_step(self, batch, batch_idx):

        x, y = batch
        # flatten any input
        # x = x.view(x.size(0), -1)
        ##
        y_hat = self.forward(x)  ##

        ## MTL Loss
        loss, loss_log = self.mtl_loss_func(y_hat, y, x)

        tensorboard_logs = {"train_loss": loss}  # ,
        progress_bar_metrics = tensorboard_logs

        return {
            "loss": loss,
            "log": {**tensorboard_logs, "train_mtl_loss": loss_log},
            "progress_bar": progress_bar_metrics,
        }

    def training_epoch_end(self, outputs):

        train_losses = torch.stack([x["log"]["train_loss"] for x in outputs])
        train_avg_loss = train_losses.detach().mean()
        train_std_loss = train_losses.detach().std()
        train_recon_loss_epoch = (
            torch.stack(
                [x["log"]["train_mtl_loss"]["train_recon_loss"] for x in outputs]
            )
            .detach()
            .mean()
        )
        train_mmd_loss_epoch = (
            torch.stack([x["log"]["train_mtl_loss"]["train_mmd_loss"] for x in outputs])
            .detach()
            .mean()
        )

        tensorboard_logs = {
            "train_epoch_loss": train_avg_loss,
            "training_loss_std": train_std_loss,
            "normalized_train_loss_std": train_std_loss / train_avg_loss,
            "train_recon_loss_per_epoch": train_recon_loss_epoch,
            "train_mmd_loss_per_epoch": train_mmd_loss_epoch,
        }
        return {"log": tensorboard_logs}

    def validation_step(self, batch, batch_idx, dataloader_idx):

        x, y = batch
        # Flatten input for linear layers
        # x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        ## Caluculate only reconstruction and MMD loss
        recon_loss, mmd_loss = self.mtl_loss_func.loss_function(
            pred=y_hat[0], latent=y_hat[1], true=x
        )

        ## Combine based on reg_para with MMD Loss
        val_loss = recon_loss  # + self.hparams.reg_para*mmd_loss

        tensorboard_logs = {"val_recon_loss": recon_loss, "val_mmd_loss": mmd_loss}
        return {"val_loss": val_loss, "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        # import pdb; pdb.set_trace()
        val_losses = torch.stack([x["val_loss"] for dls in outputs for x in dls])
        val_avg_loss = val_losses.detach().mean()
        val_std_loss = val_losses.detach().std()

        tensorboard_logs = {
            "val_loss": val_avg_loss,
            "val_loss_std": val_std_loss,
            "normalized_val_loss_std": val_std_loss / val_avg_loss,
        }
        return {
            "avg_val_loss": val_avg_loss,
            "log": tensorboard_logs,
            "progress_bar": {"val_loss": val_avg_loss},
        }

    def test_step(self, batch, batch_idx, dataloader_idx):
        output = self.validation_step(batch, batch_idx, dataloader_idx)
        # Rename output keys
        # output['test_loss'] = output.pop('val_loss')

        return output

        # return {'test_loss': loss}

    def test_epoch_end(self, outputs):

        test_loss = (
            torch.stack([x["val_loss"] for dls in outputs for x in dls]).detach().mean()
        )
        tensorboard_logs = {"test_epoch_loss": test_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            "test_loss": test_loss,
            "log": tensorboard_logs,
            "progress_bar": progress_bar_metrics,
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--input_dim", type=int, default=100)
        # parser.add_argument('--bias', default='store_true')
        parser.add_argument("--hidden_layer_1_dim", type=int, default=500)

        return parser
