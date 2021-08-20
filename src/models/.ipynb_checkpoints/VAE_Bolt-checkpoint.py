from argparse import ArgumentParser


from pl_bolts.models.autoencoders.components import resnet18_encoder, resnet18_decoder
from pl_bolts.models.autoencoders.components import resnet50_encoder, resnet50_decoder


from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
#from .lightningVAE import *
#from .lightningAE import *
#from .lightningXAS import *
#from .MTL import *

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

class XasExpDataset(Dataset):
    """ Custom XAS Dataset"""
    def __init__(self,json_file, root_dir, descriptor = 'CN', transform=None):
        self.XAS_Frame = pd.read_json(json_file, orient='records')
        #self.XAS_Frame.Abs = self.XAS_Frame.Abs.apply(lambda x: x/np.max(x)) ## Questionable Normalization
        self.root_dir = root_dir
        self.transform = transform
        #if type(descriptor)==type(''):
        self.descriptor = descriptor
        
    def __len__(self):
        return len(self.XAS_Frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==int:
            idx=[idx]
        data_Energy = self.XAS_Frame['Energy'].iloc[idx]
        data_Abs    = self.XAS_Frame['Abs'].iloc[idx]
        data_Y      = self.XAS_Frame[self.descriptor].iloc[idx].round(3)#.to_numpy()
        data_Y      = np.vstack(data_Y)#.reshape(-1,2) #Not sure why do I need to reshape
        sample      = pd.Series({'Energy': np.vstack(data_Energy), 'Abs': np.vstack(data_Abs), 'Descriptor': data_Y})
        #print(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def show(self, idx):
        x, y = self.__getitem__(idx)
        data = pd.Series({'x_axis':np.array(energymesh).reshape(1,-1).squeeze(), 'y_axis':x.numpy().squeeze()})
        print(data)
        fig, ax = plt.subplots()
        _ = sns.lineplot(x='x_axis', y='y_axis', data=data)
        _ = plt.title(f"Sample_#{idx}_{self.descriptor}{y}")
        return fig, ax


class Encoderv2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._encoder = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1), # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.BatchNorm1d(8),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1), # Output: #bs, 8, 98
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1),# Output: b, 16, 50
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1),  # Output: b, 32, 10
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),# Output: b, 32, 3
            Flatten(),
            nn.Linear(in_features = 3*32, out_features = self.hparams.hidden_layer_1_dim),
            nn.BatchNorm1d(self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.hparams.hidden_layer_1_dim, out_features = hparams.latent_size)
        )
    
    def forward(self, X):
        return self._encoder(X)

class Decoderv2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._decoder = nn.Sequential(
                nn.Linear(in_features = hparams.latent_size, out_features = self.hparams.hidden_layer_1_dim),
                nn.ReLU(),
                nn.Linear(in_features = self.hparams.hidden_layer_1_dim, out_features = 3*32),
                nn.ReLU(),
                UnFlatten(),
                nn.ConvTranspose1d(32, 16, 6, stride=2), # b, 16, 10
                nn.ReLU(),
                nn.ConvTranspose1d(16, 8, 3, stride=5, padding=1),  # b, 8, 46
                nn.ReLU(),
                nn.ConvTranspose1d(8, 1, 10, stride=2, padding=0),  # b, 1, 100
                )
    
    def forward(self, X):
        return self._decoder(X)

class Encoderv3(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(Encoderv3, self).__init__()
        self.model = torch.nn.ModuleList([
            Reshape((1,10,10)),
            torch.nn.Conv2d(1, 8, 3, 1, padding=1),
            torch.nn.LeakyReLU(),
            #torch.nn.AvgPool1d(3,1),
            torch.nn.Conv2d(8, 16, 2, 2, padding=1),
            torch.nn.LeakyReLU(),
            #torch.nn.AvgPool1d(3,2),
            torch.nn.Conv2d(16, 32, 2, 2, padding=1),
            torch.nn.LeakyReLU(),
            #torch.nn.AvgPool1d(3,3),
            Flatten(),
            torch.nn.Linear(512, 1024),
            #torch.nn.LeakyReLU(),
            #torch.nn.Linear(1024, self.hparams.latent_size)
        ])
        
    def forward(self, x):
        #print("Encoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
    
class Decoderv3(torch.nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(Decoderv3, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(self.hparams.latent_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            Reshape((32,4,4)),
            torch.nn.ConvTranspose2d(32, 16, 2, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 2, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3, 1, padding=1),
            Reshape((1,100)),
            #torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        #print("Decoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x

class Encoderv4(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(Encoderv4, self).__init__()
        self.model = torch.nn.ModuleList([
            Reshape((1,10,10)),
            torch.nn.Conv2d(1, 8, 3, 1, padding=1),
            torch.nn.LeakyReLU(),
            #torch.nn.AvgPool1d(3,1),
            torch.nn.Conv2d(8, 16, 2, 2, padding=1),
            torch.nn.LeakyReLU(),
            #torch.nn.AvgPool1d(3,2),
            torch.nn.Conv2d(16, 32, 2, 2, padding=1),
            torch.nn.LeakyReLU(),
            #torch.nn.AvgPool1d(3,3),
            Flatten(),
            torch.nn.Linear(512, 1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, self.hparams.latent_size)
        ])
        
    def forward(self, x):
        #print("Encoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x
    
class Decoderv4(torch.nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(Decoderv4, self).__init__()
        self.model = torch.nn.ModuleList([
            torch.nn.Linear(self.hparams.latent_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            Reshape((32,4,4)),
            torch.nn.ConvTranspose2d(32, 16, 2, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 2, 2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 3, 1, padding=1),
            Reshape((1,100)),
            #torch.nn.Sigmoid()
        ])
        
    def forward(self, x):
        #print("Decoder")
        #print(x.size())
        for layer in self.model:
            x = layer(x)
            #print(x.size())
        return x

class Reshape(torch.nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

class Flatten(nn.Module):
    def __init__(self, full=False):
        super(Flatten, self).__init__()
        self.full = full
    def forward(self,x):
        return x.view(-1) if self.full else x.view(x.size(0),-1)

class UnFlatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),32,-1)

class VAE(pl.LightningModule):
    def __init__(
        self,
        hparams,
        enc_out_dim=96,
        kl_coeff=0.1,
        latent_dim=256,
        lr=1e-4,
        **kwargs
    ):
        """
        Standard VAE with Gaussian Prior and approx posterior.

        Model is available pretrained on different datasets:

        Example::

            # not pretrained
            vae = VAE()

            # pretrained on cifar10
            vae = VAE.from_pretrained('cifar10-resnet18')

            # pretrained on stl10
            vae = VAE.from_pretrained('stl10-resnet18')

        Args:

            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()
        self.hparams = hparams
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        #self.input_height = input_height

        
        self.encoder = Encoderv3(self.hparams)
        self.decoder = Decoderv3(self.hparams)

        self.fc_mu = nn.Linear(self.enc_out_dim, self.hparams.latent_size)
        self.fc_var = nn.Linear(self.enc_out_dim, self.hparams.latent_size)

    

  
    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return self.decoder(z)

    
    
    def _run_step(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat, p, q = self._run_step(x)

        recon_loss = F.mse_loss(x_hat, x, reduction='mean')

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

        loss = self.kl_coeff*kl + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "kl": kl,
            "loss": loss,
        }
        return loss, logs

    # def setup(self, stage):
    #     train_dataset = XasMultiTaskDataset(json_file = self.hparams.train_json_file,
    #                 root_dir='../data/',
    #                 descriptor=self.hparams.descriptor,
    #                 MTL=self.hparams.MTL,
    #                 transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
    #     self.train_size = len(train_dataset)
       
    #     if self.hparams.test_json_file:
    #         self.test_ds = XasMultiTaskDataset(json_file = self.hparams.test_json_file,
    #                                             root_dir='../data/',
    #                                             descriptor=self.hparams.descriptor,
    #                                             MTL=self.hparams.MTL,
    #                                             transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]))
    
    #     exp_dataset = XasExpDataset(json_file=self.hparams.exp_json_file,
    #                              root_dir = '../data/',
    #                              descriptor = 'Distances',
    #                              transform = Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        
    #     self.train_ds, self.val_ds = train_dataset, exp_dataset


    # def val_dataloader(self):
    #     return DataLoader(self.val_ds, batch_size=2, shuffle=True,
    #                       num_workers=self.hparams.cpus, drop_last=False)


    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return result

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({f"val_{k}": v for k, v in logs.items()})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='resnet18', help="resnet18/resnet50")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim", type=int, default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=0.1)
        parser.add_argument("--latent_dim", type=int, default=256)

        #parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        parser.add_argument('--input_dim', type=int, default=100)
        #parser.add_argument('--bias', default='store_true')
        parser.add_argument('--hidden_layer_1_dim', type=int, default=500)
        
        return parser


class VAE_MMD(pl.LightningModule):
    def __init__(
        self,
        hparams,
        enc_out_dim=96,
        mmd_coeff=0.1,
        **kwargs
    ):
        """
        

        Example::

    

        Args:

    
        """

        super(VAE_MMD, self).__init__()

        self.save_hyperparameters()
        self.hparams = hparams
        self.mmd_coeff = mmd_coeff
        self.enc_out_dim = enc_out_dim
        # self.latent_dim = latent_dim
        #self.input_height = input_height

        
        self.encoder = Encoderv4(self.hparams)
        self.decoder = Decoderv4(self.hparams)

        # self.fc_mu = nn.Linear(self.enc_out_dim, self.hparams.latent_size)
        # self.fc_var = nn.Linear(self.enc_out_dim, self.hparams.latent_size)

    

  
    def forward(self, x):
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return z, x_reconstructed

    
    
    def _run_step(self, x):
        z, x_reconstructed = self(x)
        return z, x_reconstructed

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1) # (x_size, 1, dim)
        y = y.unsqueeze(0) # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
        return torch.exp(-kernel_input) # (x_size, y_size)

    def compute_mmd(self, x, y):
        x_kernel = self.compute_kernel(x, x)
        y_kernel = self.compute_kernel(y, y)
        xy_kernel = self.compute_kernel(x, y)
        mmd = x_kernel.mean() + y_kernel.mean() - 2*xy_kernel.mean()
        return mmd

    def step(self, batch, batch_idx):
        x, y = batch
        z, x_hat = self._run_step(x)
        samples = Variable(torch.randn(200, self.hparams.latent_size, requires_grad=False))


        recon_loss = F.mse_loss(x_hat, x, reduction='mean')
        mmd_loss = self.compute_mmd(samples, z)

        
        mmd_loss =  mmd_loss.mean()

        loss = self.mmd_coeff*mmd_loss + recon_loss

        logs = {
            "recon_loss": recon_loss,
            "mmd": mmd_loss,
            "loss": loss,
        }
        return loss, logs

    # def setup(self, stage):
    #     train_dataset = XasMultiTaskDataset(json_file = self.hparams.train_json_file,
    #                 root_dir='../data/',
    #                 descriptor=self.hparams.descriptor,
    #                 MTL=self.hparams.MTL,
    #                 transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
    #     self.train_size = len(train_dataset)
       
    #     if self.hparams.test_json_file:
    #         self.test_ds = XasMultiTaskDataset(json_file = self.hparams.test_json_file,
    #                                             root_dir='../data/',
    #                                             descriptor=self.hparams.descriptor,
    #                                             MTL=self.hparams.MTL,
    #                                             transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]))
    
    #     exp_dataset = XasExpDataset(json_file=self.hparams.exp_json_file,
    #                              root_dir = '../data/',
    #                              descriptor = 'Distances',
    #                              transform = Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        
    #     self.train_ds, self.val_ds = train_dataset, exp_dataset


    # def val_dataloader(self):
    #     return DataLoader(self.val_ds, batch_size=2, shuffle=True,
    #                       num_workers=self.hparams.cpus, drop_last=False)


    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return result

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({f"val_{k}": v for k, v in logs.items()})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--enc_out_dim", type=int, default=512,
            help="512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--mmd_coeff", type=float, default=0.1)

        #parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")

        parser.add_argument('--input_dim', type=int, default=100)


        
        return parser

# def cli_main(args=None):
#     from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule, STL10DataModule

#     pl.seed_everything()

#     parser = ArgumentParser()
#     parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10", "stl10", "imagenet"])
#     script_args, _ = parser.parse_known_args(args)

#     if script_args.dataset == "cifar10":
#         dm_cls = CIFAR10DataModule
#     elif script_args.dataset == "stl10":
#         dm_cls = STL10DataModule
#     elif script_args.dataset == "imagenet":
#         dm_cls = ImagenetDataModule
#     else:
#         raise ValueError(f"undefined dataset {script_args.dataset}")

#     parser = VAE.add_model_specific_args(parser)
#     parser = pl.Trainer.add_argparse_args(parser)
#     args = parser.parse_args(args)

#     dm = dm_cls.from_argparse_args(args)
#     args.input_height = dm.size()[-1]

#     if args.max_steps == -1:
#         args.max_steps = None

#     model = VAE(**vars(args))

#     trainer = pl.Trainer.from_argparse_args(args)
#     trainer.fit(model, dm)
#     return dm, model, trainer


# if __name__ == "__main__":
#     dm, model, trainer = cli_main()