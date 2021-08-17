from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
from .lightningVAE import *
from .lightningAE import *
from .lightningXAS import *
from .MTL import *

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

class MTLModelv3(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_classifier(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v3(6, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
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
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]
    
    def _get_model(self):
        model_name = self.hparams.best_model_type
        filepath = self.hparams.best_model_path
        MODEL_CHECKPOINT_PATH = glob.glob(filepath)[0]
        print("...Loading Model Checkpoint from: ", MODEL_CHECKPOINT_PATH, '\n')
        model = model_name.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
        return model

    def setup(self, stage):
        train_dataset = XasMultiTaskDataset(json_file = self.hparams.train_json_file,
                    root_dir='../data/',
                    descriptor=self.hparams.descriptor,
                    MTL=self.hparams.MTL,
                    transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        self.train_size = int(0.98*len(train_dataset))
        self.val_size = len(train_dataset)-self.train_size
        self.train_ds, self.val_ds = random_split(train_dataset,[self.train_size,self.val_size])
        if self.hparams.test_json_file:
            self.test_ds = XasMultiTaskDataset(json_file = self.hparams.test_json_file,
                                                root_dir='../data/',
                                                descriptor=self.hparams.descriptor,
                                                MTL=self.hparams.MTL,
                                                transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, 
                    weight_decay=self.hparams.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 1)

        return [optimizer], [scheduler]


    @pl.data_loader  
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.cpus, drop_last=False)
    
    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.cpus, drop_last=False)
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.cpus, drop_last=False)
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        # flatten any input
        #x = x.view(x.size(0), -1)
        ## 
        y_hat = self.forward(x) ## 
        

        ## MTL Loss
        loss, loss_log = self.mtl_loss_func(y_hat, y, x)
        
        tensorboard_logs = {'train_loss': loss}#, 
        progress_bar_metrics = tensorboard_logs
        
        return {
            'loss': loss,
            'log': {**tensorboard_logs, 'train_mtl_loss':loss_log},
            'progress_bar': progress_bar_metrics
        }
    
    def training_epoch_end( self, outputs):
        
        train_losses = torch.stack([x['log']['train_loss'] for x in outputs])
        train_avg_loss = train_losses.detach().mean()
        train_std_loss = train_losses.detach().std()
        train_recon_loss_epoch = torch.stack([x['log']['train_mtl_loss']['train_recon_loss'] for x in outputs]).detach().mean()
        train_mmd_loss_epoch = torch.stack([x['log']['train_mtl_loss']['train_mmd_loss'] for x in outputs]).detach().mean()
        
        tensorboard_logs = {'train_epoch_loss':train_avg_loss,
                            'training_loss_std':train_std_loss,
                            'normalized_train_loss_std':train_std_loss/train_avg_loss,
                            'train_recon_loss_per_epoch':train_recon_loss_epoch,
                            'train_mmd_loss_per_epoch':train_mmd_loss_epoch}
        return {'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Flatten input for linear layers
        #x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        loss, loss_log = self.mtl_loss_func(y_hat, y, x)


        tensorboard_logs = {'val_mtl_loss':loss_log}
        return {'val_loss': loss, 
                'log': tensorboard_logs,
                }

    def validation_epoch_end(self, outputs):
        val_losses = torch.stack([x['val_loss'] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        val_recon_loss_epoch = torch.stack([x['log']['val_mtl_loss']['train_recon_loss'] for x in outputs]).mean()
        val_mmd_loss_epoch = torch.stack([x['log']['val_mtl_loss']['train_mmd_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': val_avg_loss,
                            'val_loss_std':val_std_loss, 
                            'normalized_val_loss_std':val_std_loss/val_avg_loss,
                            'val_recon_loss_per_epoch':val_recon_loss_epoch,
                            'val_mmd_loss_per_epoch':val_mmd_loss_epoch,
                            }
        return {'avg_val_loss': val_avg_loss,
                 'log': tensorboard_logs,
                 'progress_bar': {'val_loss': val_avg_loss}}


        #val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        #tensorboard_logs = {'val_epoch_loss': val_loss}
        #progress_bar_metrics = tensorboard_logs
        #return {
        #    'val_loss': val_loss,
        #    'log': tensorboard_logs,
        #    'progress_bar': progress_bar_metrics
        #}

    def test_step(self, batch, batch_idx):
        output = self.validation_step(batch, batch_idx)
        # Rename output keys
        #output['test_loss'] = output.pop('val_loss')


        return output

        #return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        
        test_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_epoch_loss':test_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'test_loss': test_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }



    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_dim', type=int, default=100)
        #parser.add_argument('--bias', default='store_true')
        parser.add_argument('--hidden_layer_1_dim', type=int, default=500)
        
        return parser

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

class create_mlp(nn.Module):
    def __init__(self, input_size, descriptor, hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.descriptor = descriptor
        self.hidden_layers = hidden_layers
        
        if self.descriptor in ['CN','Distances']:
            self.out_features = 6
            
        elif self.descriptor == 'H_fraction':
            self.out_features = 1
        elif self.descriptor == 'Crystal':
            self.out_features = 25
        
        if self.hidden_layers==1:
            self.task_network = nn.Sequential(
                    torch.nn.Linear(in_features=self.input_size, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=self.out_features),
                    nn.ReLU())
        elif self.hidden_layers==2:
            self.task_network = nn.Sequential(
                    torch.nn.Linear(in_features=self.input_size, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=self.out_features),
                    nn.ReLU())
        
    def create(self):
        return self.task_network
    
    def forward(self, X):
        return self.task_network(X)

class create_classifier(nn.Module):
    def __init__(self, input_size, descriptor, hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.descriptor = descriptor
        self.hidden_layers = hidden_layers
        
        if self.descriptor == 'Crystal':
            self.out_features = 25
        
        if self.hidden_layers==1:
            self.task_network = nn.Sequential(
                    torch.nn.Linear(in_features=self.input_size, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=self.out_features),
                    nn.ReLU(),
                    nn.Softmax())
        elif self.hidden_layers==2:
            self.task_network = nn.Sequential(
                    torch.nn.Linear(in_features=self.input_size, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=self.out_features),
                    nn.ReLU(),
                    nn.Softmax())
        
                
        
    def create(self):
        return self.task_network
    
    def forward(self, X):
        return self.task_network(X)
    

class MultiTaskLossWrapper_v3(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v3, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams
    def forward(self, preds, y, x):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce='sum'), nn.CrossEntropyLoss()
        
        #sages = (age*4.75).exp_()
        
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(100*preds[4],100*y[:,15]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:,14].long()).mean()
        #loss_Crystal = 0.
        # Need to put a Softmax layer..

        precision_CN = torch.exp(-self.log_vars[0])
        loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)

        precision_Dist = torch.exp(-self.log_vars[1])
        loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)

        precision_H_fraction = torch.exp(-self.log_vars[2])
        loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        
        precision_Crystal = torch.exp(-self.log_vars[3])
        loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        
        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_MMD = self.loss_function(pred=reconstruction, true=x, latent=mu)
        
        precision_recon = torch.exp(-self.log_vars[4])
        loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        
        ## MMD Loss ##
        precision_mmd = torch.exp(-self.log_vars[5])
        loss_mmd = torch.sum(precision_mmd*loss_MMD+self.log_vars[5],-1)


           

        return loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 
                                                                'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 
                                                                'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),
                                                                'train_recon_loss':loss_reconstruction.detach(), 
                                                                'loss_weight_recon':self.log_vars[4].detach(),
                                                                'train_mmd_loss':loss_mmd.detach(),
                                                                'loss_weight_MMD':self.log_vars[5].detach(),}

    def gaussian_kernel(self, a, b):
            
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)#.cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)#.cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        #a_core = a_core.to(self.device)
        #b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        
        return torch.exp(-numerator)

    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()
    

    def loss_function(self, pred, true, latent):
            return (pred-true).pow(2).sum(), self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)

class create_mlp_v4(nn.Module):
    '''
    Removed Activation function from Output Layer
    '''
    def __init__(self, input_size, descriptor, hidden_layers=1):
        super().__init__()
        self.input_size = input_size
        self.descriptor = descriptor
        self.hidden_layers = hidden_layers
        
        if self.descriptor in ['CN','Distances']:
            self.out_features = 6
            
        elif self.descriptor == 'H_fraction':
            self.out_features = 1
        elif self.descriptor == 'Crystal':
            self.out_features = 25
        
        if self.hidden_layers==1:
            self.task_network = nn.Sequential(
                    torch.nn.Linear(in_features=self.input_size, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=self.out_features))
        elif self.hidden_layers==2:
            self.task_network = nn.Sequential(
                    torch.nn.Linear(in_features=self.input_size, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=512),
                    nn.ReLU(),
                    torch.nn.Linear(in_features=512, out_features=self.out_features))
        
    def create(self):
        return self.task_network
    
    def forward(self, X):
        return self.task_network(X)

class MultiTaskLossWrapper_v4(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v4, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams
    def forward(self, preds, y, x):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce='sum'), nn.CrossEntropyLoss()
        
        #sages = (age*4.75).exp_()
        
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(preds[4],y[:,15]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:,14].long()).mean()
        #loss_Crystal = 0.
        # Need to put a Softmax layer..

        precision_CN = torch.exp(-self.log_vars[0])
        #loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)
        loss_CN = torch.sum(precision_CN*loss_CN + 0.*self.log_vars[0], -1)

        precision_Dist = torch.exp(-self.log_vars[1])
        #loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)
        loss_Dist = torch.sum(precision_Dist*loss_Dist + 0.*self.log_vars[1],-1)

        precision_H_fraction = torch.exp(-self.log_vars[2])
        #loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + 0.*self.log_vars[2], -1)
        
        precision_Crystal = torch.exp(-self.log_vars[3])
        #loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + 0.*self.log_vars[3],-1)
        
        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_MMD = self.loss_function(pred=reconstruction, true=x, latent=mu)
        
        precision_recon = torch.exp(-self.log_vars[4])
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        loss_reconstruction = torch.sum(precision_recon*loss_reconstruction + 0.*self.log_vars[4],-1)
        
        ## MMD Loss ##
        precision_mmd = torch.exp(-self.log_vars[5])
        #loss_mmd = torch.sum(precision_mmd*loss_MMD+self.log_vars[5],-1)
        loss_mmd = torch.sum(precision_mmd*loss_MMD + 0.*self.log_vars[5],-1)


           

        return loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 
                                                                'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 
                                                                'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),
                                                                'train_recon_loss':loss_reconstruction.detach(), 
                                                                'loss_weight_recon':self.log_vars[4].detach(),
                                                                'train_mmd_loss':loss_mmd.detach(),
                                                                'loss_weight_MMD':self.log_vars[5].detach(),}

    def gaussian_kernel(self, a, b):
            
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)#.cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)#.cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        #a_core = a_core.to(self.device)
        #b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        
        return torch.exp(-numerator)

    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()
    

    def loss_function(self, pred, true, latent):
            return (pred-true).pow(2).sum(), self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)

class MultiTaskLossWrapper_v5(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v5, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams
    def forward(self, preds, y, x):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce='sum'), nn.CrossEntropyLoss()
        
        #sages = (age*4.75).exp_()
        
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(preds[4],y[:,15]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:,14].long()).mean()
        #loss_Crystal = 0.
        # Need to put a Softmax layer..

        precision_CN = torch.exp(-self.log_vars[0])
        #loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)
        loss_CN = torch.sum(precision_CN*loss_CN + torch.log(1+self.log_vars[0].exp()), -1)

        precision_Dist = torch.exp(-self.log_vars[1])
        #loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)
        loss_Dist = torch.sum(precision_Dist*loss_Dist + torch.log(1+self.log_vars[1].exp()),-1)

        precision_H_fraction = torch.exp(-self.log_vars[2])
        #loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + torch.log(1+self.log_vars[2].exp()), -1)
        
        precision_Crystal = torch.exp(-self.log_vars[3])
        #loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + torch.log(1+self.log_vars[3].exp()),-1)
        
        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_mmd = self.loss_function(pred=reconstruction, true=x, latent=mu)
        
        precision_recon = torch.exp(-self.log_vars[4])
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction + torch.log(1+self.log_vars[4].exp()),-1)
        
        ## MMD Loss ##
        #precision_mmd = torch.exp(-self.log_vars[5])
        #loss_mmd = torch.sum(precision_mmd*loss_mmd+self.log_vars[5],-1)
        #loss_mmd = torch.sum(precision_mmd*loss_mmd + torch.log(1+self.log_vars[5].exp()),-1)

        ## Reconstruction + 0.01 MMD Loss ##
        wtloss_reconstruction_mmd = torch.sum(precision_recon*(loss_reconstruction+self.hparams.reg_para*loss_mmd) + torch.log(1+self.log_vars[4].exp()),-1)

        ## loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal   

        return wtloss_reconstruction_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 
                                                                'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 
                                                                'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),
                                                                'train_recon_loss':loss_reconstruction.detach(), 
                                                                'loss_weight_recon':self.log_vars[4].detach(),
                                                                'train_mmd_loss':loss_mmd.detach(),
                                                                'loss_weight_MMD':0.01*self.log_vars[4].detach(), ### Modified
                                                                'train_loss_recon_mmd_wt':wtloss_reconstruction_mmd.detach() ## Combined weighted Recon+MMD Loss
                                                                }

    def gaussian_kernel(self, a, b):
            
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)#.cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)#.cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        #a_core = a_core.to(self.device)
        #b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        
        return torch.exp(-numerator)

    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()
    

    def loss_function(self, pred, true, latent):
            return (pred-true).pow(2).sum(), self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)

class MultiTaskLossWrapper_v6(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v6, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams
    def forward(self, preds, y, x):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce='none'), nn.CrossEntropyLoss()

        #sages = (age*4.75).exp_()
        
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(preds[4],y[:,15]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:,14].long()).mean()

        precision_CN = torch.exp(-self.log_vars[0])
        #loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)
        loss_CN = torch.sum(precision_CN*loss_CN + torch.log(1+self.log_vars[0].exp()), -1)

        precision_Dist = torch.exp(-self.log_vars[1])
        #loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)
        loss_Dist = torch.sum(precision_Dist*loss_Dist + torch.log(1+self.log_vars[1].exp()),-1)

        precision_H_fraction = torch.exp(-self.log_vars[2])
        #loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + torch.log(1+self.log_vars[2].exp()), -1)
        
        precision_Crystal = torch.exp(-self.log_vars[3])
        #loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + torch.log(1+self.log_vars[3].exp()),-1)
        
        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_mmd = self.loss_function(pred=reconstruction, true=x, latent=mu)
        
        precision_recon = torch.exp(-self.log_vars[4])
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction + torch.log(1+self.log_vars[4].exp()),-1)
        
        ## MMD Loss ##
        #precision_mmd = torch.exp(-self.log_vars[5])
        #loss_mmd = torch.sum(precision_mmd*loss_mmd+self.log_vars[5],-1)
        #loss_mmd = torch.sum(precision_mmd*loss_mmd + torch.log(1+self.log_vars[5].exp()),-1)

        ## Reconstruction + 0.01 MMD Loss ##
        wtloss_reconstruction_mmd = torch.sum(precision_recon*(loss_reconstruction+self.hparams.reg_para*loss_mmd) + torch.log(1+self.log_vars[4].exp()),-1)

        ## loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal   

        return wtloss_reconstruction_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 
                                                                'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 
                                                                'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),
                                                                'train_recon_loss':loss_reconstruction.detach(), 
                                                                'loss_weight_recon':self.log_vars[4].detach(),
                                                                'train_mmd_loss':loss_mmd.detach(),
                                                                'loss_weight_MMD':0.01*self.log_vars[4].detach(), ### Modified
                                                                'train_loss_recon_mmd_wt':wtloss_reconstruction_mmd.detach() ## Combined weighted Recon+MMD Loss
                                                                }

    def gaussian_kernel(self, a, b):
            
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)#.cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)#.cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        #a_core = a_core.to(self.device)
        #b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        
        return torch.exp(-numerator)

    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()
    

    def loss_function(self, pred, true, latent):
            return (pred-true).pow(2).mean(), self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)


class MTLModelv4(MTLModelv3):
    '''
    Changes/Assumption:
    H-Fraction: nn.Sigmoid in the end. To keep it between 0 to 1.
    Crystal: nn.Softmax in the end. 
    '''
    def __init__(self, hparams):
        super().__init__(hparams)
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp_v4(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp_v4(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp_v4(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_mlp_v4(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v5(6, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_)
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent) 
        H2_hat = nn.Sigmoid()(self.task_H2(latent))
        Crystal_hat = (self.task_Crystal(latent))
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

class MTLModelv5(MTLModelv3):
    '''
    Changes/Assumption:
    H-Fraction: nn.Sigmoid in the end. To keep it between 0 to 1.
    ---Crystal: nn.Softmax in the end---
    '''
    def __init__(self, hparams):
        super().__init__(hparams)
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp_v4(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp_v4(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp_v4(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_mlp_v4(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v5(5, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_) ### Buggs me ...
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent) 
        H2_hat = nn.Sigmoid()(self.task_H2(latent))
        Crystal_hat = (self.task_Crystal(latent))
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

    def setup(self, stage):
        train_dataset = XasMultiTaskDataset(json_file = self.hparams.train_json_file,
                    root_dir='../data/',
                    descriptor=self.hparams.descriptor,
                    MTL=self.hparams.MTL,
                    transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        self.train_size = len(train_dataset)
       
        if self.hparams.test_json_file:
            self.test_ds = XasMultiTaskDataset(json_file = self.hparams.test_json_file,
                                                root_dir='../data/',
                                                descriptor=self.hparams.descriptor,
                                                MTL=self.hparams.MTL,
                                                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]))
    
        exp_dataset = XasExpDataset(json_file=self.hparams.exp_json_file,
                                 root_dir = '../data/',
                                 descriptor = 'Distances',
                                 transform = Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        
        self.train_ds, self.val_ds = train_dataset, exp_dataset

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=2, shuffle=True,
                          num_workers=self.hparams.cpus, drop_last=False)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Flatten input for linear layers
        #x = x.view(x.size(0), -1)
        y_hat = self.forward(x)

        ## Caluculate only reconstruction and MMD loss
        recon_loss, mmd_loss = self.mtl_loss_func.loss_function(pred=y_hat[0], latent=y_hat[1], true=x)

        ## Combine based on reg_para with MMD Loss
        val_loss = recon_loss #+ self.hparams.reg_para*mmd_loss

        tensorboard_logs = {'val_recon_loss': recon_loss, 'val_mmd_loss': mmd_loss}
        return {'val_loss': val_loss, 
                'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_losses = torch.stack([x['val_loss'] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        
        tensorboard_logs = {'val_loss': val_avg_loss,
                            'val_loss_std':val_std_loss, 
                            'normalized_val_loss_std':val_std_loss/val_avg_loss,
                            }
        return {'avg_val_loss': val_avg_loss,
                 'log': tensorboard_logs,
                 'progress_bar': {'val_loss': val_avg_loss}}
                 
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_dim', type=int, default=100)
        #parser.add_argument('--bias', default='store_true')
        parser.add_argument('--hidden_layer_1_dim', type=int, default=500)
        
        return parser

class MTLModelv6(MTLModelv5):
    '''
    Changes from v5:
    Removing BatchNorm from encoder and decoder layers.
    '''
    def __init__(self, hparams):
        super().__init__(hparams)
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoder(self.hparams)
            self.decoder = Decoder(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp_v4(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp_v4(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp_v4(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_mlp_v4(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v5(5, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_) ### Buggs me ...
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent) 
        H2_hat = nn.Sigmoid()(self.task_H2(latent))
        Crystal_hat = (self.task_Crystal(latent))
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]
    
## Prioritize Reconstruction as a Major Task: Manual Wt.
class MTLModelv7(MTLModelv5):
    '''
    Changes from v5:
    Removing BatchNorm from encoder and decoder layers.
    '''
    def __init__(self, hparams):
        super().__init__(hparams)
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoder(self.hparams)
            self.decoder = Decoder(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp_v4(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp_v4(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp_v4(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_mlp_v4(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v7(4, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
        if not self.hparams.train_encoder:
            recon, latent_ = self.autoencoder(X)
        else:
            latent_ = self.encoder(X)
            recon = self.decoder(nn.Tanh()(latent_))
        latent = nn.Tanh()(latent_) ### Buggs me ...
        CN_hat = self.task_CN(latent)
        Dist_hat = self.task_Dist(latent) 
        H2_hat = nn.Sigmoid()(self.task_H2(latent))
        Crystal_hat = (self.task_Crystal(latent))
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]

class MultiTaskLossWrapper_v7(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v7, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams
    def forward(self, preds, y, x):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce='none'), nn.CrossEntropyLoss()

        #sages = (age*4.75).exp_()
        
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(preds[4],y[:,15]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:,14].long()).mean()

        precision_CN = torch.exp(-self.log_vars[0])
        #loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)
        loss_CN = torch.sum(precision_CN*loss_CN + torch.log(1+self.log_vars[0].exp()), -1)

        precision_Dist = torch.exp(-self.log_vars[1])
        #loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)
        loss_Dist = torch.sum(precision_Dist*loss_Dist + torch.log(1+self.log_vars[1].exp()),-1)

        precision_H_fraction = torch.exp(-self.log_vars[2])
        #loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + torch.log(1+self.log_vars[2].exp()), -1)
        
        precision_Crystal = torch.exp(-self.log_vars[3])
        #loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + torch.log(1+self.log_vars[3].exp()),-1)
        
        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_mmd = self.loss_function(pred=reconstruction, true=x, latent=mu)
        
        precision_recon = torch.exp(-self.log_vars[4])
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction + torch.log(1+self.log_vars[4].exp()),-1)
        
        ## MMD Loss ##
        #precision_mmd = torch.exp(-self.log_vars[5])
        #loss_mmd = torch.sum(precision_mmd*loss_mmd+self.log_vars[5],-1)
        #loss_mmd = torch.sum(precision_mmd*loss_mmd + torch.log(1+self.log_vars[5].exp()),-1)

        ## Reconstruction + 0.01 MMD Loss ##
        #wtloss_reconstruction_mmd = torch.sum(precision_recon*(loss_reconstruction+self.hparams.reg_para*loss_mmd) + torch.log(1+self.log_vars[4].exp()),-1)

        ## Main Task - [Reconstruction + 0.01 MMD] Loss ##
        wtloss_reconstruction_mmd = 1000*(loss_reconstruction+self.hparams.reg_para*loss_mmd)


        ## loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal   

        return wtloss_reconstruction_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 
                                                                'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 
                                                                'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),
                                                                'train_recon_loss':loss_reconstruction.detach(), 
                                                                'loss_weight_recon':self.log_vars[4].detach(),
                                                                'train_mmd_loss':loss_mmd.detach(),
                                                                'loss_weight_MMD':self.hparams.reg_para*self.log_vars[4].detach(), ### Modified
                                                                'train_loss_recon_mmd_wt':wtloss_reconstruction_mmd.detach() ## Combined weighted Recon+MMD Loss
                                                                }

    def gaussian_kernel(self, a, b):
            
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)#.cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)#.cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        #a_core = a_core.to(self.device)
        #b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        
        return torch.exp(-numerator)

    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()
    

    def loss_function(self, pred, true, latent):
            return (pred-true).pow(2).mean(), self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)

class MTLModelv8(MTLModelv5):
    '''
    Changes from v5:
    BatchNorm from encoder and decoder layers ARE BACK.
    SOFTMAX IN CRYSTAL
    LOSS_V3: ALL IN AUTO ..LEADS TO NEGATIVE VALS.
    '''
    def __init__(self, hparams):
        super().__init__(hparams)
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_classifier(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v3(6, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
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
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]


class MultiTaskLossWrapper_v8(nn.Module):
    def __init__(self, task_num, hparams):
        super(MultiTaskLossWrapper_v8, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))
        self.hparams = hparams
    def forward(self, preds, y, x):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(reduce='none'), nn.CrossEntropyLoss()

        #sages = (age*4.75).exp_()
        
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(preds[4],y[:,15]).mean()
        loss_Crystal = crossEntropy(preds[5], target=y[:,14].long()).mean()

        precision_CN = torch.exp(-self.log_vars[0])
        #loss_CN = torch.sum(precision_CN*loss_CN + self.log_vars[0], -1)
        loss_CN = torch.sum(precision_CN*loss_CN + torch.log(1+self.log_vars[0].exp()), -1)

        precision_Dist = torch.exp(-self.log_vars[1])
        #loss_Dist = torch.sum(precision_Dist*loss_Dist + self.log_vars[1],-1)
        loss_Dist = torch.sum(precision_Dist*loss_Dist + torch.log(1+self.log_vars[1].exp()),-1)

        precision_H_fraction = torch.exp(-self.log_vars[2])
        #loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + self.log_vars[2], -1)
        loss_H_fraction = torch.sum(precision_H_fraction*loss_H_fraction + torch.log(1+self.log_vars[2].exp()), -1)
        
        precision_Crystal = torch.exp(-self.log_vars[3])
        #loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + self.log_vars[3],-1)
        loss_Crystal = torch.sum(precision_Crystal*loss_Crystal + torch.log(1+self.log_vars[3].exp()),-1)
        
        ##Reconstruction ###
        reconstruction, mu = preds[0], preds[1]
        loss_reconstruction, loss_mmd = self.loss_function(pred=reconstruction, true=x, latent=mu)
        
        precision_recon = torch.exp(-self.log_vars[4])
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction+self.log_vars[4],-1)
        #loss_reconstruction = torch.sum(precision_recon*loss_reconstruction + torch.log(1+self.log_vars[4].exp()),-1)
        
        ## MMD Loss ##
        #precision_mmd = torch.exp(-self.log_vars[5])
        #loss_mmd = torch.sum(precision_mmd*loss_mmd+self.log_vars[5],-1)
        #loss_mmd = torch.sum(precision_mmd*loss_mmd + torch.log(1+self.log_vars[5].exp()),-1)

        ## Reconstruction + 0.01 MMD Loss ##
        #wtloss_reconstruction_mmd = torch.sum(precision_recon*(loss_reconstruction+self.hparams.reg_para*loss_mmd) + torch.log(1+self.log_vars[4].exp()),-1)

        ## Main Task - [Reconstruction + 0.01 MMD] Loss ##
        wtloss_reconstruction_mmd = 1000*(loss_reconstruction+self.hparams.reg_para*loss_mmd)


        ## loss_reconstruction+loss_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal   

        return wtloss_reconstruction_mmd+loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 
                                                                'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 
                                                                'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),
                                                                'train_recon_loss':loss_reconstruction.detach(), 
                                                                'loss_weight_recon':self.log_vars[4].detach(),
                                                                'train_mmd_loss':loss_mmd.detach(),
                                                                'loss_weight_MMD':self.hparams.reg_para*self.log_vars[4].detach(), ### Modified
                                                                'train_loss_recon_mmd_wt':wtloss_reconstruction_mmd.detach() ## Combined weighted Recon+MMD Loss
                                                                }

    def gaussian_kernel(self, a, b):
            
        dim1_1, dim1_2 = a.shape[0], b.shape[0]
        depth = a.shape[1]
        a = a.view(dim1_1, 1, depth)
        b = b.view(1, dim1_2, depth)
        a_core = a.expand(dim1_1, dim1_2, depth)#.cuda()
        b_core = b.expand(dim1_1, dim1_2, depth)#.cuda()
        if torch.cuda.is_available():
            a_core = a_core.cuda()
            b_core = b_core.cuda()
        #a_core = a_core.to(self.device)
        #b_core = b_core.to(self.device)
        numerator = (a_core - b_core).pow(2).mean(2)/depth
        
        return torch.exp(-numerator)

    def MMD(self, a, b):
        return self.gaussian_kernel(a, a).mean() + self.gaussian_kernel(b, b).mean() - 2*self.gaussian_kernel(a, b).mean()
    

    def loss_function(self, pred, true, latent):
            return (pred-true).pow(2).mean(), self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)

class MTLModelv10(pl.LightningModule):
    '''
    Changes from v5:
    BatchNorm from encoder and decoder layers ARE BACK.
    SOFTMAX IN CRYSTAL
    LOSS_V3: ALL IN AUTO ..LEADS TO NEGATIVE VALS.
    '''
    def __init__(self, hparams):
        super().__init__()
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoderv2(self.hparams)
            self.decoder = Decoderv2(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = create_mlp(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = create_mlp(self.hparams.latent_size, 'Distances', hidden_layers=2).create()
        self.task_H2 = create_mlp(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = create_classifier(self.hparams.latent_size, 'Crystal', hidden_layers=2, out_features=25).create()
        self.mtl_loss_func = MultiTaskLossWrapper_v8(4, self.hparams)

    def forward(self,X):
        X = X.view(X.size(0),1,-1)
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
        
        #import pdb; pdb.set_trace()
        return [recon, latent_, CN_hat, Dist_hat, H2_hat, Crystal_hat]
    
    def setup(self, stage):
        train_dataset = XasMultiTaskDataset(json_file = self.hparams.train_json_file,
                    root_dir='../data/',
                    descriptor=self.hparams.descriptor,
                    MTL=self.hparams.MTL,
                    transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        #self.train_size = len(train_dataset)
       
        self.train_size = int(0.98*len(train_dataset))
        self.val_size = len(train_dataset)-self.train_size
        self.train_ds, self.val_ds = random_split(train_dataset,[self.train_size,self.val_size])
    
        if self.hparams.test_json_file:
            self.test_ds = XasMultiTaskDataset(json_file = self.hparams.test_json_file,
                                                root_dir='../data/',
                                                descriptor=self.hparams.descriptor,
                                                MTL=self.hparams.MTL,
                                                transform=Compose([XasInterpolate(), XasNormalize(), XasToTensor()]))
    
        exp_dataset = XasExpDataset(json_file=self.hparams.exp_json_file,
                                 root_dir = '../data/',
                                 descriptor = 'Distances',
                                 transform = Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
        
        self.exp_ds = exp_dataset
        
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, 
                    weight_decay=self.hparams.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 10, 1)

        return [optimizer], [scheduler]
    
    @pl.data_loader  
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.cpus, drop_last=False, pin_memory=True)
    
    @pl.data_loader
    def val_dataloader(self):
        theo_DL =  DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.cpus, drop_last=False, pin_memory=True)
        exp_DL =  DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.cpus, drop_last=False, pin_memory=True)
        return [theo_DL, exp_DL]
    
    @pl.data_loader
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.cpus, drop_last=False, pin_memory=True)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # flatten any input
        #x = x.view(x.size(0), -1)
        ## 
        y_hat = self.forward(x) ## 
        

        ## MTL Loss
        loss, loss_log = self.mtl_loss_func(y_hat, y, x)
        
        tensorboard_logs = {'train_loss': loss}#, 
        progress_bar_metrics = tensorboard_logs
        
        return {
            'loss': loss,
            'log': {**tensorboard_logs, 'train_mtl_loss':loss_log},
            'progress_bar': progress_bar_metrics
        }
    
    def training_epoch_end( self, outputs):
        
        train_losses = torch.stack([x['log']['train_loss'] for x in outputs])
        train_avg_loss = train_losses.detach().mean()
        train_std_loss = train_losses.detach().std()
        train_recon_loss_epoch = torch.stack([x['log']['train_mtl_loss']['train_recon_loss'] for x in outputs]).detach().mean()
        train_mmd_loss_epoch = torch.stack([x['log']['train_mtl_loss']['train_mmd_loss'] for x in outputs]).detach().mean()
        
        tensorboard_logs = {'train_epoch_loss':train_avg_loss,
                            'training_loss_std':train_std_loss,
                            'normalized_train_loss_std':train_std_loss/train_avg_loss,
                            'train_recon_loss_per_epoch':train_recon_loss_epoch,
                            'train_mmd_loss_per_epoch':train_mmd_loss_epoch}
        return {'log': tensorboard_logs}

    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        
        x, y = batch
        # Flatten input for linear layers
        #x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        
        ## Caluculate only reconstruction and MMD loss
        recon_loss, mmd_loss = self.mtl_loss_func.loss_function(pred=y_hat[0], latent=y_hat[1], true=x)
        
        ## Combine based on reg_para with MMD Loss
        val_loss = recon_loss #+ self.hparams.reg_para*mmd_loss

        tensorboard_logs = {'val_recon_loss': recon_loss, 'val_mmd_loss': mmd_loss}
        return {'val_loss': val_loss, 
                'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        #import pdb; pdb.set_trace()
        val_losses = torch.stack([x['val_loss'] for dls in outputs for x in dls])
        val_avg_loss = val_losses.detach().mean()
        val_std_loss = val_losses.detach().std()
        
        tensorboard_logs = {'val_loss': val_avg_loss,
                            'val_loss_std':val_std_loss, 
                            'normalized_val_loss_std':val_std_loss/val_avg_loss,
                            }
        return {'avg_val_loss': val_avg_loss,
                 'log': tensorboard_logs,
                 'progress_bar': {'val_loss': val_avg_loss}}
    
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        output = self.validation_step(batch, batch_idx, dataloader_idx)
        # Rename output keys
        #output['test_loss'] = output.pop('val_loss')


        return output

        #return {'test_loss': loss}

    def test_epoch_end(self, outputs):
        
        test_loss = torch.stack([x['val_loss'] for dls in outputs for x in dls]).detach().mean()
        tensorboard_logs = {'test_epoch_loss':test_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'test_loss': test_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_dim', type=int, default=100)
        #parser.add_argument('--bias', default='store_true')
        parser.add_argument('--hidden_layer_1_dim', type=int, default=500)
        
        return parser
    
    ##------##