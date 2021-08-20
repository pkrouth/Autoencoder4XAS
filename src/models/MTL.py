from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
from .lightningVAE import *
from .lightningAE import *
from .lightningXAS import *



import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer

class XasMultiTaskDataset(Dataset):
    """ Custom XAS Dataset"""
    def __init__(self,json_file, root_dir, descriptor = 'CN', transform=None, MTL=True):
        self.XAS_Frame = pd.read_json(json_file, orient='records')
        #self.XAS_Frame.Abs = self.XAS_Frame.Abs.apply(lambda x: x/np.max(x)) ## Questionable Normalization
        self.root_dir = root_dir
        self.transform = transform
        #if type(descriptor)==type(''):
        self.descriptor = descriptor
        ## Need Improvement with possible One-Hot Encoding
        self.MTL = MTL
        self.XAS_Frame['Crystal'] = (self.XAS_Frame.Crystal.apply(lambda x: x[0]).astype('category'))
        self.categories = self.XAS_Frame.Crystal.cat.categories
        self.XAS_Frame.Crystal = self.XAS_Frame.Crystal.cat.codes
        self.XAS_Frame['MTL_Label'] = self.XAS_Frame[['CN', 'Distances', 'H_fraction', 'Crystal']].apply(lambda x: x.CN+x.Distances+[x.Crystal, x.H_fraction], axis=1)
        
    def __len__(self):
        return len(self.XAS_Frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if type(idx)==int:
            idx=[idx]
        data_Energy = self.XAS_Frame['Energy'].iloc[idx]
        data_Abs    = self.XAS_Frame['Abs'].iloc[idx]
        if self.MTL:
            data_Y      = self.XAS_Frame['MTL_Label'].iloc[idx]#.to_numpy()
        else:
            data_Y      = self.XAS_Frame[self.descriptor].iloc[idx]#.to_numpy()
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
        _ = sns.lineplot(x='x_axis', y='y_axis', data=data)
        _ = plt.title(f"Sample_#{idx}_{'MTL[CN+Distances+Crystal+H_fraction]:' if self.MTL else self.descriptor}{y}")
    
class MTLModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        #self.save_parameters() 
        self.hparams = hparams
        if self.hparams.train_encoder:
            self.encoder = Encoder(self.hparams)
            self.decoder = Decoder(self.hparams)
        else:
            self.autoencoder = self._get_model()
            self.autoencoder.freeze()
        self.task_CN = _create_mlp(self.hparams.latent_size, 'CN', hidden_layers=2).create()
        self.task_Dist = _create_mlp(self.hparams.latent_size, 'Distances', hidden_layers=1).create()
        self.task_H2 = _create_mlp(self.hparams.latent_size, 'H_fraction', hidden_layers=2).create()
        self.task_Crystal = _create_classifier(self.hparams.latent_size, 'Crystal', hidden_layers=2).create()
        self.mtl_loss_func = MultiTaskLossWrapper(4)

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

    def prepare_data(self):
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
        loss, loss_log = self.mtl_loss_func(y_hat, y)

        reconstruction, mu = y_hat[0], y_hat[1]
        if self.hparams.train_encoder:
            reconstruction_error, mmd = self.loss_function(reconstruction, x, mu)
            encoder_loss = reconstruction_error + mmd
        else:
            encoder_loss = 0.
            reconstruction_error = 0.
            mmd = 0.
        
        
        loss = 1000*encoder_loss + loss
        tensorboard_logs = {'train_loss': loss}#, 
        progress_bar_metrics = tensorboard_logs
        
        return {
            'loss': loss,
            'log': {**tensorboard_logs, 'train_mtl_loss':loss_log, 
            'train_recon_loss':reconstruction_error, 'train_mmd_loss':mmd},
            'progress_bar': progress_bar_metrics
        }
    
    def training_epoch_end( self, outputs):
        train_losses = torch.stack([x['log']['train_loss'] for x in outputs])
        train_avg_loss = train_losses.detach().mean()
        train_std_loss = train_losses.detach().std()
        train_recon_loss_epoch = torch.stack([x['log']['train_recon_loss'] for x in outputs]).detach().mean()
        train_mmd_loss_epoch = torch.stack([x['log']['train_mmd_loss'] for x in outputs]).detach().mean()
        
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
        loss, loss_log = self.mtl_loss_func(y_hat, y)

        reconstruction, mu = y_hat[0], y_hat[1]
        if self.hparams.train_encoder:
            reconstruction_error, mmd = self.loss_function(reconstruction, x, mu)
            encoder_loss = reconstruction_error + mmd
        else:
            encoder_loss = 0.
            reconstruction_error = 0.
            mmd = 0.
        
        
        loss = 1000*encoder_loss + loss


        tensorboard_logs = {'val_mtl_loss':loss_log}
        return {'val_loss': loss, 'log': tensorboard_logs, 
        'val_recon_loss':reconstruction_error, 'val_mmd_loss': mmd}

    def validation_epoch_end(self, outputs):

        val_losses = torch.stack([x['val_loss'] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        val_recon_loss_epoch = torch.stack([x['val_recon_loss'] for x in outputs]).mean()
        val_mmd_loss_epoch = torch.stack([x['val_mmd_loss'] for x in outputs]).mean()
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
    
    def L1_regularization(self,input):
        loss=0
        #values = torch.from_numpy(input)#.float()
        values = input
        for l in self.encoder.children():
            values = nn.LeakyReLU()(l(values))
            loss+=torch.mean(abs(values))
        values = nn.Tanh()(values)
        for l in self.decoder.children():
            values = nn.ReLU()(l(values))
            loss+=torch.mean(abs(values))
        return loss 

    def loss_function(self, pred, true, latent):
        if self.hparams.sparsity:
            return (pred-true).pow(2).mean(), (self.hparams.reg_para)*(self.L1_regularization(true))
        else:
            return (pred-true).pow(2).mean(), self.hparams.reg_para*self.MMD(a=torch.randn(self.hparams.batch_size, self.hparams.latent_size, requires_grad = False), b=latent)
        #mse_loss = nn.MSELoss(reduce='mean')(pred, true)
        #mmd_loss = self.compute_mmd(a=true, b=latent)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--input_dim', type=int, default=100)
        #parser.add_argument('--bias', default='store_true')
        parser.add_argument('--hidden_layer_1_dim', type=int, default=500)
        
        return parser

class _create_mlp(nn.Module):
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

class _create_classifier(nn.Module):
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
    

#https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example-pytorch.ipynb
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, y):
        #[recon, latent, CN_hat, Dist_hat, H2_hat, Crystal_hat]
        mse, crossEntropy = nn.MSELoss(), nn.CrossEntropyLoss()
        
        #sages = (age*4.75).exp_()
        loss_CN = mse(preds[2], y[:,0:6]).mean()
        loss_Dist = mse(preds[3],y[:,7:13]).mean()
        loss_H_fraction = mse(preds[4],y[:,15]).mean()
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
        
        
        return loss_CN+loss_Dist+loss_H_fraction+loss_Crystal, {'loss_CN':loss_CN.detach(), 'loss_Distances': loss_Dist.detach(),
                                                                'loss_H_fraction': loss_H_fraction.detach(), 'loss_Crsytal': loss_Crystal.detach(),
                                                                'loss_weights_CN':self.log_vars[0].detach(),
                                                                'loss_weights_Distances':self.log_vars[1].detach(),
                                                                'loss_weights_Crystal':self.log_vars[3].detach(),
                                                                'loss_weights_H_fraction':self.log_vars[2].detach(),}

class Encoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self._encoder = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=3,stride=1,padding=1), # Input: (bs,nc=1,l=100) Output: (bs,8,100)
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=1), # Output: #bs, 8, 98
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2, stride=2, padding=1),  # Output: b, 16, 50
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=2),  # Output: b, 16, 24
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=1),  # Output: b, 32, 10
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=3, stride=3),# Output: b, 32, 3
            Flatten(),
            nn.Linear(in_features = 3*32, out_features = self.hparams.hidden_layer_1_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features = self.hparams.hidden_layer_1_dim, out_features = hparams.latent_size)
        )
    
    def forward(self, X):
        return self._encoder(X)

class Decoder(nn.Module):
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


class VaeMtl2Dsc(pl.LightningModule):## Maybe Vae2Dsc is needed
    
    def __init__(self, hparams, 
                input_dim = 100, 
                Descriptor='CN',
                TaskList = None):
        super().__init__(hparams=hparams, input_dim=input_dim, Descriptor=Descriptor)
        self.save_hyperparameters()
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)



    def forward(self, X):
        #if self.training:
        latent = self.encoder(X)
        reconstruction = self.decoder(latent)


        return self.decoder(nn.Tanh()(latent)), latent ## Added Tanh
        #else:
        #    return self.decoder( self.encoder(X) )
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        #parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--hidden_layer_1_dim', type=int, default=500)
        parser.add_argument('--input_dim', type=int, default=100)
        parser.add_argument('--bias', default='store_true')
        #parser.add_argument('--batch_size', type=int, default=50)
        return parser


class LinearRegression(pl.LightningModule):
    def __init__(self,
                 hparams,
                 input_dim: int,
                 #output_dim:int,
                 bias: bool = True,
                 learning_rate: float = 0.0001,
                 optimizer: Optimizer = Adam,
                 l1_strength: float = None,
                 l2_strength: float = None,
                 
                 **kwargs):
        """
        Linear regression model implementing - with optional L1/L2 regularization
        $$min_{W} ||(Wx + b) - y ||_2^2 $$
        Args:
            input_dim: number of dimensions of the input (1+)
            bias: If false, will not use $+b$
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        self.hparams=hparams
        self.layer_1 = torch.nn.Linear(in_features=100, out_features=256)
        self.layer_2 = torch.nn.Linear(in_features=256, out_features=7)

    def forward(self, x):
        x = self.layer_1(x)
        x= nn.ReLU()(x)
        y_hat = self.layer_2(x)
        return y_hat

    def prepare_data(self):
            train_dataset = XasDataset(json_file=self.hparams.json_file,
                                    root_dir='../data/',
                                    descriptor='Distances',
                                    transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
            self.train_size = int(0.98*len(train_dataset))
            self.val_size = len(train_dataset)-self.train_size
            self.train_ds, self.val_ds = random_split(train_dataset,[self.train_size,self.val_size])
            if self.hparams.test_json_file:
                self.test_ds = XasDataset(json_file=self.hparams.json_file,
                                    root_dir='../data/',
                                    descriptor='Distances',
                                    transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
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
        x = x.float()
        y = y.float()
        # flatten any input
        x = x.view(x.size(0), -1)

        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        # L1 regularizer
        if self.hparams.l1_strength is not None:
            l1_reg = torch.tensor(0.)
            for param in self.parameters():
                l1_reg += torch.norm(param, 1)
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength is not None:
            l2_reg = torch.tensor(0.)
            for param in self.parameters():
                l2_reg += torch.norm(param, 2)
            loss += self.hparams.l2_strength * l2_reg

        tensorboard_logs = {'train_mse_loss': loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'loss': loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        return {'val_loss': F.mse_loss(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_mse_loss': val_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'val_loss': val_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.mse_loss(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_mse_loss': test_loss}
        progress_bar_metrics = tensorboard_logs
        return {
            'test_loss': test_loss,
            'log': tensorboard_logs,
            'progress_bar': progress_bar_metrics
        }

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        #parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=100)
        parser.add_argument('--bias', default='store_true')
        #parser.add_argument('--batch_size', type=int, default=50)
        return parser
