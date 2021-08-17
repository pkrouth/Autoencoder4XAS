from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
from .lightningVAE import *
from .lightningAE import *
from .lightningXAS import *
from .MTL import *



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
    
class MTLModelv2(MTLModel):
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
    

