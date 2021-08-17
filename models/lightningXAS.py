from ..utils import *
from ..xasdata import *
from ..utils.transforms import *
from torch.nn import BatchNorm1d, Dropout
from .lightningVAE import *
from .lightningAE import *


import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.optimizer import Optimizer


#
#
class Lin2AEXas(VaeMmdConv3Lin1XAS):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            #nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            #nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim), 
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.latent_size))
        self.decoder = nn.Sequential(
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.latent_size, self.hparams.hidden_layer_2_dim),
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            #nn.Linear(self.hparams.hidden_layer_3_dim, self.hparams.hidden_layer_2_dim),
            #nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_1_dim, 10 * 10),
            )
    def forward(self, x):
        x = self.encoder(x)
        latent = x
        x = nn.ReLU()(x)
        #x = nn.BatchNorm1d(1)(x)
        x = self.decoder(x)
        return x, latent

   
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed, _ = self.forward(x)
        loss = nn.MSELoss()(reconstructed, x)#, reduction='mean')##Removing reduce here...
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}
    
    def training_epoch_end( self, outputs):
        train_losses = torch.stack([x['train_loss'] for x in outputs])
        train_avg_loss = train_losses.mean()
        train_std_loss = train_losses.std()
        tensorboard_logs = {'train_epoch_loss':train_avg_loss, 'training_loss_std':train_std_loss, 'normalized_train_loss_std':train_std_loss/train_avg_loss ,'step': self.current_epoch+1}
        return {'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed, _ = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        # add logs
        logs = {'loss':{'val_loss': loss}}
        return {'val_loss': loss, 'log': logs}
    
    def validation_epoch_end(self, outputs):
        #print(outputs)
        #pdb.set_trace()
        val_losses = torch.stack([x['val_loss'] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        tensorboard_logs = {'val_loss': val_avg_loss, 'val_loss_std':val_std_loss, 'normalized_val_loss_std':val_std_loss/val_avg_loss ,'step': self.current_epoch+1}
        return {'avg_val_loss': val_avg_loss, 'log': tensorboard_logs}
    
    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_epoch_end(outputs)
        # rename some keys
        results['log'].update({
            'test_loss': results['log'].pop('val_loss'),
            'test_loss_std': results['log'].pop('val_loss_std'),
            'normalized_test_loss_std': results['log'].pop('normalized_val_loss_std'),
            #'test_recon_loss_per_epoch': results['log'].pop('val_recon_loss_per_epoch'),
            #'test_mmd_loss_per_epoch': results['log'].pop('val_mmd_loss_per_epoch'),
        })
        results['avg_test_loss'] = results.pop('avg_val_loss')


        return results
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--hidden_layer_1_dim', default=128, type=int)
        parser.add_argument('--hidden_layer_2_dim', default=64, type=int)
        parser.add_argument('--hidden_layer_3_dim', default=12, type=int)
        parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL 
        #parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        #parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        #parser.add_argument('--learning_rate', default=0.001, type=float)
        #parser.add_argument('--batch_size', default=32, type=int)
        #parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--latent_size', default=3, type=int)
        # training specific (for this model)
        return parser


class Lin2AEXasL1(VaeMmdConv3Lin1XAS):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.hparams = hparams
        self.encoder = nn.Sequential(
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(10 * 10, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            #nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim),
            #nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_1_dim, self.hparams.hidden_layer_2_dim), 
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.latent_size))
        self.decoder = nn.Sequential(
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.latent_size, self.hparams.hidden_layer_2_dim),
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            #nn.Linear(self.hparams.hidden_layer_3_dim, self.hparams.hidden_layer_2_dim),
            #nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_2_dim, self.hparams.hidden_layer_1_dim),
            nn.ReLU(True),
            #nn.BatchNorm1d(1),
            #nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_layer_1_dim, 10 * 10),
            )
    def forward(self, x):
        x = self.encoder(x)
        latent = x
        x = nn.ReLU()(x)
        #x = nn.BatchNorm1d(1)(x)
        x = self.decoder(x)
        return x, latent

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
   
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed, _ = self.forward(x)
        loss = nn.MSELoss()(reconstructed, x) #, reduction='mean')##Removing reduce here...
        tensorboard_logs = {'train_loss':loss}
        return {'loss':loss, 'log':tensorboard_logs}
    
    def training_epoch_end( self, outputs):
        train_losses = torch.stack([x['train_loss'] for x in outputs])
        train_avg_loss = train_losses.mean()
        train_std_loss = train_losses.std()
        tensorboard_logs = {'train_epoch_loss':train_avg_loss, 'training_loss_std':train_std_loss, 'normalized_train_loss_std':train_std_loss/train_avg_loss ,'step': self.current_epoch+1}
        return {'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        reconstructed, _ = self(x)
        loss = nn.MSELoss()(reconstructed, x)
        # add logs
        logs = {'loss':{'val_loss': loss}}
        return {'val_loss': loss, 'log': logs}
    
    def validation_epoch_end(self, outputs):
        #print(outputs)
        #pdb.set_trace()
        val_losses = torch.stack([x['val_loss'] for x in outputs])
        val_avg_loss = val_losses.mean()
        val_std_loss = val_losses.std()
        tensorboard_logs = {'val_loss': val_avg_loss, 'val_loss_std':val_std_loss, 'normalized_val_loss_std':val_std_loss/val_avg_loss ,'step': self.current_epoch+1}
        return {'avg_val_loss': val_avg_loss, 'log': tensorboard_logs}
    
    def test_epoch_end(self, outputs):
        """
        Called at the end of test to aggregate outputs, similar to `validation_epoch_end`.
        :param outputs: list of individual outputs of each test step
        """
        results = self.validation_epoch_end(outputs)
        # rename some keys
        results['log'].update({
            'test_loss': results['log'].pop('val_loss'),
            'test_loss_std': results['log'].pop('val_loss_std'),
            'normalized_test_loss_std': results['log'].pop('normalized_val_loss_std'),
            #'test_recon_loss_per_epoch': results['log'].pop('val_recon_loss_per_epoch'),
            #'test_mmd_loss_per_epoch': results['log'].pop('val_mmd_loss_per_epoch'),
        })
        results['avg_test_loss'] = results.pop('avg_val_loss')


        return results
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--hidden_layer_1_dim', default=128, type=int)
        parser.add_argument('--hidden_layer_2_dim', default=64, type=int)
        parser.add_argument('--hidden_layer_3_dim', default=12, type=int)
        parser.add_argument('--input_dim', default=100, type=int)

        ## CONV MODEL 
        #parser.add_argument('--fc_layer_1_dim', default=64, type=int)
        #parser.add_argument('--dropout', default=0.5, type=float)
        ## GENERIC
        #parser.add_argument('--learning_rate', default=0.001, type=float)
        #parser.add_argument('--batch_size', default=32, type=int)
        #parser.add_argument('--max_epochs', default=100, type=int)
        parser.add_argument('--latent_size', default=3, type=int)
        # training specific (for this model)
        return parser

class Vae2Dsc(pl.LightningModule):

    def __init__(self,
                 hparams,
                 #input_dim: int,
                 #output_dim:int,
                 bias: bool = True,
                 learning_rate: float = 0.0001,
                 optimizer: Optimizer = Adam,
                 l1_strength: float = None,
                 l2_strength: float = None,
                 best_model = None,
                 MultiTask = None,
                 Descriptor = 'CN',
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
        self.Descriptor = Descriptor
        ## Linear 1 Layer Model
        self.layer_1 = torch.nn.Linear(in_features=self.hparams.latent_size, out_features=256)
        self.layer_2 = torch.nn.Linear(in_features=256, out_features=1)

        ## Encoder
        if self.hparams.best_model_path is not None:
            self.best_model = self._get_model()
            #self.best_model.eval()
            self.best_model.freeze()

    def forward(self, x):
        x = x.view(-1,1,100)
        if self.best_model is not None:
            x = self.best_model.encoder(x)
            x = x.squeeze()
        x = self.layer_1(x)
        x= nn.ReLU()(x)
        y_hat = self.layer_2(x)
        return y_hat

    def _get_model(self):
        model_name = self.hparams.best_model_type
        filepath = self.hparams.best_model_path
        MODEL_CHECKPOINT_PATH = glob.glob(filepath)[0]
        print("...Loading Model Checkpoint from: ", MODEL_CHECKPOINT_PATH, '\n')
        model = model_name.load_from_checkpoint(MODEL_CHECKPOINT_PATH)
        return model

    def prepare_data(self):
            train_dataset = XasDataset(json_file=self.hparams.json_file,
                                    root_dir='../data/',
                                    descriptor=self.Descriptor,
                                    transform=Compose([XasInterpolate(), XasNormalize(),XasToTensor()]))
            self.train_size = int(0.98*len(train_dataset))
            self.val_size = len(train_dataset)-self.train_size
            self.train_ds, self.val_ds = random_split(train_dataset,[self.train_size,self.val_size])
            if self.hparams.test_json_file:
                self.test_ds = XasDataset(json_file=self.hparams.json_file,
                                    root_dir='../data/',
                                    descriptor=self.Descriptor,
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
        # Flatten input for linear layers
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
        # flatten any input
        x = x.view(x.size(0), -1)
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
