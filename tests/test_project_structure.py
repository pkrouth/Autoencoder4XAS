import unittest
from xas_encoder.models.autoencoders import * 
from xas_encoder.models.lightning import *
from xas_encoder.models.lightningAE import *
import warnings
warnings.simplefilter('always')
print(f"Using {pl.__version__=}")

class TestCase(unittest.TestCase):
    def test_unittest(self):
        assert True

    def test_import(self):
        import os
        import torch
        from torch import nn
        import torch.nn.functional as F
        from torchvision import transforms
        from torchvision.datasets import MNIST
        from torch.utils.data import DataLoader, random_split
        import pytorch_lightning as pl

    def test_recon_shape(self, model=None, recon_index=None):
        x = torch.rand(10,100)
        if model is None:
            model = autoencoder()
    
        y = model(x)

        if recon_index is not None:
            assert x.shape == y[recon_index].shape
        else:
            assert x.shape == y.shape

    def test_recon_shape_convae(self, model=None, recon_index=None):
        x = torch.rand(10, 1,100)
        if model is None:
            model = Conv1Dautoencoder(latent_size=10)
        y = model(x)
        if recon_index is not None:
            assert x.shape == y[recon_index].shape
        else:
            assert x.shape == y.shape

    # def test_multiple_latent_sizes(self):
    #     for size in range(10, 50, 10):
    #         self.test_base_convae_model(ls=size)
    
    def test_models_with_PL(self, MODEL=None):

        PARAMS = {'func':'XAS',
                    'x_data':'energy_mesh',
                    'json_file':'../data/20200728_Dataset_Pd_ver1_augmented_train.json',
                    'train_json_file':'../data/20200728_Dataset_Pd_ver1_augmented_train.json',
                    'test_json_file':'../data/20200728_Dataset_Pd_ver1_augmented_test.json',
                    'exp_json_file' :'../data/raw_data/dataset_Pd_exp.json',
                    #   'best_model_path': './best_model/VaeMmdConv3Lin1xas/_ckpt_epoch_1129.ckpt',
                    #'best_model_type':VaeMmdConv3Lin1XAS,
                    'latent_size': 10
                    }
            
            
        #CHECKPOINTS_DIR = f'my_models/checkpoints/20200728/'
        PARAMETERS={**PARAMS,
                    # 'Model': 'MTLModelv5',
                    # 'loss_function':'MSE_Loss+MMD',
                    # 'optimizer':'Adam',
                    'weight_decay':0.,  
                    # 'name_exp':'default',
                    # 'MTL':True,
                    # 'train_encoder': True,
                    # 'descriptor':'H_fraction'
                    'learning_rate': 0.001,
                    'dropout': 0.5,
                    # 'hidden_layer_1_dim': 256,
                    # 'hidden_layer_2_dim': 64,
                    # 'hidden_layer_3_dim': 32,
                    'input_dim': 100,
                    'latent_size': 10,
                    'fc_layer_1_dim': 256,
                    }

        

        if MODEL is None:
            MODEL=XasLinEncoders
        model = MODEL(hparams=PARAMETERS)
        if model:
            assert True
        else:
            assert False

    def test_all_PL_models(self):
        for m in [XasLinEncoders, XasLinAutoEncoder2, XasConvAutoEncoder, XasConvAutoEncoder2,
        Lin3AE, Conv3FC1AE, Lin2AE1, Lin2AE2, Lin2AE3, 
        Lin2AE4, Conv1FC2AE1, Lin2AEBN
        ]:
            print(f"Testing Model: {m.__name__}")
            self.test_models_with_PL(MODEL=m)
            print(f"Model: {m.__name__} passed")
            # if 'Conv' in m.__name__:
            #     self.test_recon_shape_convae(model=m, recon_index=None)
            # else:
            #      self.test_recon_shape(model=m, recon_index=None)

        # self.test_recon_shape(model=Lin3AE, recon_index=None)
        # self.test_recon_shape(model=Lin2AEXas, recon_index=1)
        
#    def test_project_import(self):
#        import xafscouleurs

#    def test_project_submodule_import(self):
#        from xafscouleurs.utils.datamodules import XAFSDataModule
#        from xafscouleurs.utils import imports
#        from xafscouleurs.models.ANNs import Rouge, Bleu
#        from xafscouleurs.functional.extests import RougeTest
#        from xafscouleurs.functional.trainer import step, train

#    def test_submodules(self):
#        from xafscouleurs.functional.trainhandlers import JauneTrainPL, RougeTrain
