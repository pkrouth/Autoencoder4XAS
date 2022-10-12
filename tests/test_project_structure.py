import unittest


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

        # from encodex.utils.imports import *


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
