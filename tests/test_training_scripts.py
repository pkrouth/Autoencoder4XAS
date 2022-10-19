import unittest
from train_MNIST import *
from trainer_test import *
class TestScript(unittest.TestCase):
    def test_self(self):
        assert True

    def test_MNIST(self):
        run(3)
        assert True
    def test_trainer_v0(self):
        from argparse import ArgumentParser
        parent_parser = ArgumentParser(add_help=False)
        parent_parser.add_argument('--wandb_id', type=str, default=None, help='resume to wandb')
        parent_parser.add_argument('--batch_size', type=int, default=32, help='increase by n*32 if n GPUs')
        parent_parser.add_argument('--train_percent_check', type=float, default=0.01, help='For quick iteration lower the numbers')
        parent_parser.add_argument('--val_percent_check', type=float, default=1.0, help='For quick iteration lower the numbers')
        parent_parser.add_argument('--checkpoint', type=str, default=None, help='resume from checkpoint')
        # gpu args
        parent_parser.add_argument('--gpus', default=None, help='how many gpus')
        parent_parser.add_argument('--cpus', type=int, default=4, help='how many cpus')
        parent_parser.add_argument('--distributed_backend', type=str, default='None', help='supports three options dp, ddp, ddp2')

        ## Regularization
        parent_parser.add_argument('--sparsity', type=bool, default=False, help='add L1 Regularization')
        parent_parser.add_argument('--reg_para', type=float, default=0., help='Co-effecoient for L1 regularization')
        
        ## 
        parent_parser.add_argument('--fast_dev_run', type=bool, default=False, help='dry run')
        parent_parser.add_argument('--profiler', type=bool, default=False, help='debug speed')
        parent_parser.add_argument('--clip_grad', type=float, default=1, help='gradient clip value')
        parent_parser.add_argument('--max_epochs', type=int, default=5, help='maximum number of epochs to run (early stop can happen)')
        parent_parser.add_argument('--min_epochs', type=int, default=10, help='minimum number of epochs to run (early stop can happen)')
        parent_parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    
        # Testing args
        parent_parser.add_argument('--use_16bit', dest='use_16bit', action='store_true', help='if true uses 16 bit precision')
    
        parent_parser.add_argument('--track_gradient', type=int, default=-1,help='For quick iteration set False')
        # each LightningModule defines arguments relevant to it
        parser = Conv1FC2AE1.add_model_specific_args(parent_parser)
        hyperparams = parser.parse_args()

        # ---------------------
        # RUN TRAINING
        # ---------------------
        main(hyperparams)

        assert True

