import xas_encoder
from xas_encoder.utils.imports import *
from xas_encoder.models.lightningAE import Conv1FC2AE1
# from xas_encoder.models.lightningVAE import VaeMmdConv3Lin1XAS
import warnings
warnings.filterwarnings('ignore')
SEED = 2334

import pytorch_lightning as pl
pl.seed_everything(SEED)

PATH_DATASETS = os.environ.get("PATH_DATASETS", "../DATA/")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

### DO NOT CHANGE ###
### This version is created to test the conda environment and CUDA compatibility.
### Current Status: CUDA works.
### Feature addition: Custom Checkpoint


def main(hyperparams):
    
    
    ####_____Add information about data generation____
    PARAMS = {'func':'XAS',
                     'x_data':'energy_mesh',
              'train_json_file':'../data/20200728_Dataset_Pd_ver1_augmented_train.json',
              'test_json_file':'../data/20200728_Dataset_Pd_ver1_augmented_test.json',
              'exp_json_file' :'../data/raw_data/dataset_Pd_exp.json',
            #   'best_model_path': './best_model/VaeMmdConv3Lin1xas/_ckpt_epoch_1129.ckpt',
            #   'best_model_type':VaeMmdConv3Lin1XAS,
            #   'latent_size': 10
             }
    
    
    #CHECKPOINTS_DIR = f'my_models/checkpoints/20200728/'
    PARAMETERS={**PARAMS,
                'Model': 'Conv1FC2AE1',
                'loss_function':'MSE_Loss+MMD',
                'optimizer':'Adam',
                'weight_decay':0.,  
                'name_exp':'default',
                'MTL':True,
                'train_encoder': True,
                'descriptor':'H_fraction'}

  
    
    ### Model
    from argparse import Namespace
    # hparams = Namespace(**PARAMETERS, **vars(hyperparams))
    hparams = vars(hyperparams)
    hparams.update(**PARAMETERS)
    hparams = Namespace(**hparams)
    print(hparams)
    #model = LinearRegression(input_dim= hparams.input_dim, hparams=hparams).float() ## Change below as well
    model = Conv1FC2AE1(hparams=vars(hparams)).float()
    ## Debug
    #torch.autograd.set_detect_anomaly(True)
    
    ### Loggers
    # from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
    # import os
    # if hparams.wandb_id is not None:
    #     os.environ["WANDB_RESUME"] = "allow"
    # else:
    #     id=None

    # #tb_logger = TensorBoardLogger("tb_logs", name="test_log_model")
    # wandblogger = WandbLogger(name = f'MTL_v4_Full_Training_LS{hparams.latent_size}_1+log_var',
    #                      #save_dir = './wand_logs_Cu/',
    #                      id=hparams.wandb_id,offline = False,project = 'MTL',tags = ['test', 'dataset_Pd', 'ver3',  'CPU', 'ReLU', 'Normalized'],log_model=True)
    # logger = wandblogger # neptune_logger#[tb_logger, neptune_logger]





    ### Add CallBacks
    # model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=logger.experiment.dir+'/'+logger.experiment.project+'/checkpoints/',monitor='val_loss' )
    # custom_early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=200, verbose=True, mode='min')

    ### Initialize Training
    trainer = pl.Trainer(fast_dev_run=hparams.fast_dev_run, 
                      track_grad_norm=hparams.track_gradient,
                      profiler=hparams.profiler,
                      gradient_clip_val=hparams.clip_grad,
                      max_epochs=hparams.max_epochs,
                      min_epochs=hparams.min_epochs,
                    #   logger=logger, #neptune_logger,
                    #   resume_from_checkpoint=hparams.checkpoint if hparams.checkpoint else None,
                      auto_lr_find=False, 
                    #   early_stop_callback=custom_early_stop_callback,
                      gpus=hparams.gpus,
                      #cpus=hparams.cpus,`
                    #   distributed_backend=hparams.distributed_backend,
                    #   precision=16 if hparams.use_16bit else 32,
                      train_percent_check=hparams.train_percent_check,
                      val_percent_check=hparams.val_percent_check,
                    #   checkpoint_callback=model_checkpoint ## Added Train Loss monitor
                    )


    trainer.fit(model)
    # trainer.test()





if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
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



