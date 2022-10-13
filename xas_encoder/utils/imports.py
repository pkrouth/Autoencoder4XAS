# Imports

import torch
import torch.nn as nn
import torch.nn.functional as F

##PyTorch Lightning
import pytorch_lightning as pl
from torch.optim import Adam
from argparse import ArgumentParser
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split

## Logging: Tensorboard, livelossplot, Neptune
from torch.utils.tensorboard import SummaryWriter

# from livelossplot import PlotLosses
# import neptune
import wandb

## Plot Tools
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#%matplotlib inline
import seaborn as sns

# sns.set_style('darkgrid')


import os
from ast import literal_eval
from torch.autograd import Variable
from sklearn.decomposition import PCA


global energymesh

# For Pd
energymesh = [
    24339.1,
    24339.8,
    24340.5,
    24341.0,
    24341.5,
    24342.0,
    24342.3,
    24342.6,
    24342.8,
    24342.9,
    24342.95,
    24343.0,
    24343.1,
    24343.15,
    24343.3,
    24343.4,
    24343.5,
    24343.7,
    24343.9,
    24344.1,
    24344.3,
    24344.5,
    24344.8,
    24345.1,
    24345.3,
    24345.7,
    24346.0,
    24346.3,
    24346.7,
    24347.1,
    24347.5,
    24347.9,
    24348.4,
    24348.9,
    24349.3,
    24349.9,
    24350.4,
    24350.9,
    24351.5,
    24352.1,
    24352.7,
    24353.3,
    24353.9,
    24354.6,
    24355.3,
    24356.0,
    24356.7,
    24357.4,
    24358.1,
    24358.9,
    24359.7,
    24360.5,
    24361.4,
    24362.2,
    24363.1,
    24364.0,
    24364.9,
    24365.8,
    24366.7,
    24367.7,
    24368.7,
    24369.7,
    24370.7,
    24371.7,
    24372.8,
    24373.9,
    24375.0,
    24376.1,
    24377.2,
    24378.4,
    24379.5,
    24380.7,
    24381.9,
    24383.2,
    24384.4,
    24385.7,
    24387.0,
    24388.3,
    24389.6,
    24390.9,
    24392.3,
    24393.7,
    24395.1,
    24396.5,
    24397.9,
    24399.4,
    24400.9,
    24402.4,
    24403.9,
    24405.4,
    24407.0,
    24408.5,
    24410.1,
    24411.7,
    24413.4,
    24415.0,
    24416.7,
    24418.4,
    24420.1,
    24421.8,
]

# For Cu, Cu Oxides
# energymesh = [8979.42, 8980.144, 8980.792, 8981.363, 8981.859, 8982.278, 8982.621, 8982.887, 8983.078, 8983.192, 8983.23, 8983.33, 8983.43, 8983.468, 8983.573, 8983.697, 8983.84, 8984.002, 8984.183, 8984.383, 8984.602, 8984.84, 8985.097, 8985.373, 8985.669, 8985.983, 8986.316, 8986.669, 8987.04, 8987.431, 8987.84, 8988.269, 8988.717, 8989.183, 8989.669, 8990.174, 8990.698, 8991.241, 8991.803, 8992.384, 8992.984, 8993.603, 8994.241, 8994.898, 8995.575, 8996.27, 8996.984, 8997.718, 8998.47, 8999.242, 9000.032, 9000.842, 9001.671, 9002.518, 9003.385, 9004.271, 9005.176, 9006.1, 9007.043, 9008.005, 9008.986, 9009.986, 9011.005, 9012.043, 9013.1, 9014.177, 9015.272, 9016.387, 9017.52, 9018.673, 9019.844, 9021.035, 9022.244, 9023.473, 9024.721, 9025.988, 9027.274, 9028.579, 9029.903, 9031.246, 9032.608, 9033.989, 9035.389, 9036.808, 9038.246, 9039.704, 9041.18, 9042.675, 9044.19, 9045.723, 9047.276, 9048.848, 9050.438, 9052.048, 9053.677, 9055.325, 9056.991, 9058.677, 9060.382, 9062.106];
