import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from ..modules import utils, models, losses

config = {
    'IMG_SIZE': (256, 256),
    'DATA_DIR': 'data/gdxray/welding/W0001',
    'LABEL_DIR': 'data/gdxray/welding/W0002',
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LR': 0.001,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize()
])

dataset = utils.GDXrayDataset(config, "train", transform=transform, )
