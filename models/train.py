#%%

import os
import sys

import torch
import torch.optim as optim

import albumentations as A
import segmentation_models_pytorch as smp

sys.path.append(os.path.abspath('..'))
from modules import utils, models, losses

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

config = {
        'name': "gdxray",
        'data_dir': os.path.join(PARENT_DIR, "data/gdxray"),
        'metadata': os.path.join(PARENT_DIR, "metadata"),
        'subset': "train",
        'labels': True,
   }

config = utils.Config

dataset = utils.GDXrayDataset(config, labels=config['labels'], transform=None)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    num_workers = torch.cuda.device_count() * 4
