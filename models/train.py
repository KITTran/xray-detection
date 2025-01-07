#%%

import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('..'))
from modules import utils, models, losses, augmented

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

config = {
        'name': "gdxray",
        'data_dir': os.path.join(PARENT_DIR, "data/gdxray"),
        'metadata': os.path.join(PARENT_DIR, "metadata"),
        'subset': "train",
        'labels': True,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (224, 224),
        'learning_rate': 3e-4,
        'batch_size': 8,
   }

transform_train = augmented.albumentations_transform_train(config['image_size'])
transform_valid = augmented.albumentations_transform_valid(config['image_size'])

train_dataset = utils.GDXrayDataset(config, labels=config['labels'], transform=transform_train)
valid_dataset = utils.GDXrayDataset(config, labels=config['labels'], transform=transform_valid)

if config['device'] == "cuda":
    num_workers = torch.cuda.device_count() * 4

train_dataloader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=config['batch_size'],
                              shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=config['batch_size'],
                              shuffle=True)

output_list = [64, 128, 256, 512]
num_parallel = 2
num_classes = 2
upsampling_cfg = dict(type='carafe', scale_factor=2, kernel_up=5, kernel_encoder=3)
model = models.WResHDC_FF(num_classes, 3, output_list, num_parallel, upsample_cfg=upsampling_cfg)
model.to(config['device'])

optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
criterion = losses.

torch.cuda.empty_cache()