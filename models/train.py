#%%

import os
import sys
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from torchvision import transforms

import segmentation_models_pytorch as smp

sys.path.append(os.path.abspath('..'))
from modules import utils, models, losses, test
config = {
    'IMG_SIZE': (256, 256),
    'DATA_DIR': 'data/gdxray/welding/W0001',
    'LABEL_DIR': 'data/gdxray/welding/W0002',
    'BATCH_SIZE': 1,
    'EPOCHS': 10,
    'LR': 0.001,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print('Device: ', config['DEVICE'])

datadir = "/home/kittran/projects/cracked-detection/dataset/gdxray/welding"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 128x128
    transforms.ToTensor(),         # Convert images to PyTorch tensors
])

# Create dataset instance
dataset = test.ImageLabelDataset(datadir, transform=transform)

# Get a sample
img, lbl = dataset[0]
print(f"Image shape: {img.shape}, Label shape: {lbl.shape}")

# Create a DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# initialize the model
output_list = [64, 128, 256, 512, 1024]
num_parallel = 2
upsampling_cfg = dict(type='carafe', scale_factor=2, kernel_up=5, kernel_encoder=3)

model = models.WResHDC_FF(output_list, num_parallel, upsampling_cfg)
model.to(config['DEVICE'])

# Define the loss function
criterion = smp.losses.DiceLoss(mode='binary')

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train one epoch function and print the loss
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)

# Train the model for 10 epochs and reduce memory consumption each epoch 
for epoch in range(10):
    loss = train_one_epoch(model, dataloader, criterion, optimizer, config['DEVICE'])
    print(f"Epoch {epoch+1}/{config['EPOCHS']}, Loss: {loss}")

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        torch.cuda.empty_cache()