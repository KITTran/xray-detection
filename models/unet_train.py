# Get time of execution
%load_ext autoreload
%autoreload 2

import datetime
import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tqdm

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from modules import network, utils, losses, augmented, metrics

now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d_%H-%M-%S')
print(f"Time of execution: {now}")


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

config = {
        'name': "gdxray",
        'data_dir': os.path.join(PARENT_DIR, "data/gdxray"),
        'metadata': os.path.join(PARENT_DIR, "metadata/gdxray"),
        'subset': "train",
        'labels': True,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (224, 224),
        'learning_rate': 3e-4,
        'batch_size': 8,
        'epochs': 10,
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray")
   }

transform_train = augmented.unet_augmentation_train(config['image_size'])
transform_valid = augmented.unet_augmentation_valid(config['image_size'])

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

# Initialize model, loss function, and optimizer
output_list = [64, 128, 256, 512, 1024]  # Channel dimensions for each level
model = network.UNet(num_classes=1, input_channels=3, output_list=output_list)
model = model.to(config['device'])

# Loss function and optimizer
criterion = losses.BinaryDiceLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Metrics
metrics = metrics.UnetMetrics(threshold=0.7)

# Training loop
best_val_dice = 0.0
for epoch in range(config['epochs']):
    # Training phase
    model.train()
    train_metrics = {}
    train_loss = 0.0
    
    for batch_idx, (images, masks) in enumerate(tqdm.tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{config["epochs"]}')):
        images = images.to(config['device'])
        masks = masks.to(config['device'])
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.squeeze(), masks.squeeze())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_metrics = metrics.compute_metrics(outputs, masks)
        train_loss += loss.item()
        
        # Accumulate metrics
        if not train_metrics:
            train_metrics = {k: v for k, v in batch_metrics.items()}
        else:
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]
    
    # Average training metrics
    train_loss /= len(train_dataloader)
    for k in train_metrics:
        train_metrics[k] /= len(train_dataloader)
    
    # Validation phase
    model.eval()
    val_metrics = {}
    val_loss = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm.tqdm(valid_dataloader, desc='Validation'):
            images = images.to(config['device'])
            masks = masks.to(config['device'])
            
            outputs = model(images)
            loss = criterion(outputs.squeeze(), masks.squeeze())
            
            # Calculate metrics
            batch_metrics = metrics.compute_metrics(outputs, masks)
            val_loss += loss.item()
            
            # Accumulate metrics
            if not val_metrics:
                val_metrics = {k: v for k, v in batch_metrics.items()}
            else:
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]
    
    # Average validation metrics
    val_loss /= len(valid_dataloader)
    for k in val_metrics:
        val_metrics[k] /= len(valid_dataloader)
    
    # Print epoch results
    print(f'\nEpoch {epoch+1}/{config["epochs"]}:')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    print('Training Metrics:')
    for k, v in train_metrics.items():
        print(f'{k}: {v:.4f}')
    print('Validation Metrics:')
    for k, v in val_metrics.items():
        print(f'{k}: {v:.4f}')
    
    # Save best model
    if val_metrics['dice'] > best_val_dice:
        best_val_dice = val_metrics['dice']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': best_val_dice,
        }, os.path.join(config['save_dir'], 'best_model.pth'))
    
    # Save checkpoint
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_dice': val_metrics['dice'],
    }, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))

print('Training completed!')
