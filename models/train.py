# Get time of execution
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
        'metadata': os.path.join(PARENT_DIR, "metadata"),
        'subset': "train",
        'labels': True,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        'image_size': (224, 224),
        'learning_rate': 3e-4,
        'batch_size': 8,
        'epochs': 10,
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray")
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
model = network.WResHDC_FF(num_classes, 3, output_list, num_parallel, upsample_cfg=upsampling_cfg)
model.to(config['device'])

optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
criterion = nn.BCEWithLogitsLoss()

torch.cuda.empty_cache()

# Train model
train_losses = []
train_dcs = []
valid_losses = []
valid_dcs = []

for epoch in tqdm(range(config['epochs'])):
    model.train()
    train_running_loss = 0.0
    train_running_dc = 0.0

    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].to(config['device'], dtype=torch.float32)
        mask = img_mask[1].to(config['device'], dtype=torch.long)

        y_pred = model(img)
        optimizer.zero_grad()

        dc = metrics.dice_coefficient(y_pred.squeeze(1), mask.float())
        loss = criterion(y_pred.squeeze(1), mask.float())
        loss += losses.dice_loss(F.sigmoid(y_pred.squeeze(1)), mask.float(), multiclass=False)


        train_running_loss += loss.item()
        train_running_dc += dc.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (idx + 1)
    train_dc = train_running_dc / (idx + 1)

    train_losses.append(train_loss)
    train_dcs.append(train_dc)

    model.eval()
    val_running_loss = 0.0
    val_running_dc = 0.0

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(valid_dataloader, position=0, leave=True)):
            img = img_mask[0].to(config['device'], dtype=torch.float32)
            mask = img_mask[1].to(config['device'], dtype=torch.long)

            y_pred = model(img)
            loss = criterion(y_pred.squeeze(1), mask)
            loss += losses.dice_loss(F.sigmoid(y_pred.squeeze(1)), mask.float(), multiclass=False)
            dc = metrics.dice_coefficient(y_pred.squeeze(1), mask.float())

            val_running_loss += loss.item()
            val_running_dc += dc.item()

        val_loss = val_running_loss / (idx + 1)
        val_dc = val_running_dc / (idx + 1)

    valid_losses.append(val_loss)
    valid_dcs.append(val_dc)

    print("-" * 30)
    print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
    print(f"Training DICE EPOCH {epoch + 1}: {train_dc:.4f}")
    print("\n")
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print(f"Validation DICE EPOCH {epoch + 1}: {val_dc:.4f}")
    print("-" * 30)

# Save model
if not os.path.exists(config['save_dir']):
    os.makedirs(config['save_dir'])

torch.save(model.state_dict(), os.path.join(config['save_dir'], f"{config['name']}_model_{now}.pth"))

# Save training and validation metrics
utils.save_metrics(config['save_dir'], now, train_losses, train_dcs, valid_losses, valid_dcs)
epochs_list = list(range(1, config['epochs'] + 1))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_losses, label='Training Loss')
plt.plot(epochs_list, valid_losses, label='Validation Loss')
plt.xticks(ticks=list(range(1, config['epochs'] + 1, 1)))
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.tight_layout()

plt.legend()


plt.subplot(1, 2, 2)
plt.plot(epochs_list, train_dcs, label='Training DICE')
plt.plot(epochs_list, valid_dcs, label='Validation DICE')
plt.xticks(ticks=list(range(1, config['epochs'] + 1, 1)))
plt.title('DICE Coefficient over epochs')
plt.xlabel('Epochs')
plt.ylabel('DICE')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
