# Get time of execution
import datetime
import os
import sys
from tqdm import tqdm

import torch
import torchmetrics
from torch import optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))
from modules import utils, models, losses, augmented, metrics

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
        'image_size': (320, 640),
        'learning_rate': 1e-4,
        'batch_size': 4,
        'epochs': 1500,
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray")
   }

transform_train = augmented.reshdc_augmentation_train(config['image_size'])
transform_valid = augmented.reshdc_augmentation_valid(config['image_size'])

train_dataset = utils.GDXrayDataset(config, labels=config['labels'], transform=transform_train)
valid_dataset = utils.GDXrayDataset(config, labels=config['labels'], transform=transform_valid)

if config['device'] == "cuda":
    num_workers = 8

train_dataloader = DataLoader(dataset=train_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=config['batch_size'],
                              shuffle=True)
valid_dataloader = DataLoader(dataset=valid_dataset,
                              num_workers=num_workers, pin_memory=False,
                              batch_size=config['batch_size'],
                              shuffle=True)

output_list = [32, 64, 128, 256, 512]
num_classes = 1
model = models.ResHDC_Model(num_classes, 3, output_list)
model.to(config['device'])

optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])
# criterion = losses.ImprovedCE((0.3, 0.7))
criterion = losses.BinaryDiceLoss()

metrics = metrics.ResMetrics(threshold=0.7)

torch.cuda.empty_cache()

# Train model
train_losses = []
train_dices = []
valid_losses = []
valid_dices = []

for epoch in tqdm(range(config['epochs'])):
    model.train()
    train_running_loss = 0.0
    train_running_metrics = {}

    for idx, img_mask in enumerate(tqdm(train_dataloader, position=0, leave=True)):
        img = img_mask[0].to(config['device'], dtype=torch.float32)
        mask = img_mask[1].to(config['device'], dtype=torch.long)

        y_pred = model(img)
        optimizer.zero_grad()

        loss = criterion(y_pred.squeeze(), mask.float().squeeze())

        # Metrics inlcude dice, precision, recall, jaccard, sensitivity
        metric = metrics(y_pred.squeeze(), mask.squeeze())

        try:
            for key in metric.keys():
                train_running_metrics[key] += metric[key]
        except KeyError:
            train_running_metrics = metric

        train_running_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss = train_running_loss / (idx + 1)
    train_metric = {key: value / (idx + 1) for key, value in train_running_metrics.items()}

    train_losses.append(train_loss)
    train_dices.append(train_metric['dice'])

    model.eval()
    val_running_loss = 0.0
    val_running_metrics = {}

    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(valid_dataloader, position=0, leave=True)):
            img = img_mask[0].to(config['device'], dtype=torch.float32)
            mask = img_mask[1].to(config['device'], dtype=torch.long)

            y_pred = model(img)
            loss = criterion(y_pred.squeeze(1), mask.float().squeeze())

            # Metrics inlcude dice, precision, recall, jaccard, sensitivity
            metric = metrics(y_pred.squeeze(), mask.squeeze())

            try:
                for key in metric.keys():
                    val_running_metrics[key] += metric[key]
            except KeyError:
                val_running_metrics = metric

            val_running_loss += loss.item()

        val_loss = val_running_loss / (idx + 1)
        val_metric = {key: value / (idx + 1) for key, value in val_running_metrics.items()}

    valid_losses.append(val_loss)
    valid_dices.append(val_metric['dice'])

    if epoch % 100 == 0:
        print("-" * 30)
        print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
        print("\n")
        print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
        utils.print_metrics(epoch + 1, **val_metric)
        print("-" * 30)

# Save model
if not os.path.exists(config['save_dir']):
    os.makedirs(config['save_dir'])

torch.save(model.state_dict(), os.path.join(config['save_dir'], f"{config['name']}_model_{now}.pth"))

# Save training and validation metrics
utils.save_metrics(config['save_dir'], now, train_losses, train_dices, valid_losses, valid_dices)
epochs_list = list(range(1, config['epochs'] + 1))


# Plot training and validation metrics
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
plt.plot(epochs_list, train_dices, label='Training DICE')
plt.plot(epochs_list, valid_dices, label='Validation DICE')
plt.xticks(ticks=list(range(1, config['epochs'] + 1, 1)))
plt.title('DICE Coefficient over epochs')
plt.xlabel('Epochs')
plt.ylabel('DICE')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()
