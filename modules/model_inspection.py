%load_ext autoreload
%autoreload 2
import json
from cv2 import transform
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import plotly.express as px

import os
import torch
import random
from tqdm import tqdm

import models
import utils
import metrics
import augmented

from torchvision.transforms import v2 as transforms

# Load metrics from JSON file
with open('/home/tuank/projects/cracked-detection/logs/gdxray/metrics_2025-01-30_17-48-13.json', 'r') as f:
    metrics_dict = json.load(f)

# Extract loss and dice score arrays
train_loss = metrics_dict['train_losses']
train_dice = metrics_dict['train_dcs']
valid_loss = metrics_dict['valid_losses']
valid_dice = metrics_dict['valid_dcs']

epochs_list = list(range(1, len(train_loss) + 1))

# Plot training and validation metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_loss, label='Training Loss')
plt.plot(epochs_list, valid_loss, label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid()
plt.tight_layout()

plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_list, train_dice, label='Training DICE')
plt.plot(epochs_list, valid_dice, label='Validation DICE')
plt.title('DICE Coefficient over epochs')
plt.xlabel('Epochs')
plt.ylabel('DICE')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Replace this with your actual metric data
metrics_train = train_loss
metrics_valid = valid_loss
epochs = np.arange(1, len(metrics_train) + 1)

# Calculate the moving average
window_size = 100  # Adjust the window size as needed
moving_average_train = np.convolve(metrics_train, np.ones(window_size)/window_size, mode='valid')
moving_average_valid = np.convolve(metrics_valid, np.ones(window_size)/window_size, mode='valid')

# Adjust epochs for the moving average plot
avg_epochs = epochs[window_size - 1:]

# Plotting the original metrics and moving average
plt.figure(figsize=(12, 6))
plt.plot(epochs, metrics_train, label='Original Training Loss', color='blue', alpha=0.5)
plt.plot(avg_epochs, moving_average_train, label=f'{window_size}-Epoch Moving Average (Train)', color='red', linewidth=2)
plt.plot(epochs, metrics_valid, label='Original Validation Loss', color='green', alpha=0.5)
plt.plot(avg_epochs, moving_average_valid, label=f'{window_size}-Epoch Moving Average (Valid)', color='orange', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs with Moving Average')
plt.legend()
plt.show()

# Perform LOWESS smoothing for train_dice and valid_dice
lowess_train_dice = sm.nonparametric.lowess(train_dice, epochs_list, frac=0.1)
lowess_valid_dice = sm.nonparametric.lowess(valid_dice, epochs_list, frac=0.1)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(epochs_list, train_dice, label='Original Training DICE', color='blue', alpha=0.5)
plt.plot(lowess_train_dice[:, 0], lowess_train_dice[:, 1], label='LOWESS Smoothed Training DICE', color='red', linewidth=2)
plt.plot(epochs_list, valid_dice, label='Original Validation DICE', color='green', alpha=0.5)
plt.plot(lowess_valid_dice[:, 0], lowess_valid_dice[:, 1], label='LOWESS Smoothed Validation DICE', color='orange', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('DICE')
plt.title('Training and Validation DICE over Epochs with LOWESS Smoothing')
plt.legend()
plt.show()

# Create a DataFrame for Plotly
df = pd.DataFrame({
    'Epochs': epochs,
    'Training Loss': metrics_train,
    'Validation Loss': metrics_valid
})

# # Melt the DataFrame to have a long format suitable for Plotly
# df_melted = df.melt(id_vars=['Epochs'], value_vars=['Training Loss', 'Validation Loss'],
#                     var_name='Metric', value_name='Loss')

# # Create the line plot
# fig = px.line(df_melted, x='Epochs', y='Loss', color='Metric', title='Training and Validation Loss over Epochs')
# fig.show()

# Load dataset
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
        'batch_size': 8,
        'epochs': 40000,100.98.42.94
        'save_dir': os.path.join(PARENT_DIR, "logs/gdxray")
   }

transform_test = augmented.unet_augmentation_valid(config['image_size'])

test_dataset = utils.GDXrayDataset(config, labels=config['labels'], transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,num_workers=4, pin_memory=False,
batch_size=config['batch_size'],
shuffle=True)

output_list = [32, 64, 128, 256, 512]
num_classes = 1
model = models.UNet(num_classes, 3, output_list).to(config['device'])

# Load model weights
weights_path = '/home/tuank/projects/cracked-detection/logs/gdxray/gdxray_model_2025-01-30_17-48-13.pth'
model.load_state_dict(torch.load(weights_path, map_location=torch.device(config['device'])))

test_running_metric = {}
metric_load = metrics.UnetMetrics(threshold=0.7)

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    for inputs, targets in tqdm(test_dataloader):
        inputs, targets = inputs.to(config['device']), targets.to(config['device'])

        outputs = model(inputs)

        # calculate metrics
        metric = metric_load.compute_metrics(outputs, targets)
        try:
            for key in metric.keys():
                test_running_metric[key] += metric[key]
        except KeyError:
            test_running_metric = metric

    test_metric = {key: value / len(test_dataloader) for key, value in test_running_metric.items()}

utils.print_metrics(0, **test_metric)

def random_images_inference(image_tensors, mask_tensors, image_paths, model_pth, device):
    model = models.UNet(num_classes, 3, output_list).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    # Iterate for the images, masks and paths
    for image_pth, mask_pth, image_paths in zip(image_tensors, mask_tensors, image_paths):
        # Load the image
        img = image_pth

        # Predict the imagen with the model
        pred_mask = model(img.unsqueeze(0))
        pred_mask = pred_mask.squeeze(0).permute(1,2,0)

        # Load the mask to compare
        mask = mask_pth.permute(1, 2, 0).to(device)

        print(f"Image: {os.path.basename(image_paths)}, DICE coefficient: {round(float(metrics.dice_coefficient(pred_mask, mask)),5)}")

        # Show the images
        img = img.cpu().detach().permute(1, 2, 0)
        pred_mask = pred_mask.cpu().detach()
        pred_mask[pred_mask < 0] = 0
        pred_mask[pred_mask > 0] = 1

        plt.figure(figsize=(15, 16))
        plt.subplot(131), plt.imshow(img), plt.title("original")
        plt.axis('off')
        plt.subplot(132), plt.imshow(pred_mask, cmap="gray"), plt.title("predicted")
        plt.axis('off')
        plt.subplot(133), plt.imshow(mask, cmap="gray"), plt.title("mask")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Load the images
n = 5

image_tensors = []
mask_tensors = []
image_paths = []

for _ in range(n):
    random_index = random.randint(0, len(test_dataloader.dataset) - 1)
    random_sample = test_dataloader.dataset[random_index]

    image_tensors.append(random_sample[0])
    mask_tensors.append(random_sample[1])
    image_paths.append(test_dataset.image_info[random_index]['path'])

# Perform inference
random_images_inference(image_tensors, mask_tensors, image_paths, weights_path, config['device'])

# Image path
image_path = '/home/tuank/projects/cracked-detection/data/sample/IMG（1~30）/1 (A) のコピー.jpg'

# Load the image
from PIL import Image

image = Image.open(image_path).convert("RGB")

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(config['image_size']),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = preprocess(image)

# Predict the image
model.eval()
with torch.no_grad():
    output = model(image.unsqueeze(0).to(config['device']))
    output = torch.sigmoid(output).squeeze(0).cpu().detach().numpy()

output[output < 0.5] = 0
output[output >= 0.5] = 1

# Plot the image and the predicted mask
plt.figure(figsize=(15, 8))
plt.subplot(121), plt.imshow(image.permute(1, 2, 0)), plt.title("Original Image")
plt.subplot(122), plt.imshow(output.squeeze(), cmap="gray"), plt.title("Predicted Mask")
plt.axis("off")
plt.tight_layout()
plt.show()
