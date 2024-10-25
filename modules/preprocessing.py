import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray

current_dir = os.path.dirname(os.path.realpath(__file__))
project_dir = os.path.dirname(current_dir)

# Import dataset
def load_images_from_folders(folder_list):
    images = []
    for folder in folder_list:
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path)
                images.append(np.array(img))
    return images

# Example usage
folder_list = [os.path.join(project_dir, 'dataset/sample', folder) for folder in os.listdir(project_dir + '/dataset/sample')]
images = load_images_from_folders(folder_list)

# Display 5 last images
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
    axs[i].imshow(images[len(images) - i - 1], cmap='gray')
    axs[i].axis('off')
plt.show()

# Preprocess images to hightlight defects in images
# Thresholding Otsu -> Labeling -> area calcuation -> defect segmentation

def preprocess_images(images):
    processed_images = []
    for img in images:
        thresh = threshold_otsu(img)
        binary_img = img < thresh
        processed_images.append(binary_img)
    return processed_images

processed_images = preprocess_images(images)

# Display 5 last processed images
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
    axs[i].imshow(processed_images[len(processed_images) - i - 1], cmap='gray')
    axs[i].axis('off')
plt.show()
