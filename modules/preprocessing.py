import os
import cv2
import math
import keras_ocr
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.filters import threshold_otsu

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

# Load images from sample dataset
folder_list = [os.path.join(project_dir, 'dataset/sample', folder) for folder in os.listdir(project_dir + '/dataset/sample')]
images = load_images_from_folders(folder_list)

# Display 5 last images
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
    axs[i].imshow(images[len(images) - i - 1], cmap='gray')
    axs[i].axis('off')
plt.show()

# Add one more dimension to images
images = [np.expand_dims(img, axis=-1) for img in images]
rgb_batch = [np.repeat(img, 3, -1) for img in images]

# Remove text from images
def mid_point(x1, y1, x2, y2):
    x_mid = int((x1 + x2) / 2)
    y_mid = int((y1 + y2) / 2)
    return x_mid, y_mid

def inpaint_text(img, pipeline):
    # Get text bounding boxes
    prediction_groups = pipeline.recognize([img])

    # Inpaint text
    mask = np.zeros(img.shape[:2], np.uint8)
    for bbox in prediction_groups[0]:
        x0, y0 = bbox[1][0]
        x1, y1 = bbox[1][1]
        x2, y2 = bbox[1][2]
        x3, y3 = bbox[1][3]

        x_mid0, y_mid0 = mid_point(x1, y1, x2, y2)
        x_mid1, y_mid1 = mid_point(x0, y0, x3, y3)

        thickness = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mid1), 255, thickness)
        inpainted_img = cv2.inpaint(img, mask, inpaintRadius=thickness, flags=cv2.INPAINT_NS)

    return inpainted_img

pipeline = keras_ocr.pipeline.Pipeline()

# Remove text from images
img_text_removed = [inpaint_text(img, pipeline) for img in rgb_batch[-5:]]

# Display 5 last images with text removed
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
    axs[i].imshow(img_text_removed[len(img_text_removed) - i - 1], cmap='gray')
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

processed_images = preprocess_images(img_text_removed)

# Display 5 last processed images
fig, axs = plt.subplots(1, 5, figsize=(20, 20))
for i in range(5):
    axs[i].imshow(processed_images[len(processed_images) - i - 1], cmap='gray')
    axs[i].axis('off')
plt.show()
