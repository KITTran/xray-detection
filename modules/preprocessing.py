import os
import cv2
import math
import keras_ocr
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

# Preprocess images to hightlight defects in images
# Thresholding Otsu -> Labeling -> area calcuation -> defect segmentation

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = rgb2gray(img)
        thresh = threshold_otsu(img)
        binary_img = img < thresh
        processed_images.append(binary_img)
    return processed_images

def visual_img(img_list, save_dir = None):
    """
    Visualize images in a list
    """
    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 20))
    for i, img in enumerate(img_list):
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.show()
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()

def hist_vis(img):
    """
    Visualize histogram of an image
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()

    plt.figure(figsize=(10, 5))
    plt.plot(cdf_normalized, color = 'black')
    plt.plot(hist)
    plt.legend(['cumulative', 'histogram'], loc = 'upper left')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Number of Pixels')
    plt.title('Grayscale Histogram')
    plt.show()

def hist_equalization(img, method = None):
    """
    Apply histogram equalization to an image
    """
    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)

    img = cv2.equalizeHist(img)
    return img

if __name__ == '__main__':
    # Load images from sample dataset
    folder_list = [os.path.join(project_dir, 'dataset/sample', folder) for folder in os.listdir(project_dir + '/dataset/sample')]
    images = load_images_from_folders(folder_list)

    # Display 5 last images
    visual_img(images[-5:])

    # # Add one more dimension to images
    # images = [np.expand_dims(img, axis=-1) for img in images]
    # rgb_batch = [np.repeat(img, 3, -1) for img in images]

    # pipeline = keras_ocr.pipeline.Pipeline()

    # # Remove text from images
    # img_text_removed = [inpaint_text(img, pipeline) for img in rgb_batch[-5:]]
    # visual_img(img_text_removed)

    # processed_images = preprocess_images(img_text_removed)
    # visual_img(processed_images)

    # Euqalize histogram of images
    hist_img = hist_equalization(images[0], method='clahe')
    hist_vis(hist_img)
    visual_img([hist_img])
