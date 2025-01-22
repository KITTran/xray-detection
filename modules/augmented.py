%load_ext autoreload
%autoreload 2

from shapely import transform
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

import utils
import torch

import random
import numpy as np
from PIL import Image, ImageOps
import cv2

# Define the transformations as functions

def torch_train_transform(size=(224, 224), scale=(0.08, 1.0), rotation=30, flip=0.5):
    image_transforms = v2.Compose([
        # v2.ToImage(),
        # v2.RandomCrop(size),  # Random crop of size 256x256
        # v2.Resize(size),
        # v2.RandomRotation(degrees=rotation, interpolation=InterpolationMode.BILINEAR),  # Random rotation within 30 degrees
        # v2.RandomResize(size=size, scale=scale),  # Random rescale and crop
        # v2.RandomAffine(degrees=0, translate=(0.2, 0)),  # Random affine transformation
        # v2.RandomHorizontalFlip(p=flip),  # Random horizontal flip
        # v2.GaussianBlur(kernel_size=9),  # Apply Gaussian blur
        # v2.RandomErasing(p=0.5),  # Random cutout
        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        # v2.ToDtype(torch.float32, scale=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return image_transforms

def torch_val_transform(size=(224, 224)):
    image_transforms = v2.Compose([
        v2.Resize(size),
        v2.PILToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return image_transforms

class CustomAugmentation:
    def __init__(self):
        self.random_rotation = (1, 3)  # RandomRotation
        self.num_holes = 10  # RandomCutout
        self.max_h_size = 40
        self.max_w_size = 40
        self.fill_value = 0
        self.gaussian_mean = 0.2  # RandomGaussian
        self.gaussian_sigma = 0.3

        self.transforms = [
            lambda img, msk: self.random_crop(img, msk),
            lambda img, msk: v2.RandomRotation(self.random_rotation)(img, msk),
            lambda img, msk: self.random_shift(img, msk),
            lambda img, msk: self.random_rotate(img, msk),
            lambda img, msk: v2.RandomHorizontalFlip()(img, msk),
            lambda img, msk: self.random_cutout(img, msk),
        ]

        self.color_transforms = [
            lambda img: v2.ColorJitter()(img),
            lambda img: self.random_gaussian(img),
        ]

    def random_crop(self, image, mask):
        cropping_number = 2  # RandomCrop
        width, height = image.size
        for _ in range(cropping_number):
            left = random.randint(0, width // 4)
            top = random.randint(0, height // 4)
            right = random.randint(3 * width // 4, width)
            bottom = random.randint(3 * height // 4, height)
            image = image.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))
        return image, mask

    def random_cutout(self, image, mask):
        img_np = np.array(image)
        mask_np = np.array(mask)
        for _ in range(self.num_holes):
            y = random.randint(0, img_np.shape[0] - self.max_h_size)
            x = random.randint(0, img_np.shape[1] - self.max_w_size)
            img_np[y:y + self.max_h_size, x:x + self.max_w_size, :] = self.fill_value
            mask_np[y:y + self.max_h_size, x:x + self.max_w_size] = self.fill_value
        return Image.fromarray(img_np), Image.fromarray(mask_np)

    def random_gaussian(self, image):
        img_np = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(self.gaussian_mean, self.gaussian_sigma, img_np.shape)
        img_np = np.clip(img_np + noise, 0, 1) * 255.0
        return Image.fromarray(img_np.astype(np.uint8))

    def random_rotate(self, image, mask):
        angle = random.uniform(-180, 180)
        image = ImageOps.expand(image.rotate(angle, expand=True), border=1, fill=0)
        mask = ImageOps.expand(mask.rotate(angle, expand=True), border=1, fill=0)
        return image, mask

    def random_shift(self, image, mask):
        width, height = image.size
        shift_x = random.randint(-width // 10, width // 10)
        shift_y = random.randint(-height // 10, height // 10)
        image = ImageOps.offset(image, shift_x, shift_y)
        mask = ImageOps.offset(mask, shift_x, shift_y)
        return image, mask

    def __call__(self, image, mask):

        for geo_transform in self.transforms:
            image, mask = geo_transform(image, mask)

        # Apply color transformations
        for color_transform in self.color_transforms:
            image = color_transform(image)

        return image, mask

class UNetAugmentation:
    def __init__(self, patch_size=(224, 224), training=True):

        self.patch_size = patch_size
        self.training = training
        self.transforms = [
            lambda img: self.gamma_transformation(img),
            lambda img: self.linear_transformation(img),
            lambda img: self.add_gaussian_noise(img),
            lambda img: self.histogram_equalization(img),
        ]

        self.img_mask_transforms = [
            lambda img, mask: self.random_crop(img, mask, self.patch_size) if self.training else self.uniform_crop(img, mask, self.patch_size),
        ]

    def __call__(self, image, mask):

        for img_transform in self.transforms:
            image = img_transform(image)

        for img_mask_transform in self.img_mask_transforms:
            image, mask = img_mask_transform(image, mask)

        return image, mask

    def gamma_transformation(self, image, c=1, gamma=0.7):
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        transformed = c * (image ** gamma)
        transformed = np.clip(transformed * 255.0, 0, 255).astype(np.uint8)  # Scale back to [0, 255]
        return Image.fromarray(transformed)

    def linear_transformation(self, image, k=0.85, b=0.3):
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        transformed = k * image + b
        transformed = np.clip(transformed * 255.0, 0, 255).astype(np.uint8)  # Scale back to [0, 255]
        return Image.fromarray(transformed)

    def add_gaussian_noise(self, image, mean=0, std=0.02):
        image = np.array(image) / 255.0  # Normalize to [0, 1]
        noise = np.random.normal(mean, std, image.shape)
        noisy_image = image + noise
        noisy_image = np.clip(noisy_image * 255.0, 0, 255).astype(np.uint8)  # Scale back to [0, 255]
        return Image.fromarray(noisy_image)

    def histogram_equalization(self, image):
        image = np.array(image)
        if len(image.shape) == 3 and image.shape[2] == 3:  # Color image
            img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:  # Grayscale image
            equalized = cv2.equalizeHist(image)
        return Image.fromarray(equalized)

    def random_crop(self, image, mask, patch_size):
        width, height = image.size
        crop_width, crop_height = patch_size
        left = random.randint(0, width - crop_width)
        upper = random.randint(0, height - crop_height)
        right = left + crop_width
        lower = upper + crop_height
        return image.crop((left, upper, right, lower)), mask.crop((left, upper, right, lower))

    def uniform_crop(self, image, mask, patch_size):
        width, height = image.size
        crop_width, crop_height = patch_size
        left = (width - crop_width) // 2
        upper = (height - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height
        return image.crop((left, upper, right, lower)), mask.crop((left, upper, right, lower))

if __name__ == '__main__':
    config = {
        "name": "gdxray",
        "data_dir": "../data/gdxray",
        "metadata": "../metadata",
        "subset": "train",
        "image_size": (224, 224),
    }

    # transform = torch_train_transform(size=config["image_size"])
    # transform = UNetAugmentation(patch_size=config["image_size"], training=True)
    transform = CustomAugmentation()

    dataset = utils.GDXrayDataset(config, labels=True, transform=transform)

    img, lbl = dataset[0]

    # utils.visualize_samples(dataset, 5)
    utils.visualize_augmentations(dataset, 6, samples=4, cols=2)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # for idx, img_mask in enumerate(dataloader):
    #     print(img_mask[0].shape, img_mask[1].shape)
    #     break
