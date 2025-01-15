%load_ext autoreload
%autoreload 2

from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

import utils
import torch

import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

# Define the transformations as functions

def torch_train_transform(size=(224, 224), scale=(0.08, 1.0), rotation=30, flip=0.5):
    image_transforms = v2.Compose([
        # v2.ToImage(),
        v2.RandomCrop(size),  # Random crop of size 256x256
        # v2.Resize(size),
        v2.RandomRotation(degrees=rotation, interpolation=InterpolationMode.BILINEAR),  # Random rotation within 30 degrees
        # v2.RandomResize(size=size, scale=scale),  # Random rescale and crop
        v2.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Random affine transformation
        v2.RandomHorizontalFlip(p=flip),  # Random horizontal flip
        v2.GaussianBlur(kernel_size=3),  # Apply Gaussian blur
        # v2.RandomErasing(p=0.5),  # Random cutout
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        v2.ToDtype(torch.float32, scale=True),
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
        # Apply all augmentations in order
        geo_transforms = [
            lambda img, msk: self.random_crop(img, msk),
            lambda img, msk: v2.RandomRotation(self.random_rotation)(img, msk),
            lambda img, msk: self.random_shift(img, msk),
            lambda img, msk: self.random_rotate(img, msk),
            lambda img, msk: v2.RandomHorizontalFlip()(img, msk),
            lambda img, msk: self.random_cutout(img, msk),
        ]

        color_transforms = [
            lambda img: v2.ColorJitter()(img),
            lambda img: self.random_gaussian(img),
        ]

        for geo_transform in geo_transforms:
            image, mask = geo_transform(image, mask)

        # Apply color transformations
        for color_transform in color_transforms:
            image = color_transform(image)

        return image, mask

if __name__ == '__main__':
    config = {
        "name": "gdxray",
        "data_dir": "../data/gdxray",
        "metadata": "../metadata",
        "subset": "train",
        "image_size": (224, 224),
    }

    transform = torch_train_transform(size=config["image_size"])

    dataset = utils.GDXrayDataset(config, labels=True, transform=transform)

    img, lbl = dataset[0]

    # utils.visualize_samples(dataset, 5)
    utils.visualize_augmentations(dataset, 2, samples=2, cols=1)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # for idx, img_mask in enumerate(dataloader):
    #     print(img_mask[0].shape, img_mask[1].shape)
    #     break
