%load_ext autoreload
%autoreload 2

from random import sample
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

import utils
import torch

# Define the transformations as functions

def torh_train_transform(size=(224, 224), scale=(0.08, 1.0), rotation=30, flip=0.5):
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

def torh_val_transform(size=(224, 224)):
    image_transforms = v2.Compose([
        v2.Resize(size),
        v2.PILToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return image_transforms

if __name__ == '__main__':
    config = {
        "name": "gdxray",
        "data_dir": "../data/gdxray",
        "metadata": "../metadata",
        "subset": "train",
        "image_size": (224, 224),
    }

    transform = torh_train_transform(size=config["image_size"])

    dataset = utils.GDXrayDataset(config, labels=True, transform=transform)

    img, lbl = dataset[0]

    # utils.visualize_samples(dataset, 5)
    utils.visualize_augmentations(dataset, 2, samples=2, cols=1)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # for idx, img_mask in enumerate(dataloader):
    #     print(img_mask[0].shape, img_mask[1].shape)
    #     break
