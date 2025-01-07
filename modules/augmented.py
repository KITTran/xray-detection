import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

import utils
import torch

# Define the transformations as functions
# TODO: FIX ToTensorV2 disapear label channel

def albumentations_transform_train(IMG_SIZE):
    al_transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2(),
    ])

    return al_transform

def torch_transform_train(IMG_SIZE):
    tv_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(IMG_SIZE),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(90),
        v2.RandomAffine(0, translate=(0.1, 0.1)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return tv_transform

def albumentations_transform_valid(IMG_SIZE):
    al_transform = A.Compose([
        A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ToTensorV2(),
    ])

    return al_transform

def torch_transform_valid(IMG_SIZE):
    tv_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(IMG_SIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return tv_transform

if __name__ == '__main__':
    config = {
        "name": "gdxray",
        "data_dir": "../data/gdxray",
        "metadata": "../metadata",
        "subset": "train",
        "image_size": (224, 224),
    }

    transform = albumentations_transform_train(config["image_size"])

    dataset = utils.GDXrayDataset(config, labels=True, transform=transform)

    utils.visualize_samples(dataset, num_samples=3, labels=True)