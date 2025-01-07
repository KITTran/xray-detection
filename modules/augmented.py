import albumentations as A
from torchvision.transforms import v2

# Define the transformations as functions

def albumentations_transform_train(IMG_HEIGHT, IMG_WIDTH):
    al_transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])

    return al_transform

def torch_transform_train(IMG_HEIGHT, IMG_WIDTH):
    tv_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(90),
        v2.RandomAffine(0, translate=(0.1, 0.1)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return tv_transform

def albumentations_transform_valid(IMG_HEIGHT, IMG_WIDTH):
    al_transform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.pytorch.transforms.ToTensorV2()
    ])

    return al_transform

def torch_transform_valid(IMG_HEIGHT, IMG_WIDTH):
    tv_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((IMG_HEIGHT, IMG_WIDTH)),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return tv_transform