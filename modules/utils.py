import copy
import os
import json
import math
import random
import torch

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.utils import draw_segmentation_masks

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

BACKGROUND_CLASS = 0
WELDING_DEFECT = 1
OBJECT_CLASSES = [BACKGROUND_CLASS, WELDING_DEFECT]


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.


class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 50

    # Number of classification classes (including background)
    NUM_CLASSES = 2  # Override in sub-classes

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (128, 128)  # (height, width) of the mini-mask

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 512
    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

        # Input image size
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Compute backbone size from input image size
        self.BACKBONE_SHAPES = np.array(
            [
                [
                    int(math.ceil(self.IMAGE_SHAPE[0] / stride)),
                    int(math.ceil(self.IMAGE_SHAPE[1] / stride)),
                ]
                for stride in self.BACKBONE_STRIDES
            ]
        )

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class GDXrayDataset(Dataset):
    """
    Dataset of Xray Images

    Images are referred to using their image_id (relative path to image).
    An example image_id is: "Weldings/W0001/W0001_0004.png"
    """

    def __init__(self, config: dict, labels: bool, transform):
        super().__init__()
        """
        Args:
            config (dict): Contain nessesary information for the dataset
            config = {
                'name': str, # Name of the dataset
                'data_dir': str, # Path to the data directory
                'subset': str, # 'train' or 'val'
                'metadata': str, # Path to the metadata file
                }
            labels (bool): Whether to load labels
            transform (callable, optional): Transform to be applied to the images.
        """

        self.config = config
        self.labels = labels
        self.transform = transform

        # Initialize the dataset infos
        self.image_info = []
        self.image_indices = {}
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]

        # Add classes
        self.add_class(config["name"], 1, "Defect")

        # Load the dataset
        metadata_img = "{}/{}_{}.txt".format(config['metadata'], config["name"], "images")
        metadata_label = "{}/{}_{}.txt".format(config['metadata'], config["name"], "labels") if labels else None

        # Load image ids from key 'image' in dictionary in metadata file
        image_ids = []
        image_ids.extend(self.load_metadata(metadata_img, "image"))
        if self.labels:
            label_ids = []
            label_ids.extend(self.load_metadata(metadata_label, "label"))

        for i, image_id in enumerate(image_ids):
            img_path = os.path.join(config["data_dir"], image_id)
            label_path = ""
            if self.labels:
                label_path = os.path.join(config["data_dir"], label_ids[i])

                if not os.path.exists(label_path):
                    print("Skipping ", image_id, " Reason: No mask")

                    del self.image_ids[i]
                    del self.label_ids[i]

                    continue

            print("Adding image: ", image_id)

            self.add_image(config["name"], config['subset'], image_id, img_path, label=label_path)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: (image, label) where both are transformed.
        """

        image_path = self.image_info[idx]["path"]
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # self.update_info(idx, height_org=image.shape[0], width_org=image.shape[1])
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        self.update_info(idx, height_org=width, width_org=height)

        if self.labels:
            label_path = self.image_info[idx]["label"]
            # label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            label = Image.open(label_path).convert('L') # read as grayscale

        if self.transform:
            image, label = self.transform(image, label) if self.labels else self.transform(image)

        if self.labels:
            # Convert to binary mask
            label = np.array(label) / 255.0
            label = np.where(label > 0.3, 1, 0)
            return image, label

        return image

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info["source"] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append(
            {
                "source": source,
                "id": class_id,
                "name": class_name,
            }
        )

    def add_image(self, source, subset, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "subset": subset,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
        self.image_indices[image_id] = len(self.image_info) - 1

    def update_info(self, image_id, **kwargs):
        info = self.image_info[image_id]
        info.update(kwargs)
        self.image_info[image_id] = info

    def load_metadata(self, metadata, key):
        """
        metadata file has the following format:
        {
            "image": [<image_id>, <image_id>, ...],
            "label": [<label_id>, <label_id>, ...]
        }

        Args:
            metadata (str): Path to the metadata file
            key (str): Key to load from the metadata file
        """

        image_ids = []
        with open(metadata, "r") as metadata_file:
            image_ids += metadata_file.readlines()
        return [p.rstrip() for p in image_ids]


def visualize_samples(dataset, num_samples=3, labels=True):
    """
    Visualize random samples from the dataset with images in the first column and labels in the second.

    Args:
        dataset (Dataset): The PyTorch Dataset to visualize.
        num_samples (int): Number of random samples to visualize.
    """

    # Print a message notifying that the transformation is applied
    print("CAUTION!!!! Be careful with normalization...") if dataset.transform is not None else None

    # Randomly select indices
    indices = random.sample(range(len(dataset)), num_samples)

    # Set up the Matplotlib figure
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]  # Ensure axes is iterable for a single sample

    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        image_path = dataset.image_info[idx]["path"]
        label_path = dataset.image_info[idx]["label"]

        image, label = v2.ToTensor()(image), v2.ToTensor()(label)
        image_mask = draw_segmentation_masks(image, label.bool(), alpha=0.5)

        image = image.permute(1, 2, 0).numpy()
        image_mask = image_mask.permute(1, 2, 0).numpy()

        # Display the image
        axes[i][0].imshow(image)
        axes[i][0].set_title(
            f"Image: {image_path.split('/')[-1].split('.')[0]} -- Size: {image.shape}"
        )
        axes[i][0].axis("off")

        # Display the label
        axes[i][1].imshow(image_mask)
        axes[i][1].set_title(
            f"Label: {label_path.split('/')[-1].split('.')[0]} -- Size: {label.shape}"
        )
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()

def visualize_augmentations(dataset, idx=0, samples=10, cols=5):
    dataset = copy.deepcopy(dataset)
    # get transform from dataset and remove normalization
    transform = dataset.transform.transforms[:-1]
    dataset.transform = v2.Compose(transform)
    rows = samples // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols * 2, figsize=(12, 6))
    for i in range(samples):
        image, mask = dataset[idx]
        image = image.permute(1, 2, 0).numpy()
        mask = mask.permute(1, 2, 0).numpy()
        ax.ravel()[i * 2].imshow(image)
        ax.ravel()[i * 2].set_title("Image")
        ax.ravel()[i * 2].set_axis_off()
        ax.ravel()[i * 2 + 1].imshow(mask, cmap='gray')
        ax.ravel()[i * 2 + 1].set_title("Mask")
        ax.ravel()[i * 2 + 1].set_axis_off()
    plt.tight_layout()
    plt.show()

def save_metrics(save_dir, prefix, train_losses, train_dcs, valid_losses, valid_dcs):
    metrics = {
        'train_losses': train_losses,
        'train_dcs': train_dcs,
        'valid_losses': valid_losses,
        'valid_dcs': valid_dcs
    }

    with open(os.path.join(save_dir, f"metrics_{prefix}.json"), 'w') as f:
        json.dump(metrics, f)

# Print metrics as a table, inlcude precision-recall curve, sensitivity, specificity, accuracy, AUC and dice coefficient
def print_metrics(epoch, prc, rec, sen, spe, acc, auc, dice):

    # Convert to numpy array
    prc = prc.cpu()
    rec = rec.cpu()   
    sen = sen.cpu()
    spe = spe.cpu()
    acc = acc.cpu()
    auc = auc.cpu()
    dice = torch.tensor(dice)

    print(f"Epoch {epoch}")
    print(f"{'Precision':<15}{'Recall':<15}{'Sensitivity':<15}{'Specificity':<15}{'Accuracy':<15}{'AUC':<15}{'Dice':<15}")
    # print(f"{prc.item():<15.4f}{rec.item():<15.4f}{sen.item():<15.4f}{spe.item():<15.4f}{acc.item():<15.4f}{auc.item():<15.4f}{dice.item():<15.4f}")
    print(f'{prc}   {rec}    {sen}    {spe}    {acc}    {auc}    {dice}')

# Path: datasets/gdxray
if __name__ == "__main__":
    import torch
    config = {
        "name": "gdxray",
        "data_dir": os.path.join(os.path.dirname(CURRENT_DIR), "data/gdxray"),
        "metadata": os.path.join(os.path.dirname(CURRENT_DIR), "metadata"),
        "subset": "train",
    }

    # transform = v2.Compose([v2.Resize((224, 224)), v2.RandomRotation(0.5), v2.ToTensor()])

    # dataset = GDXrayDataset(config, labels=True, transform=transform)

    # loader = DataLoader(dataset, batch_size=2, shuffle=True)

    # for i, (image, label) in enumerate(loader):
    #     print(image.shape, label.shape)
    #     if i == 0:
    #         break

    # visualize_samples(dataset, num_samples=3, labels=True)

    # visualize_augmentations(dataset, num_samples=3)

    # print metrics
    prc = torch.tensor(0.8)
    rec = torch.tensor(0.6)
    sen = torch.tensor(0.9)
    spe = torch.tensor(0.7)
    acc = torch.tensor(0.85)
    auc = torch.tensor(0.95)
    dice = torch.tensor(0.75)
    print_metrics(1, prc, rec,sen, spe, acc, auc, dice)
