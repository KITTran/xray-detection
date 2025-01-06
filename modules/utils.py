import os
import math
import random

import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import io

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

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Number of classification classes (including background)
    NUM_CLASSES = 1  # Override in sub-classes

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

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

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 200

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

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
                'img_size': tuple, # Size of the image }
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
        metadata_img = "../metadata/{}_{}.txt".format(config["name"], "images")
        metadata_label = "../metadata/{}_{}.txt".format(config["name"], "labels")

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

            self.add_image(config["name"], image_id, img_path, label=label_path)

    def __len__(self):
        return len(self.image_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: (image, label) where both are transformed.
        """

        image = io.read_image(self.image_info[idx]["path"], io.ImageReadMode.RGB)
        channel, height, width = image.shape
        self.update_info(idx, height_org=height, width_org=width)

        if self.labels:
            label = io.read_image(self.image_info[idx]["label"], io.ImageReadMode.GRAY)
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)

            return image, label

        if self.transform:
            image = self.transform(image)

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

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
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

        # Convert tensors to NumPy arrays for visualization
        image = image.permute(1, 2, 0).numpy()
        label = label.permute(1, 2, 0).squeeze().numpy()

        # Display the image
        axes[i][0].imshow(image)
        axes[i][0].set_title(
            f"Image: {image_path.split('/')[-1].split('.')[0]} -- Size: {image.shape}"
        )
        axes[i][0].axis("off")

        # Display the label
        axes[i][1].imshow(label, cmap="gray")
        axes[i][1].set_title(
            f"Label: {label_path.split('/')[-1].split('.')[0]} -- Size: {label.shape}"
        )
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()


# Path: datasets/gdxray
if __name__ == "__main__":
    config = {
        "name": "gdxray",
        "data_dir": os.path.join(os.path.dirname(CURRENT_DIR), "data/gdxray"),
        "subset": "train",
        "metadata": "metadata",
        "img_size": (512, 512),
    }

    transform = A.Compose([A.Resize(224, 224), A.RandomRotate90(0.5)])

    dataset = GDXrayDataset(config, labels=True, transform=transform)

    print("Number of images: ", len(dataset))

    # Visualize
    visualize_samples(dataset, num_samples=3)
