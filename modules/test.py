import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageLabelDataset(Dataset):
    def __init__(self, datadir, transform=None):
        """
        Args:
            datadir (str): Path to the dataset directory containing 'image' and 'label' subdirectories.
            transform (callable, optional): Transform to be applied to the images.
        """
        self.image_dir = os.path.join(datadir, 'W0001')
        self.label_dir = os.path.join(datadir, 'W0002')
        self.transform = transform

        # Ensure both directories exist
        if not os.path.isdir(self.image_dir) or not os.path.isdir(self.label_dir):
            raise FileNotFoundError("Both 'image' and 'label' directories must exist in the dataset directory.")

        # Get sorted lists of image and label file paths
        self.image_paths = sorted(
            [os.path.join(self.image_dir, fname) for fname in os.listdir(self.image_dir) if fname.endswith(('.jpg', '.png'))]
        )
        self.label_paths = sorted(
            [os.path.join(self.label_dir, fname) for fname in os.listdir(self.label_dir) if fname.endswith(('.jpg', '.png'))]
        )

        # Ensure the number of images matches the number of labels
        if len(self.image_paths) != len(self.label_paths):
            raise ValueError("The number of images and labels must be the same.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: (image, label) where both are transformed.
        """
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image and label
        image = Image.open(image_path).convert("RGB")  # Convert to RGB
        label = Image.open(label_path).convert("L")    # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)  # Apply same transform to the label

        return image, label

import random
import matplotlib.pyplot as plt

def visualize_samples(dataset, num_samples=3):
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
        image_path = dataset.image_paths[idx]
        label_path = dataset.label_paths[idx]

        # Convert tensors to NumPy arrays for visualization
        image = image.permute(1, 2, 0).numpy()
        label = label.permute(1, 2, 0).squeeze().numpy()

        # Display the image
        axes[i][0].imshow(image)
        axes[i][0].set_title(f"Image: {image_path.split('/')[-1]}")
        axes[i][0].axis('off')

        # Display the label
        axes[i][1].imshow(label, cmap='gray')
        axes[i][1].set_title(f"Label: {label_path.split('/')[-1]}")
        axes[i][1].axis('off')

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    datadir = "/home/tuank/projects/cracked-detection/dataset/gdxray/welding"

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 128x128
        transforms.ToTensor(),         # Convert images to PyTorch tensors
    ])

    # Create dataset instance
    dataset = ImageLabelDataset(datadir, transform=transform)

    # Get a sample
    img, lbl = dataset[0]
    print(f"Image shape: {img.shape}, Label shape: {lbl.shape}")

    # Visualize samples
    visualize_samples(dataset, num_samples=3)
