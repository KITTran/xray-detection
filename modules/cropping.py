from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

@dataclass
class PatchData:
    """Data class to hold patch information."""
    image_patch: np.ndarray
    combined_mask: np.ndarray
    instance_masks: List[np.ndarray]
    image_name: str

def crop_and_separate_patches(image_dir: Path, mask_dir: Path) -> Tuple[List[PatchData], List[PatchData]]:
    """
    Crop images and their corresponding masks into 3 equal patches and separate them
    based on whether the mask contains any non-zero pixels. All masks for an image are
    combined before creating patches.
    
    Args:
        image_dir: Path to directory containing images
        mask_dir: Path to directory containing masks
        
    Returns:
        Tuple containing lists of empty and defect patches
    """
    empty_patches: List[PatchData] = []
    defect_patches: List[PatchData] = []
    
    # Get all image files
    image_files = list(image_dir.glob("*.jpg"))
    
    for img_path in image_files:

        print(f"Processing image: {img_path}")

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        
        # Get all corresponding mask files for this image
        mask_files = list(mask_dir.glob(f"{img_path.stem}_*.png"))
        
        if not mask_files:
            print(f"No mask files found for image: {img_path.name}")
            continue
            
        # Initialize combined mask with zeros
        combined_mask: Optional[np.ndarray] = None
        instance_masks: List[np.ndarray] = []
        
        # Load and combine all masks
        for mask_path in mask_files:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Could not read mask: {mask_path}")
                continue
                
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                # Combine masks using logical OR
                combined_mask = cv2.bitwise_or(combined_mask, mask)

            instance_masks.append(mask)
        
        if combined_mask is None:
            print(f"Cannot combine mask for image: {img_path.name}")
            continue
            
        # Calculate patch width (divide image width by 3)
        patch_width = img.shape[1] // 3
        
        # Create 3 patches
        for i in range(3):
            start_x = i * patch_width
            end_x = (i + 1) * patch_width
            
            # Extract patches
            img_patch = img[:, start_x:end_x]
            mask_patch = combined_mask[:, start_x:end_x]
            instance_masks_patch = [mask[:, start_x:end_x] for mask in instance_masks]

            patch_data = PatchData(
                image_patch=img_patch,
                combined_mask=mask_patch,
                instance_masks=instance_masks_patch,
                image_name=img_path.name
            )

            # Check if mask patch contains any defects
            if np.any(mask_patch > 0):
                defect_patches.append(patch_data)
            else:
                empty_patches.append(patch_data)
        
    return empty_patches, defect_patches

def display_patches(patches: List[PatchData], title: str, num_samples: int = 3) -> None:
    """Display random samples of patches with their masks."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(title)
    
    # Randomly sample patches
    sample_patches = random.sample(patches, min(num_samples, len(patches)))
    
    for idx, patch in enumerate(sample_patches):
        # Display original image patch
        axes[0, idx].imshow(patch.image_patch, cmap='gray')
        axes[0, idx].axis('off')
        axes[0, idx].set_title(f'Original - {patch.image_name}')
        
        # Display mask
        axes[1, idx].imshow(patch.combined_mask, cmap='gray')
        axes[1, idx].axis('off')
        axes[1, idx].set_title(f'Mask - {patch.image_name}')
        
    plt.tight_layout()
    plt.show()

def save_patches(patches: List[PatchData], output_dir: Path, is_empty_patch: bool = False) -> None:
    """
    Save image patches and their instance masks to the specified output directory.
    
    Args:
        patches: List of PatchData objects containing patch information
        output_dir: Directory to save the patches and masks
        is_empty_patch: Whether the patches are empty patches (all masks are black)
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / 'masks'
    masks_dir.mkdir(exist_ok=True)
    
    # Dictionary to keep track of image indices for each base image
    img_indices: dict[str, int] = {}
    
    for patch in patches:
        base_name = Path(patch.image_name).stem
        
        # Get or initialize the index for this base image
        if base_name not in img_indices:
            img_indices[base_name] = 0
        else:
            img_indices[base_name] += 1
            
        # Save image patch
        img_patch_name = f"{base_name}-{img_indices[base_name]}.png"
        img_patch_path = output_dir / img_patch_name
        cv2.imwrite(str(img_patch_path), patch.image_patch)
        
        # Save instance masks
        if is_empty_patch:
            # For empty patches, save only one black mask
            mask_name = f"{base_name}-{img_indices[base_name]}_0.png"
            mask_path = masks_dir / mask_name
            cv2.imwrite(str(mask_path), np.zeros_like(patch.instance_masks[0]))
        else:
            # For defect patches, save only masks with non-zero pixels
            mask_index = 0
            for instance_mask in patch.instance_masks:
                if np.any(instance_mask > 0):
                    mask_name = f"{base_name}-{img_indices[base_name]}_{mask_index}.png"
                    mask_path = masks_dir / mask_name
                    cv2.imwrite(str(mask_path), instance_mask)
                    mask_index += 1

def main() -> None:
    """Main function to demonstrate usage."""
    # Example usage
    image_dir = Path("../data/xray-gt/allType")
    mask_dir = Path("../data/xray-gt/allType/masks")
    output_dir = Path("../data/xray-gt/cropAllType/")
    
    empty_patches, defect_patches = crop_and_separate_patches(image_dir, mask_dir)
    
    print(f"Number of empty patches: {len(empty_patches)}")
    print(f"Number of defect patches: {len(defect_patches)}")

    # Display samples
    display_patches(empty_patches, "Random Samples of Empty Patches")
    display_patches(defect_patches, "Random Samples of Defect Patches")

    # Save patches
    save_patches(empty_patches, output_dir / 'empty', is_empty_patch=True)
    save_patches(defect_patches, output_dir / 'defect', is_empty_patch=False)

if __name__ == "__main__":
    main()