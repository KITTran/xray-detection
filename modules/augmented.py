%load_ext autoreload
%autoreload 2

import utils
import torch
import cv2

from PIL import Image
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode, pad, rotate, affine

# Define the transformations as functionss

class MyGammaTransform(torch.nn.Module):
    def __init__(self, c=1, gamma=0.7):
        super(MyGammaTransform, self).__init__()
        self.c = c
        self.gamma = gamma

    def forward(self, img, label = None):
        img = v2.PILToTensor()(img) / 255.0
        transformed = self.c * (img ** self.gamma)
        transformed = torch.clamp(transformed * 255.0, 0, 255).type(torch.float32)
        return transformed, label if label is not None else transformed

class MyLinearTransform(torch.nn.Module):
    def __init__(self, k=0.85, b=0.3):
        super(MyLinearTransform, self).__init__()
        self.k = k
        self.b = b

    def forward(self, img, label = None):
        img = v2.PILToTensor()(img) / 255.0
        transformed = self.k * img + self.b
        transformed = torch.clamp(transformed * 255.0, 0, 255).type(torch.float32)
        return transformed, label if label is not None else transformed

# class MyHistogramEqualization(torch.nn.Module):
#     def __init__(self):
#         super(MyHistogramEqualization, self).__init__()

#     def forward(self, img, label = None):
#         img = img.numpy()

#         if len(img.shape) == 3 and img.shape[0] == 3:  # Color image
#             img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
#             img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
#             equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
#         else:  # Grayscale image
#             equalized = cv2.equalizeHist(img)

#         print('Dtype of equalized:', equalized.dtype)
#         print('Shape of equalized:', equalized.shape)

        return torch.from_numpy(equalized), label if label is not None else torch.from_numpy(equalized)

class MyUniformCrop(torch.nn.Module):
    def __init__(self, patch_size):
        super(MyUniformCrop, self).__init__()
        self.patch_size = patch_size

    def forward(self, img, label = None):
        width, height = img.size
        crop_width, crop_height = self.patch_size
        left = (width - crop_width) // 2
        upper = (height - crop_height) // 2
        right = left + crop_width
        lower = upper + crop_height
        return img.crop((left, upper, right, lower)), label.crop((left, upper, right, lower)) if label is not None else img.crop((left, upper, right, lower))

class MyRandomCrop(torch.nn.Module):
    def __init__(self, patch_size, crop_number=2):
        super(MyRandomCrop, self).__init__()
        self.patch_size = patch_size
        self.crop_number = crop_number

    def forward(self, img, label = None):
        width, height = img.size
        crop_width, crop_height = self.patch_size
        crops = []
        for _ in range(self.crop_number):
            left = torch.randint(0, width - crop_width, (1,)).item()
            upper = torch.randint(0, height - crop_height, (1,)).item()
            right = left + crop_width
            lower = upper + crop_height
            cropped_img = img.crop((left, upper, right, lower))

            if label is not None:
                cropped_label = label.crop((left, upper, right, lower))
                crops.append((cropped_img, cropped_label))
                continue

            crops.append(cropped_img)
        return crops

class MyRandomCutout(torch.nn.Module):
    def __init__(self, num_holes=10, max_h_size=40, max_w_size=40, fill_value=0):
        super(MyRandomCutout, self).__init__()
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value

    def forward(self, img, label=None):
        img_tensor = v2.PILToTensor()(img)
        label_tensor = v2.PILToTensor()(label) if label is not None else img_tensor.clone()
        for _ in range(self.num_holes):
            y = torch.randint(0, img_tensor.shape[1] - self.max_h_size, (1,)).item()
            x = torch.randint(0, img_tensor.shape[2] - self.max_w_size, (1,)).item()
            img_tensor[:, y:y + self.max_h_size, x:x + self.max_w_size] = self.fill_value
            label_tensor[:, y:y + self.max_h_size, x:x + self.max_w_size] = self.fill_value
        return Image.fromarray(img_tensor.numpy()), Image.fromarray(label_tensor.numpy()) if label is not None else Image.fromarray(img_tensor.numpy())

class MyRandomRescale(torch.nn.Module):
    def __init__(self, scale_range=(0.5, 1.5)):
        super().__init__()
        self.scale_range = scale_range

    def forward(self, img, label = None):

        assert label is not None, "Label must be provided for rescaling"

        original_width, original_height = img.size
        scale_factor = torch.empty(1).uniform_(*self.scale_range).item()
        scaled_width = int(original_width * scale_factor)
        scaled_height = int(original_height * scale_factor)

        img = img.resize((scaled_width, scaled_height), Image.BILINEAR)
        label = label.resize((scaled_width, scaled_height), Image.NEAREST)

        if scale_factor > 1.0:  # Randomly crop back to original size
            left = torch.randint(0, scaled_width - original_width, (1,)).item()
            upper = torch.randint(0, scaled_height - original_height, (1,)).item()
            img = img.crop((left, upper, left + original_width, upper + original_height))
            label = label.crop((left, upper, left + original_width, upper + original_height))
        elif scale_factor < 1.0:  # Mirror and extend to original size
            delta_width = original_width - scaled_width
            delta_height = original_height - scaled_height

            pad_left = delta_width // 2
            pad_right = delta_width - pad_left
            pad_top = delta_height // 2
            pad_bottom = delta_height - pad_top

            img = pad(img, (pad_left, pad_top, pad_right, pad_bottom), padding_mode="reflect")
            label = pad(label, (pad_left, pad_top, pad_right, pad_bottom), padding_mode="reflect")

        return img, label

class MyRandomShift(torch.nn.Module):
    def __init__(self):
        super(MyRandomShift, self).__init__()

    def forward(self, img, label = None):
        original_width, original_height = img.size

        # Randomly select shift amounts
        shift_x = torch.randint(-original_width // 4, original_width // 4, (1,)).item()
        shift_y = torch.randint(-original_height // 4, original_height // 4, (1,)).item()

        # Shift the image and mask
        img = affine(img, angle=0, translate=(shift_x, shift_y), scale=1, shear=(0, 0), interpolation =InterpolationMode.BILINEAR, fill=None)

        if label is not None:
            label = affine(label, angle=0, translate=(shift_x, shift_y), scale=1, shear=(0, 0), interpolation =InterpolationMode.NEAREST, fill=None)

        # Mirror padding to restore original size
        img = pad(img, (-min(shift_x, 0), -min(shift_y, 0), max(shift_x, 0), max(shift_y, 0)), padding_mode="reflect")

        if label is not None:
            label = pad(label, (-min(shift_x, 0), -min(shift_y, 0), max(shift_x, 0), max(shift_y, 0)), padding_mode="reflect")

        # Crop back to original size
        img = img.crop((0, 0, original_width, original_height))

        if label is not None:
            label = label.crop((0, 0, original_width, original_height))

        return img, label if label is not None else img

class MyRandomRotate(torch.nn.Module):
    def __init__(self, angle_range=(-30, 30)):
        super(MyRandomRotate, self).__init__()
        self.angle_range = angle_range

    def forward(self, img, label=None):
        original_width, original_height = img.size

        # Randomly select an angle within the specified range
        angle = torch.empty(1).uniform_(*self.angle_range).item()

        # Rotate the image and label
        img = rotate(img, angle=angle, interpolation=InterpolationMode.BILINEAR, expand=True)
        if label is not None:
            label = rotate(label, angle=angle, interpolation=InterpolationMode.NEAREST, expand=True)

        # Calculate padding needed to restore original size
        pad_left = (img.width - original_width) // 2
        pad_right = img.width - original_width - pad_left
        pad_top = (img.height - original_height) // 2
        pad_bottom = img.height - original_height - pad_top

        # Mirror padding to restore original size
        img = pad(img, (-pad_left, -pad_top, -pad_right, -pad_bottom), padding_mode="reflect")
        if label is not None:
            label = pad(label, (-pad_left, -pad_top, -pad_right, -pad_bottom), padding_mode="reflect")

        # Crop back to original size
        img = img.crop((0, 0, original_width, original_height))
        if label is not None:
            label = label.crop((0, 0, original_width, original_height))

        return img, label if label is not None else img

def unet_augmentation_train(patch_size, rotate_angle=180, noise_mean=0, noise_cov=0.02):
        return v2.Compose([
            v2.Resize(patch_size, interpolation=InterpolationMode.BILINEAR),
            v2.RandomEqualize(p = 1),
            MyGammaTransform(c = 1, gamma = 0.5),
            MyLinearTransform(k = 0.4, b = 0.2),
            # v2.LinearTransformation(),
            # MyHistogramEqualization(),
            v2.RandomRotation(degrees=rotate_angle, interpolation=InterpolationMode.BILINEAR),
            v2.PILToTensor(),
            v2.ToDtype(torch.float32, scale=True),
            v2.GaussianNoise(mean=noise_mean, sigma=noise_cov),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

def unet_augmentation_valid(patch_size):
    return v2.Compose([
        v2.Resize(patch_size, interpolation=InterpolationMode.BILINEAR),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def reshdc_augmentation_train(patch_size, rotate_angle=(1, 3), crop_num=2, num_holes=10, gauss_mean=0.2, gauss_sigma=0.3):
    return v2.Compose([
        v2.RandomRotation(degrees=rotate_angle),
        MyRandomRescale(scale_range=(0.5, 1.5)),
        MyRandomShift(),
        MyRandomRotate(angle_range=(-30, 30)),
        v2.RandomHorizontalFlip(),
        # MyRandomCutout(num_holes=num_holes, max_h_size=40, max_w_size=40, fill_value=0),
        # v2.RandomErasing(scale = (0.02, 0.1)),
        v2.RandomCrop(patch_size),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        v2.GaussianNoise(mean=gauss_mean, sigma=gauss_sigma),
        # MyRandomCrop(patch_size, crop_num),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def reshdc_augmentation_valid(patch_size):
    return v2.Compose([
        MyUniformCrop(patch_size),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

if __name__ == '__main__':
    config = {
        "name": "gdxray",
        "data_dir": "../data/gdxray",
        "metadata": "../metadata",
        "subset": "train",
        "image_size": (320, 640),
    }

    # transform = unet_augmentation_train(config["image_size"])
    transform = reshdc_augmentation_train(config["image_size"])

    dataset = utils.GDXrayDataset(config, labels=True, transform=transform)

    # Crop the image to a fixed size and visualize the results

    utils.visualize_samples(dataset, 5)
    # utils.visualize_augmentations(dataset, 6, samples=10, cols=5)

    # print(img.min(), img.max(), img.shape)

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # for idx, img_mask in enumerate(dataloader):
    #     print(img_mask[0].shape, img_mask[1].shape)
    #     break
