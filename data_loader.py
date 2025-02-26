from __future__ import print_function, division

import random

import numpy as np
import torch
from skimage import color
from skimage import io
from skimage import transform
from torch.utils.data import Dataset


class RescaleT(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if image is None or image.size == 0:
            raise ValueError(f"Invalid image: {image}")

        h, w = image.shape[:2]

        # Ensure output size is valid
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = (self.output_size * h // w, self.output_size)
            else:
                new_h, new_w = (self.output_size, self.output_size * w // h)
        else:
            new_h, new_w = self.output_size

        new_h, new_w = max(1, int(new_h)), max(1, int(new_w))  # Prevent zero or negative values

        # Ensure `new_h` and `new_w` are not larger than original dimensions
        new_h = min(new_h, h)
        new_w = min(new_w, w)

        # Convert tensors to NumPy arrays for skimage operations
        # Ensure image is in (H, W, C) format before extracting h/w
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)  # (C, H, W) → (H, W, C)
        if isinstance(label, torch.Tensor):
            label = label.numpy().squeeze()  # Remove all singleton dimensions

        img = transform.resize(image, (new_h, new_w), mode='constant', anti_aliasing=True)
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        # Convert back to tensor
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()  # (C, H, W)
        lbl = torch.from_numpy(lbl).unsqueeze(0).float()

        print(f"Rescaled image shape: {img.shape}, label shape: {lbl.shape}")

        return {'imidx': imidx, 'image': img, 'label': lbl}


class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            print(f"Image type: {type(image)}, shape: {image.shape}")
            image = np.array(image)[::-1].copy()
            label = np.array(label)[::-1].copy()

        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        # #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
        img = transform.resize(image, (new_h, new_w), mode='constant')
        lbl = transform.resize(label, (new_h, new_w), mode='constant', order=0, preserve_range=True)

        return {'imidx': imidx, 'image': img, 'label': lbl}


class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        if random.random() >= 0.5:
            print(f"Image type: {type(image)}, shape: {image.shape}")
            image = np.array(image)[::-1].copy()
            label = np.array(label)[::-1].copy()

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        print(f"h: {h}, new_h: {new_h}, w: {w}, new_w: {new_w}")
        if h - new_h <= 0:
            new_h = h  # Ensure new_h is not greater than h
        if w - new_w <= 0:
            new_w = w  # Ensure new_w is not greater than w

        top = np.random.randint(0, max(1, h - new_h))
        left = np.random.randint(0, max(1, w - new_w))

        image = image[top: top + new_h, left: left + new_w]
        label = label[top: top + new_h, left: left + new_w]

        return {'imidx': imidx, 'image': image, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # Convert to float tensors
        image = image.float()
        label = label.float()

        # Normalize using PyTorch ops
        image = image / image.max()
        label = label if (label.max() < 1e-6) else label / label.max()

        tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
        tmpLbl = np.zeros(label.shape)

        if image.shape[2] == 1:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
        else:
            tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
            tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
            tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class ToTensorLab(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):

        imidx, image, label = sample['imidx'], sample['image'], sample['label']

        # Convert tensors to NumPy if needed
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
        # Convert label to numpy and ensure it's 2D
        if isinstance(label, torch.Tensor):
            label = label.squeeze().numpy()  # Remove extra dimensions

        tmpLbl = np.zeros(label.shape)

        label = label if (label.max() < 1e-6) else label / label.max()

        # change the color space
        if self.flag == 2:  # with rgb and Lab colors
            tmpImg = np.zeros((image.shape[0], image.shape[1], 6))
            tmpImgt = np.zeros((image.shape[0], image.shape[1], 3))
            if image.shape[2] == 1:
                tmpImgt[:, :, 0] = image[:, :, 0]
                tmpImgt[:, :, 1] = image[:, :, 0]
                tmpImgt[:, :, 2] = image[:, :, 0]
            else:
                tmpImgt = image
            tmpImgtl = color.rgb2lab(tmpImgt)

            # nomalize image to range [0,1]
            tmpImg[:, :, 0] = (tmpImgt[:, :, 0] - np.min(tmpImgt[:, :, 0])) / (
                        np.max(tmpImgt[:, :, 0]) - np.min(tmpImgt[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImgt[:, :, 1] - np.min(tmpImgt[:, :, 1])) / (
                        np.max(tmpImgt[:, :, 1]) - np.min(tmpImgt[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImgt[:, :, 2] - np.min(tmpImgt[:, :, 2])) / (
                        np.max(tmpImgt[:, :, 2]) - np.min(tmpImgt[:, :, 2]))
            tmpImg[:, :, 3] = (tmpImgtl[:, :, 0] - np.min(tmpImgtl[:, :, 0])) / (
                        np.max(tmpImgtl[:, :, 0]) - np.min(tmpImgtl[:, :, 0]))
            tmpImg[:, :, 4] = (tmpImgtl[:, :, 1] - np.min(tmpImgtl[:, :, 1])) / (
                        np.max(tmpImgtl[:, :, 1]) - np.min(tmpImgtl[:, :, 1]))
            tmpImg[:, :, 5] = (tmpImgtl[:, :, 2] - np.min(tmpImgtl[:, :, 2])) / (
                        np.max(tmpImgtl[:, :, 2]) - np.min(tmpImgtl[:, :, 2]))

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])
            tmpImg[:, :, 3] = (tmpImg[:, :, 3] - np.mean(tmpImg[:, :, 3])) / np.std(tmpImg[:, :, 3])
            tmpImg[:, :, 4] = (tmpImg[:, :, 4] - np.mean(tmpImg[:, :, 4])) / np.std(tmpImg[:, :, 4])
            tmpImg[:, :, 5] = (tmpImg[:, :, 5] - np.mean(tmpImg[:, :, 5])) / np.std(tmpImg[:, :, 5])

        elif self.flag == 1:  #with Lab color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))

            if image.shape[2] == 1:
                tmpImg[:, :, 0] = image[:, :, 0]
                tmpImg[:, :, 1] = image[:, :, 0]
                tmpImg[:, :, 2] = image[:, :, 0]
            else:
                tmpImg = image

            tmpImg = color.rgb2lab(tmpImg)

            # tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.min(tmpImg[:, :, 0])) / (
                        np.max(tmpImg[:, :, 0]) - np.min(tmpImg[:, :, 0]))
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.min(tmpImg[:, :, 1])) / (
                        np.max(tmpImg[:, :, 1]) - np.min(tmpImg[:, :, 1]))
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.min(tmpImg[:, :, 2])) / (
                        np.max(tmpImg[:, :, 2]) - np.min(tmpImg[:, :, 2]))

            tmpImg[:, :, 0] = (tmpImg[:, :, 0] - np.mean(tmpImg[:, :, 0])) / np.std(tmpImg[:, :, 0])
            tmpImg[:, :, 1] = (tmpImg[:, :, 1] - np.mean(tmpImg[:, :, 1])) / np.std(tmpImg[:, :, 1])
            tmpImg[:, :, 2] = (tmpImg[:, :, 2] - np.mean(tmpImg[:, :, 2])) / np.std(tmpImg[:, :, 2])

        else:  # with rgb color
            tmpImg = np.zeros((image.shape[0], image.shape[1], 3))
            image = image / np.max(image)
            if image.shape[2] == 1:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 2] = (image[:, :, 0] - 0.485) / 0.229
            else:
                tmpImg[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
                tmpImg[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
                tmpImg[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225

        tmpLbl[:, :, 0] = label[:, :, 0]

        tmpImg = tmpImg.transpose((2, 0, 1))
        tmpLbl = label.transpose((2, 0, 1))

        return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg), 'label': torch.from_numpy(tmpLbl)}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.image_name_list = img_name_list
        self.label_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.image_name_list)

    def __getitem__(self, idx):
        # Load image
        image = io.imread(self.image_name_list[idx])
        imidx = torch.tensor([idx], dtype=torch.int32)

        # Load label or create an empty mask if not available
        if len(self.label_name_list) == 0:
            label = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            label = io.imread(self.label_name_list[idx]).astype(np.float32)
            if label.ndim == 3:  # Convert to grayscale if necessary
                label = label[:, :, 0]

        # Ensure both image and label have correct dimensions
        if image.ndim == 2:  # Convert grayscale image to 3-channel
            image = np.stack([image] * 3, axis=-1)

        label = np.expand_dims(label, axis=-1)  # Ensure label has a channel dimension

        # Keep as NumPy arrays:
        image = image.transpose((2, 0, 1))  # (H,W,C) → (C,H,W) for PyTorch
        label = label.transpose((2, 0, 1))

        # Load image and label as NumPy arrays (no torch conversion here)
        sample = {"imidx": imidx, "image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)  # Let transforms handle processing

        # Final tensor conversion if not done by transforms
        if not isinstance(sample["image"], torch.Tensor):
            sample["image"] = torch.from_numpy(sample["image"]).float()
        if not isinstance(sample["label"], torch.Tensor):
            sample["label"] = torch.from_numpy(sample["label"]).float()

        return sample
