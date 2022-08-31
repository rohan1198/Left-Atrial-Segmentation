import os
import scipy
import numpy as np
import SimpleITK as sitk
import skimage.transform
from typing import Tuple

import torch

from lib.utils import patch_builder, Transformer, distance_map


class LocatorDataGenerator(object):
    def __init__(
        self,
        root_dir: str,
        mode: str,
        scale_factor: Tuple = (
            0.5,
            0.5,
            0.25)):
        assert mode in ["train", "test", "validate",
                        "paths_train", "paths_validate"], "Wrong mode provided!"

        self.mode = mode
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        split = mode.split("_")[-1] if mode.startswith("paths") else mode

        self.img_dir = os.path.join(root_dir, split, "raw")
        assert os.path.exists(
            self.img_dir), f"Raw folder ('raw') doesn't exist within {split} directory"
        self.img_files = sorted(os.listdir(self.img_dir))

        if self.mode != "test":
            self.mask_dir = os.path.join(root_dir, split, "label")
            assert os.path.exists(
                self.mask_dir), f"Label folder ('label') does not exist within {split} directory"
            self.mask_files = sorted(os.listdir(self.mask_dir))
        else:
            self.mask_files = []

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        raw_image_path = os.path.join(self.img_dir, self.img_files[idx])
        raw_image_array = sitk.GetArrayFromImage(
            sitk.Cast(sitk.ReadImage(raw_image_path), sitk.sitkUInt8))
        raw_depth, raw_height, raw_width = raw_image_array.shape

        if (raw_depth, raw_height, raw_width) != (96, 640, 640):
            assert all(x <= y for x, y in zip((raw_depth, raw_height, raw_width), (96, 640, 640))
                       ), f'Cannot perform padding for {raw_image_path}; data larger than expected'
            raw_image_array = np.pad(raw_image_array,
                                     ((int((96 - raw_depth) / 2),
                                       int((96 - raw_depth) / 2)),
                                         (int((640 - raw_height) / 2),
                                          int((640 - raw_height) / 2)),
                                         (int((640 - raw_width) / 2),
                                          int((640 - raw_width) / 2))),
                                     'constant',
                                     constant_values=0)

        raw_image_array = skimage.transform.rescale(
            raw_image_array,
            scale=self.scale_factor,
            order=0,
            preserve_range=True,
            anti_aliasing=True)
        raw_image_transformed = Transformer(raw_image_array, image_type='raw')

        if self.mode in ["paths_train", "paths_validate", "test"]:
            raw_images = [raw_image_path, raw_image_transformed()[0]]

        elif self.mode in ["train", "validate"]:
            raw_images = raw_image_transformed()[0]

        if self.mode == "test":
            return raw_images

        else:
            label_image_path = os.path.join(
                self.mask_dir, self.mask_files[idx])
            label_image_array = sitk.GetArrayFromImage(
                sitk.Cast(sitk.ReadImage(label_image_path), sitk.sitkUInt8))
            label_depth, label_height, label_width = label_image_array.shape

            if (label_depth, label_height, label_width) != (96, 640, 640):
                assert all(x <= y for x, y in zip((label_depth, label_height, label_width), (96, 640, 640))
                           ), f'Cannot perform padding for {label_image_path}, data larger than expected'
                label_image_array = np.pad(label_image_array,
                                           ((int((96 - label_depth) / 2),
                                             int((96 - label_depth) / 2)),
                                               (int((640 - label_height) / 2),
                                                int((640 - label_height) / 2)),
                                               (int((640 - label_width) / 2),
                                                int((640 - label_width) / 2))),
                                           'constant',
                                           constant_values=0)

            # Convert the label array to binary
            label_image_array[label_image_array == 255] = 1
            label_image_array = scipy.ndimage.zoom(
                label_image_array, zoom=self.scale_factor, order=0)
            label_image_transformed = Transformer(
                label_image_array, image_type='label')

            if self.mode in ["paths_train", "paths_validate"]:
                label_images = [label_image_path, label_image_transformed()[0]]

            elif self.mode in ["train", "validate"]:
                label_images = label_image_transformed()[0]

            assert len(raw_images) == len(label_images)

            if self.mode in ['paths_train', 'paths_validate']:
                raw_file = raw_images[0].split('/')[-1].split('_')[0]
                label_file = label_images[0].split('/')[-1].split('_')[0]
                assert raw_file == label_file, 'Different files provided for raw and label images, ensure same set for both types'

            return raw_images, label_images


class DatasetGenerator(object):
    def __init__(
            self,
            mode,
            inputs,
            pad,
            scale_factor,
            loss_criterion='Dice',
            **kwargs):
        assert mode in [
            'train', 'test', 'validate'], 'Wrong mode provided. Must be either "train", "test" or "validate"'
        self.loss_criterion = None if mode == 'test' else loss_criterion
        self.mode = mode
        patch_dims = kwargs.get('patch_dims')
        self.patch_size, paths_slices = patch_builder(
            gen_set_inputs=inputs, pad=pad, scale_factor=scale_factor, patch_size=patch_dims)
        self.random_state = np.random.RandomState(kwargs.get('seed'))
        self.transform = kwargs.get(
            'transform') if self.mode == 'train' else None

        if self.mode == 'test':
            self.mean = kwargs.get('mean')
            assert self.mean is not None, 'Mean value must be provided'

            self.std = kwargs.get('std')
            assert self.std is not None, 'Standard deviation value must be provided'

            self.raw_transformed = []
            for data in paths_slices:
                raw_image_path = data[0][0]
                raw_image_file = raw_image_path.split('/')[-1]
                slice_range = data[1]
                raw_image_array = sitk.GetArrayFromImage(
                    sitk.Cast(
                        sitk.ReadImage(raw_image_path),
                        sitk.sitkUInt8))  # data is from 0 to 255 hence uint8 for efficiency
                raw_depth, raw_height, raw_width = raw_image_array.shape
                if (raw_depth, raw_height, raw_width) != (96, 640, 640):
                    raw_image_array = np.pad(raw_image_array,
                                             ((int((96 - raw_depth) / 2),
                                               int((96 - raw_depth) / 2)),
                                                 (int((640 - raw_height) / 2),
                                                  int((640 - raw_height) / 2)),
                                                 (int((640 - raw_width) / 2),
                                                  int((640 - raw_width) / 2))),
                                             'constant',
                                             constant_values=0)
                    raw_image_array = raw_image_array[slice_range]
                transformed_raw_image = Transformer(
                    raw_image_array,
                    image_type='raw',
                    mode='test',
                    mean=self.mean,
                    std=self.std)  # mode=test performs only normalisation and Tensor generation
                self.raw_transformed.append((raw_image_file, transformed_raw_image()[
                                            0], slice_range, (raw_depth, raw_height, raw_width)))

            self.label_transformed = None

        elif self.mode in ['train', 'validate']:
            self.raw_images = []
            for data in paths_slices:
                raw_image_path = data[0][0]
                raw_image_file = raw_image_path.split('/')[-1]
                slice_range = data[1]
                raw_image_array = sitk.GetArrayFromImage(
                    sitk.Cast(sitk.ReadImage(raw_image_path), sitk.sitkUInt8))
                raw_depth, raw_height, raw_width = raw_image_array.shape
                if (raw_depth, raw_height, raw_width) != (96, 640, 640):
                    raw_image_array = np.pad(raw_image_array,
                                             ((int((96 - raw_depth) / 2),
                                               int((96 - raw_depth) / 2)),
                                                 (int((640 - raw_height) / 2),
                                                  int((640 - raw_height) / 2)),
                                                 (int((640 - raw_width) / 2),
                                                  int((640 - raw_width) / 2))),
                                             'constant',
                                             constant_values=0)
                    raw_image_array = raw_image_array[slice_range]
                self.raw_images.append(raw_image_array)

            images_array = np.concatenate(
                [image.ravel() for image in self.raw_images])
            self.mean = kwargs.get('mean') if kwargs.get(
                'mean') is not None else np.mean(images_array)
            self.std = kwargs.get('std') if kwargs.get(
                'std') is not None else np.std(images_array)

            self.label_images = []
            for data in paths_slices:
                label_image_path = data[0][1]
                label_image_file = label_image_path.split('/')[-1]
                slice_range = data[1]
                label_image_array = sitk.GetArrayFromImage(sitk.Cast(
                    sitk.ReadImage(label_image_path), sitk.sitkUInt8))
                label_depth, label_height, label_width = label_image_array.shape
                if (label_depth, label_height, label_width) != (96, 640, 640):
                    label_image_array = np.pad(label_image_array,
                                               ((int((96 - label_depth) / 2),
                                                 int((96 - label_depth) / 2)),
                                                   (int((640 - label_height) / 2),
                                                    int((640 - label_height) / 2)),
                                                   (int((640 - label_width) / 2),
                                                    int((640 - label_width) / 2))),
                                               'constant',
                                               constant_values=0)
                # Convert the label array to binary
                label_image_array[label_image_array == 255] = 1
                label_image_array = label_image_array[slice_range]
                self.label_images.append(label_image_array)

            assert len(
                self.raw_images) == len(
                self.label_images), 'Unequal number of label files w.r.t to raw files'

            # if random transforms not needed then implement and store
            # precomputed tranforms(400 data points)
            if self.transform is None:
                self.raw_transformed = []
                self.label_transformed = []
                for _, images in enumerate(
                        zip(self.raw_images, self.label_images)):
                    raw_image, label_image = images
                    transformed_raw_image = Transformer(
                        raw_image, image_type='raw', mode=self.mode, mean=self.mean, std=self.std)
                    transformed_label_image = Transformer(
                        label_image,
                        image_type='label',
                        mode=self.mode)  # only normalisation and tensor generation

                    for raw_image in transformed_raw_image():
                        self.raw_transformed.append(raw_image)
                    for label_image in transformed_label_image():
                        self.label_transformed.append(label_image)

                if self.loss_criterion == 'HybridLoss':
                    self.maps = []
                    for image in self.label_transformed:
                        self.maps.append(distance_map(image))

        self.len = len(
            self.raw_transformed) if self.transform is None else len(
            self.raw_images)

    def __getitem__(self, idx):
        if self.mode == 'test':
            return self.raw_transformed[idx]

        elif self.transform is None and self.mode != 'test' and self.loss_criterion != 'HybridLoss':
            return self.raw_transformed[idx], self.label_transformed[idx], torch.Tensor([
            ])

        elif self.transform is None and self.mode != 'test' and self.loss_criterion == 'HybridLoss':
            return self.raw_transformed[idx], self.label_transformed[idx], self.maps[idx]

        # compute and provide the randomly generating images(80 data points)
        elif self.transform is not None and self.loss_criterion != 'HybridLoss':
            # based on the previously created fixed Random State object
            # generate seed values to be used for both label and raw images to
            # ensure both undergo similar tranformations
            seed = self.random_state.randint(0, 10000)
            raw_random_transformed = Transformer(
                self.raw_images[idx],
                image_type='raw',
                mode=self.transform,
                mean=self.mean,
                std=self.std,
                seed=seed)
            label_random_transformed = Transformer(
                self.label_images[idx],
                image_type='label',
                mode=self.transform,
                seed=seed)
            return raw_random_transformed(), label_random_transformed(), torch.Tensor([])

        elif self.transform is not None and self.loss_criterion == 'HybridLoss':
            seed = self.random_state.randint(0, 10000)
            raw_random_transformed = Transformer(
                self.raw_images[idx],
                image_type='raw',
                mode=self.transform,
                mean=self.mean,
                std=self.std,
                seed=seed)
            label_random_transformed = Transformer(
                self.label_images[idx],
                image_type='label',
                mode=self.transform,
                seed=seed)
            label = label_random_transformed()
            dist_map = distance_map(label)
            return raw_random_transformed(), label, dist_map

    def __len__(self):
        return self.len
