import logging
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as distance

import torch
import torch.nn as nn
from torchvision.transforms import Compose

from lib.augment import *


loggers = {}


def get_logger(name: str, level=logging.INFO):
    global loggers

    if loggers.get(name) is not None:
        return loggers[name]

    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)

        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d::%H:%M:%S')  # message format
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(f'{name}.log')
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        loggers[name] = logger

        return logger


# Calculates the number of parameters for a supplied model
def number_of_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    return sum([np.prod(p.size()) for p in model_parameters])


# Used by Data Generator to define patches that contain mostly the Left atrium structure within raw and label images. This helps reduce the size of data
# consumed by the GPU.
def patch_builder(gen_set_inputs, pad, scale_factor, patch_size):
    sigmoid = nn.Sigmoid()
    gen_set_ranges = []
    for data in gen_set_inputs:
        raw_image_path, image = data[0]
        label_image_path = data[1][0] if data[1] is not None else None
        # store image paths so as to provide paths along with corresponding
        # slices at function output
        image_paths = (raw_image_path, label_image_path)
        image = sigmoid(image).detach().cpu().numpy()
        shape = image[0, 0].shape  # uses DXHXW
        image_range = []
        for idx, dim_range in enumerate(
                shape):  # identifies the presence of label voxels across each dimension(from the beginning and from the end)
            # essentially iterates over the available dimensions to identify
            # presence
            output = np.rollaxis(image[0, 0], idx)
            start_index = None
            stop_index = None
            for index in range(dim_range):
                if start_index is None and output[index, :, :].sum(
                ) >= 10:  # from the beginning
                    start_index = index  # store identified start index having label voxels
                if stop_index is None and output[(
                        dim_range - 1) - index, :, :].sum() >= 10:  # from the end
                    stop_index = (dim_range - 1) - index  # store end index
            assert start_index is not None and stop_index is not None and stop_index > start_index, 'Generated improper indices. Check inputs'
            image_range.append((start_index, stop_index))
        gen_set_ranges.append((image_paths, image_range))

    max_height = 0
    max_depth = 0
    max_width = 0

    # Calculate the max patch size based on the above identified ranges across all images. Use specified pad to ensure buffer around calculated
    # patch size. Calculated patches are scaled back to original dimensions using specified scale factor.
    # Also calculate unscaled centre coordinates to roughly identify centres
    # of the LA structure to then extract slice ranges later.
    gen_set_centres = []
    for _, data in enumerate(gen_set_ranges):
        image_paths = data[0]
        depth_range, height_range, width_range = data[1]

        depth = round((depth_range[1] - depth_range[0]) / scale_factor[0])
        height = round((height_range[1] - height_range[0]) / scale_factor[1])
        width = round((width_range[1] - width_range[0]) / scale_factor[2])

        max_depth = depth if depth > max_depth else max_depth
        max_height = height if height > max_height else max_height
        max_width = width if width > max_width else max_width

        # calculate the unscaled centre of the structure
        unscaled_centre = (
            round(
                depth_range[0] /
                scale_factor[0]) +
            round(
                depth /
                2),
            round(
                height_range[0] /
                scale_factor[1]) +
            round(
                height /
                2),
            round(
                width_range[0] /
                scale_factor[2]) +
            round(
                width /
                2))
        gen_set_centres.append((image_paths, unscaled_centre))

    max_depth = max_depth + pad[0] if max_depth + pad[0] <= 96 else 96
    max_height = max_height + pad[1] if max_height + pad[1] <= 640 else 640
    max_width = max_width + pad[2] if max_width + pad[2] <= 640 else 640

    # if provided (during testing and validation) use that instead.
    patch_dimension = patch_size if patch_size is not None else [
        max_depth, max_height, max_width]

    # Modify patch dimensions so as to be suitable with the segmentation
    # model(downsampling across the model)
    for idx, value in enumerate(patch_dimension):
        for _ in range(1, 16):
            if value % 16 == 0:
                break
            else:
                value += 1
        patch_dimension[idx] = value

    image_slices = []
    patch_d = patch_dimension[0] / 2
    patch_h = patch_dimension[1] / 2
    patch_w = patch_dimension[2] / 2

    # calculate the unscaled slice ranges of the centre based on the
    # calculated patch size and LA structure centre
    for data in gen_set_centres:
        paths, centre = data

        # depth slice ranges
        start_depth = centre[0] - patch_d if centre[0] - patch_d > 0 else 0
        end_depth = centre[0] + patch_d if centre[0] + patch_d < 96 else 96

        assert end_depth - start_depth <= patch_dimension[0]
        if end_depth - start_depth != patch_dimension[0]:
            start_depth = 0 if start_depth == 0 else start_depth - \
                (patch_dimension[0] - (end_depth - start_depth))
            end_depth = 96 if end_depth == 96 else end_depth + \
                (patch_dimension[0] - (end_depth - start_depth))
        assert start_depth >= 0 and end_depth <= 96

        # height slice ranges
        start_height = centre[1] - patch_h if centre[1] - patch_h > 0 else 0
        end_height = centre[1] + patch_h if centre[1] + patch_h < 640 else 640

        assert end_height - start_height <= patch_dimension[1]
        if end_height - start_height != patch_dimension[1]:
            start_height = 0 if start_height == 0 else start_height - \
                (patch_dimension[1] - (end_height - start_height))
            end_height = 640 if end_height == 640 else end_height + \
                (patch_dimension[1] - (end_height - start_height))
        assert start_height >= 0 and end_height <= 640

        # width slice ranges
        start_width = centre[2] - patch_w if centre[2] - patch_w > 0 else 0
        end_width = centre[2] + patch_w if centre[2] + patch_w < 640 else 640

        assert end_width - start_width <= patch_dimension[2]
        if end_width - start_width != patch_dimension[2]:
            start_width = 0 if start_width == 0 else start_width - \
                (patch_dimension[2] - (end_width - start_width))
            end_width = 640 if end_width == 640 else end_width + \
                (patch_dimension[2] - (end_width - start_width))
        assert start_width >= 0 and end_width <= 640

        image_slice = (slice(int(start_depth), int(end_depth), None),
                       slice(int(start_height), int(end_height), None),
                       slice(int(start_width), int(end_width), None))

        image_slices.append((paths, image_slice))

    return patch_dimension, image_slices


def distance_map(labels):
    labels = labels.numpy().astype(np.int16)
    assert set(np.unique(labels)).issubset(
        [0, 1]), 'Groundtruth labels must only have values 0 or 1'
    result = np.zeros_like(labels)  # container to fill in distance values
    for x in range(len(labels)):
        posmask = labels[x].astype(np.bool)
        negmask = ~posmask
        # Level set representation
        result[x] = distance(negmask) * negmask - \
            (distance(posmask) - 1) * posmask

    return torch.Tensor(result).to(dtype=torch.int16)


class Transformer(object):
    def __init__(self, image, image_type, mode='validate', seed=123, **kwargs):
        self.image = image
        self.mode = mode
        self.random_state = np.random.RandomState(seed)
        self.mean = kwargs.get('mean') if kwargs.get('mean') is not None else 0
        self.std = kwargs.get('std') if kwargs.get('std') is not None else 1

        normalise = Normalise(
            mean=self.mean,
            std=self.std,
            image_type=image_type)
        horizontal_flip = HorizontalFlip(
            random_state=self.random_state,
            execution_probability=1.0)
        gaussian_noise = GaussianNoise(
            random_state=self.random_state,
            image_type=image_type,
            execution_probability=1.0)
        rand_rotate = RotateImage(
            random_state=self.random_state,
            image_type=image_type,
            execution_probability=1.0)
        rand_translate = TranslateImage(
            random_state=self.random_state,
            image_type=image_type,
            execution_probability=1.0)
        to_tensor = TorchTensor(image_type=image_type)

        if self.mode == 'train':
            self.transform0 = Compose([normalise, to_tensor])
            self.h_flip = Compose([normalise, horizontal_flip, to_tensor])
            self.g_noise = Compose([normalise, gaussian_noise, to_tensor])
            #self.e_defo = Compose([normalise, elastic_deformation, to_tensor])

        elif self.mode == 'random_rotate':
            self.random = Compose([normalise, rand_rotate, to_tensor])

        elif self.mode == 'random_translate':
            self.random = Compose([normalise, rand_translate, to_tensor])

        elif self.mode == 'random_deformation':
            #self.random = Compose([normalise, elastic_deformation, to_tensor])
            pass

        else:
            self.transform = Compose([normalise, to_tensor])

    def __call__(self):
        if self.mode == 'train':
            transformed_images = []
            transformed_images.extend(
                (self.transform0(
                    self.image), self.h_flip(
                    self.image), self.g_noise(
                    self.image)))  # , self.e_defo(self.image)))

            return transformed_images

        elif self.mode in ['random_rotate', 'random_translate', 'random_deformation']:

            # no list returned when random_rotate or random_translate mode
            return self.random(self.image)

        else:

            return [self.transform(self.image)]
