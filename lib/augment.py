import numpy as np
import scipy

from typing import Tuple
from numpy import ndarray
from scipy.ndimage import rotate

import torch


class Normalise(object):
    """
    Apply Z-score normalization to a given input array based on specified mean and std values. If the provided image is a label no normalisation
    is performed.
    """

    def __init__(self, mean: float, std: float, eps: float = 1e-6, **kwargs):
        self.mean = mean
        self.std = std
        self.eps = eps
        self.image_type = 'label' if kwargs.get(
            'image_type') == 'label' else None

    def __call__(self, input_array: ndarray):
        if self.image_type != 'label':
            return (input_array - self.mean) / \
                np.clip(self.std, a_min=self.eps, a_max=None)

        return input_array


class HorizontalFlip(object):
    """
    Flips the given input array along the height axis.
    """

    def __init__(self, random_state, execution_probability: float = 0.4):
        self.random_state = random_state
        self.execution_probability = execution_probability

    def __call__(self, input_array: ndarray):
        if self.random_state.uniform() <= self.execution_probability:
            flipped_array = np.flip(input_array, axis=1)
            return flipped_array

        return input_array


class RotateImage(object):
    """
    Rotates the given input array along a given axis using a rotation angle chosen at random from a range of -angle to +angle.
    """

    def __init__(
            self,
            random_state,
            angle: int = 3,
            axes: Tuple = None,
            mode: str = 'constant',
            order: int = 0,
            execution_probability: float = 0.4,
            **kwargs):
        if axes is None:
            axes = [(2, 1)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle = angle
        self.axes = axes
        self.mode = mode
        self.order = 0 if kwargs.get('image_type') == 'label' else order
        self.execution_probability = execution_probability

    def __call__(self, input_array: ndarray):
        if self.random_state.uniform() <= self.execution_probability:
            axis = self.axes[self.random_state.randint(len(self.axes))]
            angle = self.random_state.randint(-self.angle, self.angle)
            rotated_array = rotate(
                input_array,
                angle,
                axes=axis,
                reshape=False,
                order=self.order,
                mode=self.mode)
            return rotated_array

        return input_array


class TranslateImage(object):
    """
    The provided input array is translated by an amount specified for each dimesnion in the shift parameter.
    """

    def __init__(
            self,
            random_state,
            shift: Tuple = None,
            mode: str = 'constant',
            order: int = 0,
            execution_probability: float = 0.4,
            **kwargs):
        if shift is None:
            shift = [(0, 5, 0), (0, 0, 5), (0, -5, 0), (0, 0, -5),
                     (0, 5, 5), (0, -5, -5), (0, 5, -5), (0, -5, 5)]
        else:
            assert isinstance(shift, tuple) and len(shift) > 0

        self.random_state = random_state
        self.shift = shift
        self.mode = mode
        self.order = 0 if kwargs.get('image_type') == 'label' else order
        self.execution_probability = execution_probability

    def __call__(self, input_array: ndarray):
        if self.random_state.uniform() <= self.execution_probability:
            translation = self.shift[self.random_state.randint(
                len(self.shift))]
            translated_array = scipy.ndimage.shift(
                input_array, shift=translation, order=self.order, mode=self.mode)
            return translated_array

        return input_array


class GaussianNoise(object):
    def __init__(
            self,
            random_state,
            scale: Tuple = (
                0.0,
                1.0),
            execution_probability: float = 0.4,
            **kwargs):
        self.random_state = random_state
        self.scale = scale
        self.kwargs = 'label' if kwargs.get('image_type') == 'label' else None
        self.execution_probability = execution_probability

    def __call__(self, input_array: ndarray):
        if self.random_state.uniform() <= self.execution_probability:
            if self.kwargs != 'label':
                std = self.random_state.uniform(self.scale[0], self.scale[1])
                gaussian_noise = self.random_state.normal(
                    0, std, size=input_array.shape)
                return input_array + gaussian_noise
            else:
                return input_array

        return input_array


class TorchTensor(object):
    """
    Adds additional 'channel' axis to the input and converts the given input numpy.ndarray into torch.Tensor.
    """

    def __init__(self, **kwargs):
        self.dtype = torch.uint8 if kwargs.get(
            'image_type') == 'label' else torch.float32

    def __call__(self, input_array: ndarray):
        input_array = np.expand_dims(input_array, axis=0)

        return torch.Tensor(
            input_array.astype(
                np.float32)).to(
            dtype=self.dtype)
