from abc import ABC, abstractmethod
import warnings
import numpy as np
import random
from random import randint
from scipy import ndimage
from sunpy.map import Map


class Editor(ABC):
    """
    Abstract base class for all editors.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.Editor()
    >>> data = ht.load_fits_to_ndarray(filename)
    >>> data = editor.convert(data)


    """

    def convert(self, data, **kwargs):
        result = self.call(data, **kwargs)
        if isinstance(result, tuple):
            data, add_kwargs = result
            kwargs.update(add_kwargs)
        else:
            data = result
        return data, kwargs

    @abstractmethod
    def call(self, data, **kwargs):
        raise NotImplementedError()


class MinMaxQuantileNormalizeEditor(Editor):
    """
    Normalizes the data by subtracting the minimum and dividing by the maximum, according to the 0.001 and 0.999 quantiles.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.MinMaxQuantileNormalizeEditor()
    >>> data = ht.load_fits_to_ndarray(filename)
    >>> data = editor.convert(data)
    """

    def call(self, data, **kwargs):

        vmin = np.quantile(data, 0.001)
        vmax = np.quantile(data, 0.999)

        data = (data - vmin) / (vmax - vmin) * 2 - 1
        data = np.clip(data, -1, 1)
        return data


class StretchPixelEditor(Editor):
    """
    Normalizes the data by subtracting the minimum and dividing by the maximum.
    """

    def call(self, data, **kwargs):

        vmin = np.min(data)
        vmax = np.max(data)

        data = (data - vmin) / (vmax - vmin) * 2 - 1
        return data


class WhiteningEditor(Editor):
    """ 
    Mean value is set to 0 (remove contrast) and std is set to 1.
    """

    def call(self, data, **kwargs):
        data_mean = np.nanmean(data)
        data_std = np.nanstd(data)
        data = (data - data_mean) / (data_std + 1e-6)
        return data


class BrightestPixelPatchEditor(Editor):
    """
    Extracts a patch around the brightest pixel. If random_selection is set to True, a random patch is selected with a probability of random_selection.
    Otherwise, the brightest pixel is selected and a patch is extracted around it.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.BrightestPixelPatchEditor(patch_shape=(256, 256), random_selection=0.2)
    >>> data = ht.load_fits_to_ndarray(filename)
    >>> data = editor(data)

    """

    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(
            data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(
            data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0],
                         y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == np.nanmax(smoothed))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

            x = pixel_pos[0]
            y = pixel_pos[1]
            patch = data[:,
                         x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                         y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(
            patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class DarkestPixelPatchEditor(Editor):
    """
    Extracts a patch around the darkest pixel. If random_selection is set to True, a random patch is selected with a probability of random_selection.
    Otherwise, the darkest pixel is selected (after first applying a gaussian filter) and a patch is extracted around it.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.DarkestPixelPatchEditor(patch_shape=(256, 256), random_selection=0.2)
    >>> data = ht.load_fits_to_ndarray(filename)
    >>> data = editor(data)

    """

    def __init__(self, patch_shape, idx=0, random_selection=0.2):
        self.patch_shape = patch_shape
        self.idx = idx
        self.random_selection = random_selection

    def call(self, data, **kwargs):
        assert data.shape[1] >= self.patch_shape[0], 'Invalid data shape: %s' % str(
            data.shape)
        assert data.shape[2] >= self.patch_shape[1], 'Invalid data shape: %s' % str(
            data.shape)

        if random.random() <= self.random_selection:
            x = randint(0, data.shape[1] - self.patch_shape[0])
            y = randint(0, data.shape[2] - self.patch_shape[1])
            patch = data[:, x:x + self.patch_shape[0],
                         y:y + self.patch_shape[1]]
        else:
            smoothed = ndimage.gaussian_filter(data[self.idx], sigma=5)
            pixel_pos = np.argwhere(smoothed == (np.nanmin(smoothed)))
            pixel_pos = pixel_pos[randint(0, len(pixel_pos) - 1)]
            pixel_pos = np.min([pixel_pos[0], smoothed.shape[0] - self.patch_shape[0] // 2]), np.min(
                [pixel_pos[1], smoothed.shape[1] - self.patch_shape[1] // 2])
            pixel_pos = np.max([pixel_pos[0], self.patch_shape[0] // 2]), np.max(
                [pixel_pos[1], self.patch_shape[1] // 2])

            x = pixel_pos[0]
            y = pixel_pos[1]
            patch = data[:,
                         x - int(np.floor(self.patch_shape[0] / 2)):x + int(np.ceil(self.patch_shape[0] / 2)),
                         y - int(np.floor(self.patch_shape[1] / 2)):y + int(np.ceil(self.patch_shape[1] / 2)), ]
        assert np.std(
            patch) != 0, 'Invalid patch found (all values %f)' % np.mean(patch)
        return patch


class LoadMapEditor(Editor):
    """
    Loads a fits file into a sunpy map.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.LoadMapEditor()
    >>> s_map = editor(filename)

    """

    def call(self, data, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_map = Map(data)
            s_map.meta['timesys'] = 'tai'  # fix leap seconds
            return s_map, {'path': data}
