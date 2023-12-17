from abc import ABC, abstractmethod
import warnings
import numpy as np
import random
from random import randint
from scipy import ndimage
from sunpy.map import Map
from astropy import units as u
from astropy.coordinates import SkyCoord


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


class NormalizeExposureEditor(Editor):
    """
    Normalizes the exposure time of a sunpy map to a target value.

    Parameters
    ----------
    target : `astropy.units.Quantity`
        The target exposure time.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.NormalizeExposureEditor(target=1 * u.s)
    >>> s_map = editor(s_map)


    """

    def __init__(self, target=1 * u.s):
        self.target = target
        super().__init__()

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")
        data = s_map.data
        data = data / \
            s_map.exposure_time.to(u.s).value * self.target.to(u.s).value
        return Map(data.astype(np.float32), s_map.meta)


class ScaleEditor(Editor):
    """

    Scales a sunpy map to a target arcseconds per pixel value.

    Parameters
    ----------
    arcspp : `float`
        The target arcseconds per pixel value.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.ScaleEditor(arcspp=0.6)
    >>> s_map = editor(s_map)

    """

    def __init__(self, arcspp):
        self.arcspp = arcspp
        super(ScaleEditor, self).__init__()

    def call(self, s_map, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # ignore warnings
            scale_factor = s_map.scale[0].value / self.arcspp
            new_dimensions = [int(s_map.data.shape[1] * scale_factor),
                              int(s_map.data.shape[0] * scale_factor)] * u.pixel
            s_map = s_map.resample(new_dimensions)

            return Map(s_map.data.astype(np.float32), s_map.meta)


class CropEditor(Editor):
    """

    Crops a sunpy map to specified dimensions.

    Parameters
    ----------
    start_x : `int`
        The starting x coordinate of the crop.
    end_x : `int`
        The ending x coordinate of the crop.
    start_y : `int`
        The starting y coordinate of the crop.
    end_y : `int`
        The ending y coordinate of the crop.

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.CropEditor(start_x=0, end_x=512, start_y=0, end_y=512)
    >>> s_map = editor(s_map)

    """

    def __init__(self, start_x, end_x, start_y, end_y):
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y

    def call(self, data, **kwargs):

        crop = data[self.start_x: self.end_x, self.start_y:self.end_y]
        return crop


class ShiftMeanEditor(Editor):
    """

    Normalizes the data, by setting the mean to 0, and clipping the data to [-1, 1].

    Usage:

    >>> import helio_tools as ht
    >>> editor = ht.ShiftMeanEditor()
    >>> data = ht.load_fits_to_ndarray(filename)
    >>> data = editor(data)

    """

    def call(self, data, **kwargs):
        mean = np.mean(data)
        data = (data - mean)
        data = np.clip(data, -1, 1)
        return data


class NormalizeRadiusEditor(Editor):
    def __init__(self, resolution, padding_factor=0.1, crop=True, **kwargs):
        self.padding_factor = padding_factor
        self.resolution = resolution
        self.crop = crop
        super(NormalizeRadiusEditor, self).__init__(**kwargs)

    def call(self, s_map, **kwargs):
        warnings.simplefilter("ignore")  # ignore warnings
        r_obs_pix = s_map.rsun_obs / s_map.scale[0]  # normalize solar radius
        r_obs_pix = (1 + self.padding_factor) * r_obs_pix
        scale_factor = self.resolution / (2 * r_obs_pix.value)
        s_map = Map(np.nan_to_num(s_map.data).astype(np.float32), s_map.meta)
        s_map = s_map.rotate(
            recenter=True, scale=scale_factor, missing=0, order=4)
        if self.crop:
            arcs_frame = (self.resolution / 2) * s_map.scale[0].value
            s_map = s_map.submap(bottom_left=SkyCoord(-arcs_frame * u.arcsec, -arcs_frame * u.arcsec, frame=s_map.coordinate_frame),
                                 top_right=SkyCoord(arcs_frame * u.arcsec, arcs_frame * u.arcsec, frame=s_map.coordinate_frame))
            pad_x = s_map.data.shape[0] - self.resolution
            pad_y = s_map.data.shape[1] - self.resolution
            s_map = s_map.submap(bottom_left=[pad_x // 2, pad_y // 2] * u.pix,
                                 top_right=[pad_x // 2 + self.resolution - 1, pad_y // 2 + self.resolution - 1] * u.pix)
        s_map.meta['r_sun'] = s_map.rsun_obs.value / s_map.meta['cdelt1']
        return s_map
