import pandas as pd
import numpy as np

import cv2 as cv

from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from .utilities import rgb_to_grey


def extract_features(image_data, data_resolution, padding=0, version="0"):
    """Extracts features from image.

    Parameters
    ----------
    image_data: ndarray
        2D image numpy array.
    data_resolution: int
        Steps taken over data when extracting features.
    padding: int
        Padding taken around data. Left out of result.
    version: string
        Version of feature extraction to use.

    Returns
    -------
    out: DataFrame
        Pandas dataframe of features.
    """

    return locals()["version_{}".format(version)](image_data, data_resolution, padding)


def version_0(image_data, data_resolution, padding=0):
    """Extracts features from image.

    Parameters
    ----------
    image_data: ndarray
        2D image numpy array.

    Returns
    -------
    out: DataFrame
        Pandas dataframe of features.
    """
    features = pd.DataFrame()
    padding_height = -padding if padding != 0 else image_data.shape[0]
    padding_width = -padding if padding != 0 else image_data.shape[1]

    # color feature
    features['color_r'] = image_data[padding:padding_height:data_resolution,
                                     padding:padding_width: data_resolution, 0].flatten()
    features['color_g'] = image_data[padding:padding_height:data_resolution,
                                     padding:padding_width:data_resolution, 1].flatten()
    features['color_b'] = image_data[padding:padding_height:data_resolution,
                                     padding:padding_width:data_resolution, 2].flatten()

    # blur feature
    blur_feature = cv.GaussianBlur(image_data, (39, 39), 0)
    features['color_r_blur'] = blur_feature[padding:padding_height:data_resolution,
                                            padding:padding_width:data_resolution, 0].flatten()
    features['color_g_blur'] = blur_feature[padding:padding_height:data_resolution,
                                            padding:padding_width:data_resolution, 1].flatten()
    features['color_b_blur'] = blur_feature[padding:padding_height:data_resolution,
                                            padding:padding_width:data_resolution, 2].flatten()

    # average color feature
    average_color_feature = np.average(np.average(
        image_data, axis=0), axis=0)
    features['average_color_r'] = int(
        round(average_color_feature[0]))
    features['average_color_g'] = int(
        round(average_color_feature[1]))
    features['average_color_b'] = int(
        round(average_color_feature[2]))

    # entropy small disk
    entropy_feature_small = entropy(
        img_as_ubyte(rgb_to_grey(image_data)), disk(5))
    features['entropy_small'] = entropy_feature_small[padding:padding_height:data_resolution,
                                                      padding:padding_width:data_resolution].flatten()

    # entropy large disk
    entropy_feature_large = entropy(
        img_as_ubyte(rgb_to_grey(image_data)), disk(15))
    features['entropy_large'] = entropy_feature_large[padding:padding_height:data_resolution,
                                                      padding:padding_width:data_resolution].flatten()

    return features
