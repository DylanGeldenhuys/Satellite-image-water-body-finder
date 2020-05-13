import pandas as pd
import numpy as np

import cv2 as cv

from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from .utilities import rgb_to_grey


def extract_features(image_data, padding=20):
    """Extracts features from image.

    Parameters
    ----------
    image_data: ndarray
        2D image numpy array.

    padding: int
        Padding around image. Required for entropy and average colour calculations.
        Final extracted features will have padding removed from edges.
    Returns
    -------
    out: DataFrame
        Pandas dataframe of features.
    """
    features = pd.DataFrame()

    # color feature
    features['color_r'] = image_data[padding:-
                                     padding, padding:-
                                     padding, 0].flatten()
    features['color_g'] = image_data[padding:-
                                     padding, padding:-
                                     padding, 1].flatten()
    features['color_b'] = image_data[padding:-
                                     padding, padding:-
                                     padding, 2].flatten()

    # blur feature
    blur_feature = cv.GaussianBlur(image_data, (39, 39), 0)
    features['color_r_blur'] = blur_feature[padding:-
                                            padding, padding:-
                                            padding, 0].flatten()
    features['color_g_blur'] = blur_feature[padding:-
                                            padding, padding:-
                                            padding, 1].flatten()
    features['color_b_blur'] = blur_feature[padding:-
                                            padding, padding:-
                                            padding, 2].flatten()

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
    features['entropy_small'] = entropy_feature_small[padding:-
                                                      padding, padding:-
                                                      padding].flatten()

    # entropy large disk
    entropy_feature_large = entropy(
        img_as_ubyte(rgb_to_grey(image_data)), disk(15))
    features['entropy_large'] = entropy_feature_large[padding:-
                                                      padding, padding:-
                                                      padding].flatten()

    return features
