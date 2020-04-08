import pickle
import numpy as np
import pandas as pd
from .feature_extraction import extract_entropy, extract_mean_color
from .utilities import create_window


def calc_img_dimensions(image_data, data_resolution, data_padding):
    height = int(
        image_data.shape[0] / data_resolution - data_padding / data_resolution)
    width = int(
        image_data.shape[1] / data_resolution - data_padding / data_resolution)

    return (width, height)


def extract_features_v1(image_data):
    data_resolution = 5
    data_padding = 15

    (width, height) = calc_img_dimensions(
        image_data, data_resolution, data_padding)

    offset = int(data_padding/2)

    features = {
        'color_r': [],
        'color_g': [],
        'color_b': [],
        'entropy_a': [],
    }

    for j in range(height):
        for i in range(width):
            y = j * data_resolution + offset
            x = i * data_resolution + offset

            # extract features
            window_a = create_window(image_data, [y, x], 5)

            mean_color = extract_mean_color(window_a)

            features['color_r'].append(mean_color[0])
            features['color_g'].append(mean_color[1])
            features['color_b'].append(mean_color[2])
            features['entropy_a'].append(extract_entropy(window_a))

    return (pd.DataFrame(features), height, width)


def extract_features_v3(image_data):
    data_resolution = 3
    data_padding = 9

    (width, height) = calc_img_dimensions(
        image_data, data_resolution, data_padding)

    offset = int(data_padding/2)

    features = {
        'color_r': [],
        'color_g': [],
        'color_b': [],
        'entropy_a': [],
        'entropy_b': []
    }

    for j in range(height):
        for i in range(width):
            y = j * data_resolution + offset
            x = i * data_resolution + offset

            # extract features
            window_a = create_window(image_data, [y, x], 3)
            window_b = create_window(image_data, [y, x], 9)

            mean_color = extract_mean_color(window_a)

            features['color_r'].append(mean_color[0])
            features['color_g'].append(mean_color[1])
            features['color_b'].append(mean_color[2])
            features['entropy_a'].append(extract_entropy(window_a))
            features['entropy_b'].append(extract_entropy(window_b))

    return (pd.DataFrame(features), height, width)


def find_waterbody_mask(image_data, rfc, feature_extraction_func):
    feature_df, height, width = feature_extraction_func(
        image_data)
    prediction = rfc.predict(feature_df)
    result = np.reshape(prediction, [height, width])

    return(result)
