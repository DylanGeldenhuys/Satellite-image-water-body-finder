import pickle
import numpy as np
import pandas as pd
from water_body_finder.feature_extraction import extract_variance
from water_body_finder.utilities import create_window


def extract_features(image_data, data_resolution, data_padding):
    height = int(
        image_data.shape[0] / data_resolution - data_padding / data_resolution)
    width = int(
        image_data.shape[1] / data_resolution - data_padding / data_resolution)

    offset = int(data_padding/2)

    features = {
        'variance_a': [],
        'variance_b': []
    }

    for j in range(height):
        for i in range(width):
            y = j * data_resolution + offset
            x = i * data_resolution + offset

            # extract features
            window_a = create_window(image_data, [y, x], 5)
            window_b = create_window(image_data, [y, x], 15)

            features['variance_a'].append(extract_variance(window_a))
            features['variance_b'].append(extract_variance(window_b))

    return(pd.DataFrame(features))


def find_waterbody_mask(image_data, rfc, data_resolution, data_padding):
    height = int(
        image_data.shape[0] / data_resolution - data_padding / data_resolution)
    width = int(
        image_data.shape[1] / data_resolution - data_padding / data_resolution)

    offset = int(data_padding/2)

    features = {
        'variance_a': [],
        'variance_b': []
    }

    for j in range(height):
        for i in range(width):
            y = j * data_resolution + offset
            x = i * data_resolution + offset

            # extract features
            window_a = create_window(image_data, [y, x], 5)
            window_b = create_window(image_data, [y, x], 15)

            features['variance_a'].append(extract_variance(window_a))
            features['variance_b'].append(extract_variance(window_b))

    feature_df = pd.DataFrame(features)
    prediction = rfc.predict(feature_df)
    result = np.reshape(prediction, [height, width])

    return(result)
