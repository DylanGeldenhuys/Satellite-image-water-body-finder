import pickle
import numpy as np


def find_waterbody_mask(image_data, data_resolution, data_padding):
    height = int(
        ((image_data.shape[0]) / data_resolution) - data_padding / data_resolution)
    width = int(
        ((image_data.shape[1]) / data_resolution) - data_padding / data_resolution)

    offset = round(data_padding/2)

    output = np.zeros(height, width)
    features = {
        'variance_a': [],
        'variance_b': []
    }

    for j in range(height):
        for i in range(width):
            y = j * data_resolution + offset
            x = i * data_resolution + offset

            # extract features

    print("found")
