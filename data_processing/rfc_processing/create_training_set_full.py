import sys
sys.path.append('C:/personal/satalite-image-water-body-finder')  # noqa

import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from PIL import Image

import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window

from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk


def rgb2gray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.array((0.299*r + 0.587*g + 0.114*b), dtype=np.uint8)


def create_training_set(filename, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory):
    padding = 50

    training_set = {
        'color_r': [],
        'color_g': [],
        'color_b': [],
        'entropy': [],
        'label': []
    }

    # load labels
    label_data = np.load(label_data_directory.joinpath(
        "{}.npy".format(filename)))

    # load image
    raster_image_data = rasterio.open(
        image_data_directory.joinpath("{}.tif".format(filename))).read()
    image_data = reshape_as_image(raster_image_data)

    width, height = image_data.shape

    # save for visualisation purposes
    Image.fromarray(image_data.astype(np.uint8)).save(visualisation_output_directory.joinpath(
        "{}.jpg".format(filename)))

    # entropy feature extraction (red channel)
    entropy_feature = entropy(img_as_ubyte(rgb2gray(image_data)), disk(5))

    # save for visualisation purposes
    Image.fromarray(entropy_feature * (255 / max(map(max, entropy_feature)))).convert('RGB').save(visualisation_output_directory.joinpath(
        "{}_entropy.jpg".format(filename)))

    # append features to dataframe
    for y in range(padding, padding + height):
        for x in range(padding, padding + width):

            color_features = image_data[y][x]

            # colour features
            training_set['color_r'].append(color_features[0])
            training_set['color_g'].append(color_features[1])
            training_set['color_b'].append(color_features[2])

            # entropy feature
            training_set['entropy'].append(entropy_feature[y][x])

            # label
            training_set['label'].append(
                label_data[(y + height - padding), (x + width - padding)])

    pd.DataFrame(training_set).to_csv(training_output_directory.joinpath(
        "{}.csv".format(filename)))

    return filename
