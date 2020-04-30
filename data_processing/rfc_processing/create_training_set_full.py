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
from rasterio.enums import Resampling

from skimage.util import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.morphology import disk

from water_body_finder.feature_extraction.glcm_texture import glcm_feature


def rgb2gray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.array((0.299*r + 0.587*g + 0.114*b), dtype=np.uint8)


def create_training_set(filename, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory):
    padding = 50
    upscale_factor = 1/3
    window_size = 9

    training_set = {
        'color_r': [],
        'color_g': [],
        'color_b': [],
        'entropy': [],
        # 'texture_a': [],
        # 'texture_b': [],
        'color_mean_r': [],
        'color_mean_g': [],
        'color_mean_b': [],
        'label': []
    }

    # load labels
    label_data = np.load(label_data_directory.joinpath(
        "{}.npy".format(filename)))

    print(label_data.shape)

    # load image
    dataset = rasterio.open(
        image_data_directory.joinpath("{}.tif".format(filename)))
    rasterio_image_data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * upscale_factor),
            int(dataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )
    del(dataset)

    image_data = reshape_as_image(rasterio_image_data)
    del(rasterio_image_data)

    height, width, channels = image_data.shape
    del(channels)

    # save for visualisation purposes
    Image.fromarray(image_data.astype(np.uint8)).save(visualisation_output_directory.joinpath(
        "{}.jpg".format(filename)))

    # entropy feature extraction (red channel)
    entropy_feature = entropy(img_as_ubyte(rgb2gray(image_data)), disk(5))

    # save for visualisation purposes
    Image.fromarray(entropy_feature * 25).convert('RGB').save(visualisation_output_directory.joinpath(
        "{}_entropy.jpg".format(filename)))

    # append features to dataframe
    for y in range(padding, height - padding):
        print("{0}: {1}%".format(filename, y / (height - padding) * 100))
        for x in range(padding, width - padding):

            color_features = image_data[y][x]

            # colour features
            training_set['color_r'].append(color_features[0])
            training_set['color_g'].append(color_features[1])
            training_set['color_b'].append(color_features[2])

            # entropy feature
            training_set['entropy'].append(entropy_feature[y][x])

            offset = int(window_size/2)
            window = image_data[y-offset: y + offset,
                                x-offset: x + offset]

           # texture_a, texture_b = glcm_feature(window[:, :, 0])
            # training_set['texture_a'].append(texture_a)
           # training_set['texture_b'].append(texture_b)

            mean_colour = window.mean(axis=0).mean(axis=0)
            training_set['color_mean_r'].append(mean_colour[0])
            training_set['color_mean_g'].append(mean_colour[1])
            training_set['color_mean_b'].append(mean_colour[2])

            # label
            training_set['label'].append(
                label_data[y, x])

    pd.DataFrame(training_set).to_csv(training_output_directory.joinpath(
        "{}.csv".format(filename)))

    print(filename)
    return filename
