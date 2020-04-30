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


def get_key(point):
    return "[{0}, {1}]".format(point[0], point[1])


def rgb2gray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.array((0.299*r + 0.587*g + 0.114*b), dtype=np.uint8)


def create_training_set(image_index, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory, image_label_fix_data):
    padding = 5

    filename = os.listdir(image_data_directory)[
        image_index].replace(".tif", "")

    # load labels
    label_data = np.load(label_data_directory.joinpath(
        "{}.npy".format(filename)))

    for area in image_label_fix_data[image_index]:
        # create initial dataframe
        training_set = {
            'color_r': [],
            'color_g': [],
            'color_b': [],
            'entropy': [],
            'label': []
        }

        width = area[1][0] - area[0][0]
        height = area[1][1] - area[0][1]

        # load image window
        raster_image_data = rasterio.open(
            image_data_directory.joinpath("{}.tif".format(filename))).read(window=Window(area[0][0] - padding, area[0][1] - padding, width + padding, height + padding))
        image_data = reshape_as_image(raster_image_data)

        # save for visualisation purposes
        Image.fromarray(image_data.astype(np.uint8)).save(visualisation_output_directory.joinpath(
            "{0}_{1}.jpg".format(filename, str(area))))

        # entropy feature extraction (red channel)
        entropy_feature = entropy(img_as_ubyte(rgb2gray(image_data)), disk(5))

        # save for visualisation purposes
        Image.fromarray(entropy_feature * (255 / max(map(max, entropy_feature)))).convert('RGB').save(visualisation_output_directory.joinpath(
            "{0}_{1}_entropy.jpg".format(filename, str(area))))

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
            "{0}_{1}.csv".format(filename, str(area))))

    return filename
