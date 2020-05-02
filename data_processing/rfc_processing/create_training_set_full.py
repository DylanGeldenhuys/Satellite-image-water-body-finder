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

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from water_body_finder.feature_extraction.glcm_texture import glcm_feature


def rgb2gray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.array((0.299*r + 0.587*g + 0.114*b), dtype=np.uint8)


def create_training_set(filename, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory):
    try:
        outer_padding = 50
        inner_padding = 10
        max_set_size = 6000
        window_size = 9
        data_resolution = 3

        # load labels
        label_data = np.load(label_data_directory.joinpath(
            "{}.npy".format(filename)))

        # image dataset
        dataset = rasterio.open(
            image_data_directory.joinpath("{}.tif".format(filename)))

        for y_origin in range(outer_padding, dataset.height - outer_padding, max_set_size):
            window_height = max_set_size
            if (y_origin + max_set_size) > (dataset.height - outer_padding):
                window_height = dataset.height - outer_padding - y_origin
            for x_origin in range(outer_padding, dataset.width - outer_padding, max_set_size):
                window_width = max_set_size
                if (x_origin + max_set_size) > (dataset.width - outer_padding):
                    window_width = dataset.width - outer_padding - x_origin

                # load image window
                rasterio_image_data = dataset.read(
                    window=Window(x_origin - inner_padding, y_origin - inner_padding,
                                  window_width + inner_padding * 2, window_height + inner_padding * 2)
                )
                image_data = reshape_as_image(rasterio_image_data)

                training_set = {
                    'color_r': [],
                    'color_g': [],
                    'color_b': [],
                    'entropy': [],
                    'color_mean_r': [],
                    'color_mean_g': [],
                    'color_mean_b': [],
                    'label': []
                }

                height, width, channels = image_data.shape

                # save for visualisation purposes
                Image.fromarray(image_data.astype(np.uint8)).save(visualisation_output_directory.joinpath(
                    "{0}_{1}_{2}.jpg".format(filename, y_origin, x_origin)))

                # entropy feature extraction (red channel)
                entropy_feature = entropy(
                    img_as_ubyte(rgb2gray(image_data)), disk(5))

                # save for visualisation purposes
                Image.fromarray(entropy_feature * 25).convert('RGB').save(visualisation_output_directory.joinpath(
                    "{0}_{1}_{2}_entropy.jpg".format(filename, y_origin, x_origin)))

                # save label visualisation
                plt.imsave(visualisation_output_directory.joinpath("{0}_{1}_{2}_label.jpg".format(
                    filename, y_origin, x_origin)), label_data[y_origin-inner_padding:y_origin-inner_padding+height, x_origin-inner_padding:x_origin-inner_padding+width], cmap=cm.gray)

                # append features to dataframe
                for y in range(inner_padding, height - inner_padding, data_resolution):
                    for x in range(inner_padding, width - inner_padding, data_resolution):

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

                        mean_colour = window.mean(axis=0).mean(axis=0)
                        training_set['color_mean_r'].append(
                            int(round(mean_colour[0])))
                        training_set['color_mean_g'].append(
                            int(round(mean_colour[1])))
                        training_set['color_mean_b'].append(
                            int(round(mean_colour[2])))

                        # label
                        training_set['label'].append(
                            label_data[y + y_origin - inner_padding, x + x_origin - inner_padding])

                pd.DataFrame(training_set).to_csv(training_output_directory.joinpath(
                    "{0}_{1}_{2}.csv".format(filename, y_origin, x_origin)))

        return filename, True
    except:
        return filename, False


if __name__ == '__main__':
    # define parameters
    image_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/image_data")
    geo_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/geo_data/v2")
    label_data_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/label_data/v2")

    training_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_8")
    visualisation_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/visualisations/training_set_8")

    filenames = os.listdir(geo_data_directory)
    filename = create_training_set(filenames[0].replace(".geojson", ""), image_data_directory, label_data_directory,
                                   training_output_directory, visualisation_output_directory)
    print(filename)
