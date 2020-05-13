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

import cv2 as cv

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def rgb2gray(rgb):
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.array((0.299*r + 0.587*g + 0.114*b), dtype=np.uint8)


def create_test_set(filename, image_data_directory, label_data_directory, training_output_directory, visualisation_output_directory):
    try:
        outer_padding = 50
        inner_padding = 20
        max_set_size = 3000
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

                height, width, channels = image_data.shape

                training_set = pd.DataFrame()

                # color feature
                training_set['color_r'] = image_data[inner_padding:-
                                                     inner_padding:data_resolution, inner_padding:-
                                                     inner_padding:data_resolution, 0].flatten()
                training_set['color_g'] = image_data[inner_padding:-
                                                     inner_padding:data_resolution, inner_padding:-
                                                     inner_padding:data_resolution, 1].flatten()
                training_set['color_b'] = image_data[inner_padding:-
                                                     inner_padding:data_resolution, inner_padding:-
                                                     inner_padding:data_resolution, 2].flatten()
                # save for visualisation purposes
                Image.fromarray(image_data.astype(np.uint8)).save(visualisation_output_directory.joinpath(
                    "{0}_{1}_{2}.jpg".format(filename, y_origin, x_origin)))

                # image blur feature
                blur_feature = cv.GaussianBlur(image_data, (39, 39), 0)
                training_set['color_r_blur'] = blur_feature[inner_padding:-
                                                            inner_padding:data_resolution, inner_padding:-
                                                            inner_padding:data_resolution, 0].flatten()
                training_set['color_g_blur'] = blur_feature[inner_padding:-
                                                            inner_padding:data_resolution, inner_padding:-
                                                            inner_padding:data_resolution, 1].flatten()
                training_set['color_b_blur'] = blur_feature[inner_padding:-
                                                            inner_padding:data_resolution, inner_padding:-
                                                            inner_padding:data_resolution, 2].flatten()
                Image.fromarray(blur_feature.astype(np.uint8)).save(visualisation_output_directory.joinpath(
                    "{0}_{1}_{2}_blur.jpg".format(filename, y_origin, x_origin)))

                # average color feature
                average_color_feature = np.average(np.average(
                    image_data, axis=0), axis=0)
                training_set['average_color_r'] = int(
                    round(average_color_feature[0]))
                training_set['average_color_g'] = int(
                    round(average_color_feature[1]))
                training_set['average_color_b'] = int(
                    round(average_color_feature[2]))

                # entropy feature small
                entropy_feature_small = entropy(
                    img_as_ubyte(rgb2gray(image_data)), disk(5))
                training_set['entropy_small'] = entropy_feature_small[inner_padding:-
                                                                      inner_padding:data_resolution, inner_padding:-
                                                                      inner_padding:data_resolution].flatten()
                Image.fromarray(entropy_feature_small * 25).convert('RGB').save(visualisation_output_directory.joinpath(
                    "{0}_{1}_{2}_entropy_small.jpg".format(filename, y_origin, x_origin)))

                # entropy feature large
                entropy_feature_large = entropy(
                    img_as_ubyte(rgb2gray(image_data)), disk(15))
                training_set['entropy_large'] = entropy_feature_large[inner_padding:-
                                                                      inner_padding:data_resolution, inner_padding:-
                                                                      inner_padding:data_resolution].flatten()
                Image.fromarray(entropy_feature_large * 25).convert('RGB').save(visualisation_output_directory.joinpath(
                    "{0}_{1}_{2}_entropy_large.jpg".format(filename, y_origin, x_origin)))

                # label
                training_set['label'] = label_data[y_origin:y_origin +
                                                   window_height:data_resolution, x_origin:x_origin + window_width:data_resolution].flatten()
                plt.imsave(visualisation_output_directory.joinpath("{0}_{1}_{2}_label.jpg".format(
                    filename, y_origin, x_origin)), label_data[y_origin-inner_padding:y_origin-inner_padding+height, x_origin-inner_padding:x_origin-inner_padding+width], cmap=cm.gray)

                training_set.to_csv(training_output_directory.joinpath(
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
        "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_9")
    visualisation_output_directory = Path(
        "D:/WaterBodyExtraction/WaterPolyData/visualisations/training_set_9")

    filenames = os.listdir(geo_data_directory)
    filename = create_training_set(filenames[0].replace(".geojson", ""), image_data_directory, label_data_directory,
                                   training_output_directory, visualisation_output_directory)
    print(filename)
