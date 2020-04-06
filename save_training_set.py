# imports

from water_body_finder.feature_extraction import extract_variance, extract_mean_color, extract_entropy
from water_body_finder.utilities import create_window
from rasterio.plot import reshape_as_image
import pandas as pd
import rasterio
import cv2
from pathlib import Path
import os
import numpy as np


# must be odd number
data_resolution = 5

# size of largest window used in feature extraction, must be odd number
data_padding = 15

image_data_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/image_data")
label_data_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/label_data")
output_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/training_sets/training_set_2")

# define training set format
training_set_format = {
    'color_r': [],
    'color_g': [],
    'color_b': [],
    'entropy_a': [],
    'label': []
}


def extract_features(image_data, point, data_resolution, training_set):
    '''define feature extraction functions'''
    window_a = create_window(image_data, point, 5)

    mean_color = extract_mean_color(window_a)

    training_set['color_r'].append(mean_color[0])
    training_set['color_g'].append(mean_color[1])
    training_set['color_b'].append(mean_color[2])
    training_set['entropy_a'].append(extract_entropy(window_a))


for filename in os.listdir(image_data_directory)[0:50]:
    # load files
    raster_image_data = rasterio.open(
        image_data_directory.joinpath(filename)).read()
    image_data = reshape_as_image(raster_image_data)

    mask_data = np.load(label_data_directory.joinpath(
        filename.replace("tif", "npy")))

    # loop through image data and create training
    height = int(
        image_data.shape[0] / data_resolution - data_padding / data_resolution)
    width = int(image_data.shape[1] /
                data_resolution - data_padding / data_resolution)

    data_set = []

    offset = round(data_padding/2) + 100

    num_positive = 0
    num_negative = 0

    training_set = training_set_format

    percentage_tracker = 1

    for j in range(height - 200):
        for i in range(width - 200):
            y = j * data_resolution + offset
            x = i * data_resolution + offset

            if (((j / height) * 100) > percentage_tracker):
                print("File: {0} {1}%".format(filename, percentage_tracker))
                percentage_tracker += 1

            # extract label
            label_window = create_window(mask_data, [y, x], data_resolution)
            label = label_window.mean(axis=0).mean(axis=0) < 0.5

            # if (label or (num_positive > num_negative)):

            # extract features
            extract_features(
                image_data, [y, x], data_resolution, training_set)
            training_set['label'].append(label)

            # keep track of positive to negative ratios in training data
            # if (label):
            #    num_positive += 1
            # else:
            #    num_negative += 1

    print("File: {0} completed".format(filename))
    print("\n")

    # save to csv
    training_set_df = pd.DataFrame(training_set)
    training_set_df.to_csv(output_directory.joinpath(
        filename.replace("tif", "csv")))
