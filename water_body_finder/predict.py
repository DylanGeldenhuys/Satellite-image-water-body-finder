import matplotlib.pyplot as plt
from pathlib import Path
import os
import json
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
from rasterio.enums import Resampling
import numpy as np
import pickle
import multiprocessing as mp

from .feature_extraction import extract_features
from .utilities import get_boundary, order_points, save_geojson, load_window, correct_point_offset, is_touching


def predict_mask_full(rfc, img_src, padding, window_size, resolution, pool):
    """"""
    dataset = rasterio.open(img_src)
    width = int(dataset.width / resolution)
    height = int(dataset.height / resolution)
    prediction = np.full([height, width], True, dtype=bool)

    def callback(result):
        mask, offset = result
        prediction[offset[0]:offset[0] + mask.shape[0],
                   offset[1]:offset[1] + mask.shape[1]] = mask

    def error_callback(error):
        print(str(error))

    results = []
    for y in range(padding, dataset.height - padding, window_size):
        for x in range(padding, dataset.width - padding, window_size):
            result = pool.apply_async(predict_mask, args=(
                rfc, img_src, [y, x], window_size, padding, resolution), error_callback=error_callback, callback=callback)
            results.append(result)

    [result.wait() for result in results]

    return prediction


def predict_mask(rfc, image_input_src, offset, window_size, padding, resolution):
    """"""
    dataset = rasterio.open(image_input_src)
    image_data = load_window(dataset, offset, window_size, padding)
    features = extract_features(image_data, resolution, padding)

    prediction = rfc.predict(features)
    height, width = image_data[padding:-
                               padding:resolution, padding:-padding:resolution, 0].shape
    return prediction.reshape(height, width), [int(x / resolution) for x in offset]
