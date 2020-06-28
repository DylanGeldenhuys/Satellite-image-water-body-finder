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
from .predict import Model


def save_prediction_mask(image_input_dir, output_dir, rfc_dir, version, padding, window_size, resolution):
    image_input_path = Path(image_input_dir)
    rfc_path = Path(rfc_dir).joinpath("rfc_{}.p".format(version))
    output_path = Path(output_dir).joinpath(
        "predictions").joinpath("version_{}".format(version))
    output_path.mkdir(parents=True, exist_ok=True)

    f = open(rfc_path, 'rb')
    rfc = pickle.load(f)

    filenames = os.listdir(image_input_dir)
    pool = mp.Pool()
    for filename in filenames[0:1]:
        img_src = image_input_path.joinpath(filename)
        model = Model(rfc, img_src, padding, window_size, resolution, pool)
        prediction = model.predict_mask_full(pool)
        np.save(output_path.joinpath(
            filename.replace('tif', 'npy')), prediction)
        print("Completed: {}".format(filename.replace('.geojson', '')))

    pool.close()
    pool.join()


def save_polygons(image_input_dir, output_dir, rfc_dir, version, padding, window_size, resolution):
    image_input_path = Path(image_input_dir)
    rfc_path = Path(rfc_dir).joinpath("rfc_{}.p".format(version))
    output_path = Path(output_dir).joinpath(
        "polygons").joinpath("version_{}".format(version))
    output_path.mkdir(parents=True, exist_ok=True)

    f = open(rfc_path, 'rb')
    rfc = pickle.load(f)

    filenames = os.listdir(image_input_dir)
    pool = mp.Pool()
    for filename in filenames:
        if filename[-3:] != 'tif':
            continue
        else:
            img_src = image_input_path.joinpath(filename)
            model = Model(rfc, img_src, padding, window_size, resolution, pool)
            prediction = model.predict_polygons(pool)
            np.save(output_path.joinpath(
                filename.replace('tif', 'npy')), prediction)
        print("Completed: {}".format(filename.replace('.geojson', '')))

    pool.close()
    pool.join()
