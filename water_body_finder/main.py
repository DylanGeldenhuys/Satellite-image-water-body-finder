import pickle
from pathlib import Path
import os
import numpy as np

from .feature_extraction import extract_features
from .utilities import get_boundary, order_edge_points, save_geojson
from .post_process import process
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
from rasterio.enums import Resampling


def find_waterbodies(image_input_dir, output_dir, rfc_dir=os.path.dirname(os.path.realpath(__file__)) + '/rfc', rfc_version="0", padding=20, window_size=3000):
    f = open(Path(rfc_dir).joinpath('rfc_{}.p'.format(rfc_version)), 'rb')
    rfc = pickle.load(f)

    save_multiple_async(rfc, image_input_dir, output_dir, padding, window_size)


def save_multiple_async(rfc, image_input_dir, output_dir, padding, window_size):
    filenames = os.listdir(image_input_dir)

    for filename in filenames[5:6]:
        image_src = Path(image_input_dir).joinpath(filename)
        boundary_lines = find_image_boundary_lines_async(
            rfc, image_src, padding, window_size)
        geo_json_dir = Path(output_dir).joinpath("geo_data")
        geo_json_dir.mkdir(parents=True, exist_ok=True)

        save_geojson(boundary_lines, image_src, geo_json_dir.joinpath(
            filename.replace('tif', 'geojson')))


def find_image_boundary_lines_async(rfc, image_input_src, padding, window_size):
    """"""
    dataset = rasterio.open(image_input_src)
    boundary_points = np.array([])
    for y in range(padding, dataset.height - padding, window_size):
        for x in range(padding, dataset.width - padding, window_size):
            boundary_points = np.concatenate(boundary_points, find_boundary_points(
                rfc, dataset, [y, x], window_size, padding))
            print("Completed {}, {}".format(x, y))

    print("Ordering edge points...")
    return order_edge_points(boundary_points)


def find_boundary_points(rfc, dataset, offset, window_size, padding):
    """"""
    image_data = load_window(dataset, offset, window_size, padding)
    features = extract_features(image_data, padding)

    prediction = rfc.predict(features)
    prediction_img = prediction.reshape(
        image_data.shape[0] - padding * 2, image_data.shape[1] - padding * 2)
    processed_prediction = process(prediction_img)
    boundary_points = get_boundary(processed_prediction)

    return correct_point_offset(boundary_points, offset)


def load_window(dataset, offset, window_size, padding):
    x = offset[1] - padding
    y = offset[0] - padding

    width = window_size + padding * 2
    if (x + width) > dataset.width:
        width = (dataset.width - x)

    height = window_size + padding * 2
    if (y + height) > dataset.height:
        height = (dataset.height - y)

    rasterio_image_data = dataset.read(
        window=Window(x, y,
                      width, height)
    )

    return reshape_as_image(rasterio_image_data)


def correct_point_offset(points, offset):
    return points + offset
