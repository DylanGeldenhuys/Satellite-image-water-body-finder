import pickle
from pathlib import Path
import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import rasterio
from shapely.geometry import Polygon

from .feature_extraction import extract_features
from .utilities import get_boundary, order_points, save_geojson, load_window, correct_point_offset, is_touching
from .post_process import process


def find_waterbodies(image_input_dir, output_dir, rfc_dir=os.path.dirname(os.path.realpath(__file__)) + '/rfc', rfc_version="0", padding=600, window_size=3000, resolution=3):
    f = open(Path(rfc_dir).joinpath('rfc_{}.p'.format(rfc_version)), 'rb')
    rfc = pickle.load(f)

    save_multiple(rfc, image_input_dir, output_dir,
                  padding, window_size, resolution)


def save_multiple(rfc, image_input_dir, output_dir, padding, window_size, resolution):
    filenames = os.listdir(image_input_dir)

    for filename in filenames:
        image_src = Path(image_input_dir).joinpath(filename)
        boundary_lines, grid = find_image_boundary_lines_async(
            rfc, image_src, padding, window_size, resolution)
        geo_json_dir = Path(output_dir).joinpath("geo_data")
        geo_json_dir.mkdir(parents=True, exist_ok=True)

        visual_lines = []
        visual_lines += boundary_lines
        visual_lines += grid

        np.save(geo_json_dir.joinpath(
            filename.replace('tif', 'npy')), visual_lines)
        save_geojson(boundary_lines, image_src, geo_json_dir.joinpath(
            filename.replace('tif', 'geojson')))


def find_image_boundary_lines_async(rfc, image_input_src, padding, window_size, resolution):
    """"""
    pool = mp.Pool()

    dataset = rasterio.open(image_input_src)

    boundary_lines_ls = []

    def update_boundary(new_boundary_points):
        boundary_lines_ls.append(new_boundary_points)
        print("Completed: {}".format(len(boundary_lines_ls)))

    def error_callback(error):
        print(str(error))

    for y in range(padding, dataset.height - padding, window_size):
        for x in range(padding, dataset.width - padding, window_size):
            pool.apply_async(find_boundary_lines, args=(
                rfc, image_input_src, [y, x], window_size, padding, resolution), error_callback=error_callback, callback=update_boundary)

    pool.close()
    pool.join()

    grid = []
    for y in range(padding, dataset.height - padding, window_size):
        grid.append([[y, 0], [y, dataset.width]])
    for x in range(padding, dataset.width - padding, window_size):
        grid.append([[0, x], [dataset.height, x]])

    boundary_lines = []
    for line in boundary_lines_ls:
        boundary_lines += line

    return stitch_boundary_lines(boundary_lines, resolution * 10), grid


def stitch_boundary_lines(boundary_lines, resolution):
    polygons = []
    line_ls = boundary_lines
    if len(line_ls) < 1:
        return []
    current_line = line_ls.pop(0)

    while len(line_ls) > 0:
        found = False
        if is_touching(current_line[0], current_line[-1], resolution):
            polygons.append(current_line)
            current_line = line_ls.pop(0) if len(line_ls) > 0 else None
            continue

        for i in range(len(line_ls)):
            if is_touching(current_line[-1], line_ls[i][0], resolution):
                current_line = list(current_line) + list(line_ls.pop(i))
                found = True
                break
            elif is_touching(current_line[-1], line_ls[i][-1], resolution):
                current_line = list(current_line) + \
                    list(reversed(line_ls.pop(i)))
                found = True
                break

        if found == False:
            polygons.append(current_line)
            current_line = line_ls.pop(0) if len(line_ls) > 0 else None

    return [list(polygon) + [polygon[0]] for polygon in polygons]


def find_boundary_lines(rfc, image_input_src, offset, window_size, padding, resolution):
    """"""
    dataset = rasterio.open(image_input_src)
    image_data = load_window(dataset, offset, window_size, padding)
    features = extract_features(image_data, resolution)

    prediction = rfc.predict(features)
    height, width, channels = image_data[::resolution, ::resolution, :].shape
    prediction_img = prediction.reshape(height, width)
    processed_prediction = process(prediction_img)

    boundary_points = get_boundary(
        processed_prediction, int(padding / resolution))

    if len(boundary_points) < 1:
        return []

    boundary_lines = order_points(boundary_points)

    return [correct_point_offset(np.array(line), np.array(offset), resolution) for line in boundary_lines]
