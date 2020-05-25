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
from .utilities import get_boundary, order_points, save_geojson, load_window, correct_point_offset, is_touching, post_process


def find_waterbodies(image_input_dir, output_dir, rfc_dir=os.path.dirname(os.path.realpath(__file__)) + '/rfc', rfc_version="1", padding=50, window_size=3000, resolution=3):
    image_input_path = Path(image_input_dir)
    rfc_path = Path(rfc_dir).joinpath("rfc_{}.p".format(rfc_version))
    output_path = Path(output_dir).joinpath("geo_data")
    output_path.mkdir(parents=True, exist_ok=True)

    f = open(rfc_path, 'rb')
    rfc = pickle.load(f)

    filenames = os.listdir(image_input_dir)
    pool = mp.Pool()
    for filename in filenames:
        img_src = image_input_path.joinpath(filename)
        model = Model(rfc, img_src, padding, window_size, resolution, pool)
        polygons = model.predict_polygons(pool)
        save_geojson(polygons, img_src, output_path.joinpath(
            filename.replace('tif', 'geojson')))
        print("Completed: {}".format(filename.replace('.tif', '')))

    pool.close()
    pool.join()


class Model:
    def __init__(self, rfc, img_src, padding, window_size, resolution, pool):
        self.rfc = rfc
        self.img_src = img_src
        self.padding = padding
        self.window_size = window_size
        self.resolution = resolution

    def predict_polygons(self, pool):
        mask = self.predict_mask_full(pool)
        processed_mask = post_process(mask)
        boundary_mask = get_boundary(processed_mask, self.padding)

        boundary_lines_ls = []

        def callback(result):
            boundary_lines_ls.append(result)

        def error_callback(error):
            print(str(error))

        height, width = boundary_mask.shape
        results = []
        for y in range(self.padding, height - self.padding, self.window_size):
            for x in range(self.padding, width - self.padding, self.window_size):
                offset = [y, x]
                section = get_section(
                    boundary_mask, offset, self.window_size)
                result = pool.apply_async(
                    self.get_boundary_lines, args=[section, offset], error_callback=error_callback, callback=callback)
                results.append(result)

        [result.wait() for result in results]

        boundary_lines = []
        for boundary_line_group in boundary_lines_ls:
            for boundary_line in boundary_line_group:
                if len(boundary_line) > 0:
                    boundary_lines.append(boundary_line)

        return self.stitch_polygons(boundary_lines)

    def stitch_polygons(self, lines):
        polygons = []
        iteration_depth = 10

        if len(lines) < 1:
            return []

        current_line = lines.pop(0)

        while len(lines) > 0:
            found = False

            shortest_iteration_depth = int(len(current_line) / 2) - 1
            current_line_iteration_depth = iteration_depth if shortest_iteration_depth > iteration_depth else shortest_iteration_depth

            for j in range(current_line_iteration_depth):
                current_line_index = len(current_line) - (1 + j)
                for k in range(current_line_iteration_depth):
                    if is_touching(current_line[current_line_index], current_line[k], self.resolution):
                        polygons.append(
                            current_line[k:current_line_index + 1])
                        current_line = lines.pop(
                            0) if len(lines) > 0 else None
                        found = True
                        break
                if found:
                    break
            if found:
                continue

            if is_touching(current_line[0], current_line[-1], self.resolution):
                polygons.append(current_line)
                current_line = lines.pop(0) if len(lines) > 0 else None
                continue

            for i in range(len(lines)):
                shortest_iteration_depth = int(len(lines[i]) / 2) - 1
                testing_line_iteration_depth = iteration_depth if shortest_iteration_depth > iteration_depth else shortest_iteration_depth

                for j in range(current_line_iteration_depth):
                    current_line_index = len(current_line) - (1 + j)
                    for k in range(testing_line_iteration_depth):
                        if is_touching(current_line[current_line_index], lines[i][k], self.resolution):
                            current_line = current_line[:current_line_index + 1]
                            current_line = list(
                                current_line) + list(lines.pop(i)[k:])
                            found = True
                            break
                        elif is_touching(current_line[current_line_index], list(reversed(lines[i]))[k], self.resolution):
                            current_line = current_line[:current_line_index + 1]
                            current_line = list(
                                current_line) + list(reversed(lines.pop(i)))[k:]
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

            if found == False:
                polygons.append(current_line)
                current_line = lines.pop(0) if len(lines) > 0 else None

        return polygons

    def get_boundary_lines(self, mask, offset):
        boundary_points_split = np.where(mask)
        boundary_points = [list(a) for a in zip(
            boundary_points_split[0], boundary_points_split[1])]

        lines = order_points(boundary_points)
        return [correct_point_offset(line, np.array(offset) * self.resolution, self.resolution) for line in lines]

    def predict_mask_full(self, pool):
        """"""
        dataset = rasterio.open(self.img_src)
        width = int(dataset.width / self.resolution)
        height = int(dataset.height / self.resolution)
        prediction = np.full([height, width], True, dtype=bool)

        def callback(result):
            mask, offset = result
            prediction[offset[0]: offset[0] + mask.shape[0],
                       offset[1]: offset[1] + mask.shape[1]] = mask

        def error_callback(error):
            print(str(error))

        results = []
        for y in range(self.padding, dataset.height - self.padding, self.window_size):
            for x in range(self.padding, dataset.width - self.padding, self.window_size):
                offset = [y, x]
                result = pool.apply_async(self.predict_mask, args=[
                    offset], error_callback=error_callback, callback=callback)
                results.append(result)

        [result.wait() for result in results]

        return prediction

    def predict_mask(self, offset):
        """"""
        dataset = rasterio.open(self.img_src)
        image_data = load_window(
            dataset, offset, self.window_size, self.padding)
        features = extract_features(image_data, self.resolution, self.padding)

        prediction = self.rfc.predict(features)
        height, width = image_data[self.padding: -
                                   self.padding: self.resolution, self.padding: -self.padding: self.resolution, 0].shape
        return prediction.reshape(height, width), [int(x / self.resolution) for x in offset]


def get_section(arr, offset, window_size):
    x = offset[1]
    y = offset[0]

    arr_height, arr_width = arr.shape
    width = window_size
    if (x + width) > arr_width:
        width = (arr_width - x)

    height = window_size
    if (y + height) > arr_height:
        height = (arr_height - y)

    return arr[y: y + height, x: x + width]
