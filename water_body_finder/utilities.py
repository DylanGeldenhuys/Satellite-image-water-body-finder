from shapely.geometry import LineString
import rasterio
from geojson import Point, Feature, FeatureCollection, dump
from shapely.geometry import mapping
from shapely.wkt import loads
import pandas as pd
import numpy as np
from scipy.ndimage.measurements import label
import numpy as np
import cv2 as cv
import rasterio
from rasterio.plot import reshape_as_image
from rasterio.windows import Window
from rasterio.enums import Resampling


def rgb_to_grey(rgb):
    """Converts RGB image to greyscale.

    Parameters
    ----------
    rgb : ndarray
        2D image numpy array.

    Returns
    -------
    out: ndarray (uint8)
        Output greyscale image.
    """
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    return np.array((0.299*r + 0.587*g + 0.114*b), dtype=np.uint8)


def get_boundary(mask, padding):
    """Gets boundary points from mask.

    Parameters
    ----------
    maks : ndarray
        2D mask numpy array.

    Returns
    -------
    out: array
        List of boundary points.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])

    boundary_mask = cv.filter2D(
        np.uint8(mask) * 255, -1, kernel)[padding:-padding, padding:-padding]
    boundary_points_split = np.where(boundary_mask)
    return [list(a) for a in zip(
        boundary_points_split[0], boundary_points_split[1])]


def order_points(points):
    """Orders edge points.

    Parameters
    ----------
    points : array
        Unordered array of points making up boundary.

    Returns
    -------
    out: array
        List of line boundary points.
    """
    complex_points_ls = list(points)
    ordered_list = [[]]
    polygon_index = 0
    point = complex_points_ls.pop(0)

    retry_num = 0
    temp_points = []

    while len(complex_points_ls) > 0:
        point_found = False
        for i in range(len(complex_points_ls)):
            if is_touching(complex_points_ls[i], point):
                ordered_list[polygon_index].append(point)
                point = complex_points_ls.pop(i)
                point_found = True
                temp_points = []
                retry_num = 0
                break

        if point_found == False:
            if len(ordered_list[polygon_index]) < 1:
                point = complex_points_ls.pop(0)
                continue

            if len(ordered_list[polygon_index]) > 3 and is_touching(ordered_list[polygon_index][0], point):
                ordered_list[polygon_index].append(point)
                point = complex_points_ls.pop(0)
                polygon_index += 1
                ordered_list.append([])
                temp_points = []
                retry_num = 0
            elif retry_num < 5:
                temp_points.insert(0, point)
                point = ordered_list[polygon_index].pop()
                retry_num += 1
            else:
                ordered_list[polygon_index] += temp_points
                ordered_list[polygon_index].append(point)
                point = complex_points_ls.pop(0)
                polygon_index += 1
                ordered_list.append([])
                temp_points = []
                retry_num = 0

    ordered_list[polygon_index].append(point)

    return [x for x in ordered_list if len(x) > 10]


def is_touching(point_a, point_b, resolution=1):
    return abs(point_a[0] - point_b[0]) <= resolution and abs(point_a[1] - point_b[1]) <= resolution


def reduce_noise(arr, inner_threshold, outer_threshold):
    """Reduce noise of binary 2D array.

    Parameters
    ----------
    arr : ndarray
        Binary array.

    inner_threshold: int
        Size of negative noise to reduce.

    outer_threshold: int
        Size of positive noise to reduce.

    Returns
    -------
    out: ndarray
        Binary array.
    """
    # reduce inner noise
    result = np.copy(arr)
    result = np.logical_not(result).astype(int)
    labeled_array, num = label(result)
    binc = np.bincount(labeled_array.ravel())
    noise_idx = np.where(binc <= inner_threshold)
    shp = result.shape
    mask = np.in1d(labeled_array, noise_idx).reshape(shp)
    result[mask] = 0
    result = np.logical_not(result).astype(int)

    # reduce outer noise
    labeled_array, num = label(result)
    binc = np.bincount(labeled_array.ravel())
    noise_idx = np.where(binc <= outer_threshold)
    shp = result.shape
    mask = np.in1d(labeled_array, noise_idx).reshape(shp)
    result[mask] = 0

    return result


def smooth(arr):
    """Smooth edges of blobs in binary array.

    Parameters
    ----------
    arr : ndarray
        Binary array.

    Returns
    -------
    out: ndarray
        Binary array.
    """
    result = np.copy(arr)
    blur = ((3, 3), 1)
    erode_ = (1, 1)
    dilate_ = (4, 4)
    result = np.float32(result)
    result = cv.dilate(cv.erode(cv.GaussianBlur(
        result, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))
    result = np.int8(result)

    return result


def save_geojson(ordered_list, image_src, filename):
    rasterio_object = rasterio.open(image_src)
    features = []

    def pixelcoord_to_geocoord(pixel_coordinate):
        return(rasterio_object.transform * pixel_coordinate)

    for line in ordered_list:
        tuple_of_tuples = tuple((point[1], point[0]) for point in line)
        Lstring = LineString(
            list(map(pixelcoord_to_geocoord, tuple_of_tuples)))
        features.append(Feature(geometry=Lstring.simplify(0.00001)))
    feature_collection = FeatureCollection(features)
    with open(filename, 'w') as f:
        dump(feature_collection, f)


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


def correct_point_offset(points, offset, resolution):
    return (points * resolution) + offset
