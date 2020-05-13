from shapely.geometry import LineString
import rasterio
from geojson import Point, Feature, FeatureCollection, dump
from shapely.geometry import mapping
from shapely.wkt import loads
import pandas as pd
import numpy as np
from scipy.ndimage.measurements import label
import numpy as np
import cv2


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


def get_boundary(mask):
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
    edges = []

    height = mask.shape[0] - 1
    width = mask.shape[1] - 1

    for j in range(1, height):
        for i in range(1, width):
            if (mask[j, i] == 0):
                if (mask[j, i + 1] == 1 or mask[j, i - 1] == 1 or mask[j + 1, i] == 1 or mask[j - 1, i] == 1):
                    edges.append([j, i])
    return edges


def order_edge_points(points):
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
    point = points[0]
    point_index = 0

    while len(complex_points_ls) > 1:
        point_found = False
        for i in range(len(complex_points_ls)):
            if (abs(complex_points_ls[i][0] - point[0]) <= 1 and abs(complex_points_ls[i][1] - point[1]) <= 1):
                if ((point[0], point[1]) != (points[i][0], points[i][1])):
                    ordered_list[polygon_index].append(list(point))
                    point = complex_points_ls[i]
                    complex_points_ls.pop(point_index)
                    point_index = i
                    point_found = True
                    break

        if point_found == False:
            ordered_list[polygon_index].append(list(point))
            complex_points_ls.pop(point_index)
            point = complex_points_ls[0]
            point_index = 0
            polygon_index += 1
            ordered_list.append([])

    ordered_list[polygon_index].append(list(point))

    return ordered_list


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
    result = cv2.dilate(cv2.erode(cv2.GaussianBlur(
        result, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))
    result = np.int8(result)

    return result


def save_geojson(ordered_list, image_src, filename):
    rasterio_object = rasterio.open(image_src)
    features = []

    def pixelcoord_to_geocoord(pixel_coordinate):
        return(rasterio_object.transform * pixel_coordinate)

    for line in ordered_list:
        tuple_of_tuples = tuple(tuple(point) for point in line)
        Lstring = LineString(
            list(map(pixelcoord_to_geocoord, tuple_of_tuples)))
        features.append(Feature(geometry=Lstring))
    feature_collection = FeatureCollection(features)
    with open(filename, 'w') as f:
        dump(feature_collection, f)
