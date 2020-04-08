from scipy.ndimage.measurements import label
import numpy as np
import cv2


def reduce_noise(arr, inner_threshold, outer_threshold):
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
    result = np.copy(arr)
    blur = ((3, 3), 1)
    erode_ = (1, 1)
    dilate_ = (4, 4)
    result = np.float32(result)
    result = cv2.dilate(cv2.erode(cv2.GaussianBlur(
        result, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))
    result = np.int8(result)

    return result
