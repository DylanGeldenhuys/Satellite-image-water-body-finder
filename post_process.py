import numpy as np
from pathlib import Path
from scipy.ndimage.morphology import binary_closing, binary_opening
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import label
import cv2

prediction_data_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_3")
label_data_directory = Path(
    "D:/WaterBodyExtraction/WaterPolyData/label_data")
output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/dylan")

filename = "2531DA03"

prediction = np.load(prediction_data_dir.joinpath(filename + '.npy'))
label_mask = np.load(label_data_directory.joinpath(filename + '.npy'))

result = np.copy(prediction)

n_thresh = 500

result = np.logical_not(result).astype(int)
labeled_array, num_features = label(result)
binc = np.bincount(labeled_array.ravel())
noise_idx = np.where(binc <= n_thresh)
shp = result.shape
mask = np.in1d(labeled_array, noise_idx).reshape(shp)
result[mask] = 0
result = np.logical_not(result).astype(int)

n_thresh = 200

labeled_array, num_features = label(result)
binc = np.bincount(labeled_array.ravel())
noise_idx = np.where(binc <= n_thresh)
shp = result.shape
mask = np.in1d(labeled_array, noise_idx).reshape(shp)
result[mask] = 0

#blur = ((3, 3), 1)
#erode_ = (1, 1)
#dilate_ = (4, 4)
#result = np.float32(result)
# result = cv2.dilate(cv2.erode(cv2.GaussianBlur(
#    result, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))
#result = np.int8(result)

#result = binary_opening(result, structure=np.ones((2, 2))).astype(np.int)
#result = binary_closing(result, structure=np.ones((10, 10))).astype(np.int)
#result = binary_opening(result, structure=np.ones((10, 10))).astype(np.int)

result_stretched = np.repeat(result, 3, axis=1)
result_stretched = np.repeat(result_stretched, 3, axis=0)

label_cropped = label_mask[4:-4, 4:-4]
label_cropped = np.logical_not(label_cropped).astype(int)

np.save(output_dir.joinpath("{}_prediction.npy".format(filename)), result_stretched)
np.save(output_dir.joinpath("{}_label.npy".format(filename)), label_cropped)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(label_cropped)
fig.add_subplot(1, 2, 2)
plt.imshow(result_stretched)
plt.show()
