import numpy as np
from pathlib import Path
from scipy.ndimage.morphology import binary_closing, binary_opening
import matplotlib.pyplot as plt

prediction_data_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_3")
output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/post_procession/post_processing_2")

filename = "2531DA03"

prediction = np.load(prediction_data_dir.joinpath(filename + '.npy'))

result = prediction
result = binary_opening(result, structure=np.ones((2, 2))).astype(np.int)
result = binary_closing(result, structure=np.ones((10, 10))).astype(np.int)
result = binary_opening(result, structure=np.ones((10, 10))).astype(np.int)

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(prediction)
fig.add_subplot(1, 2, 2)
plt.imshow(result)
plt.show()
