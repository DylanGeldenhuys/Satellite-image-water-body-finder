import numpy as np
from pathlib import Path
from scipy.ndimage.morphology import binary_closing, binary_opening
import matplotlib.pyplot as plt

prediction_data_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_1")
output_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/post_procession/post_processing_1")

filename = "2531DA03"

prediction = np.load(prediction_data_dir.joinpath(filename + '.npy'))

#result = binary_opening(prediction, structure=np.ones((10, 10))).astype(np.int)
#result = binary_closing(result, structure=np.ones((5, 5))).astype(np.int)

plt.imshow(prediction)
plt.show()
