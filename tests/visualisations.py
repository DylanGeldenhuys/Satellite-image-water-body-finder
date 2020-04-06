import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

prediction_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/predictions/predictions_1")
label_dir = Path(
    "D:/WaterBodyExtraction/WaterPolyData/label_data")

filename = "2531DA03"

result = np.load(prediction_dir.joinpath(filename + '.npy'))
label = np.load(Path(label_dir.joinpath(filename + '.npy')))

fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(result)
fig.add_subplot(1, 2, 2)
plt.imshow(label)
plt.show()
