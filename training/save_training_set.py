# imports

from rasterio.plot import reshape_as_image
import pandas as pd
import rasterio
import cv2
from water_body_finder.utilities import create_window
from water_body_finder.feature_extraction import extract_variance, extract_mean_color, extract_entropy
import sys
sys.path.append('../')

# define parameters

# must be odd number
data_resolution = 5

# size of largest window used in feature extraction, must be odd number
data_padding = 15

image_data_directory = "D:\\WaterBodyExtraction\\WaterPolyData\\image_data"
label_data_directory = "C:\\personal\satalite-image-water-body-finder\\data\\label_masks"
output_directory = "D:\\WaterBodyExtraction\\WaterPolyData\\training_sets\\training_set_1"
