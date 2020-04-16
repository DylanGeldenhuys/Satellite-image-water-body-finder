
from rasterio.plot import reshape_as_image
import numpy as np

#This function take an imagae and iserts a green block in the area specified by two coordinates, top left and bottom right. eg [[10,10],[50,50]]
#coordinate list is a list of coordinate pairs, eg [[[10,10],[50,50]], [[300,300],[700,800]]]


def image_cut(image, coordinates_list):
    imagecopy = image.copy()
    for crop_points in coordinate_list:
        window = np.zeros((abs(crop_points[0][0]-crop_points[1][0]),abs(crop_points[0][1]-crop_points[1][1]),3))
        window[:] = [0,177,64]
        imagecopy[crop_points[0][1]:crop_points[1][1],crop_points[0][0]:crop_points[1][0]] = window
    return(imagecopy)

