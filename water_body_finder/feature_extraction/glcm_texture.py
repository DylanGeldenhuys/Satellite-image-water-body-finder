import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data


'''texture classification using grey level co-occurrence matrices (GLCMs). 
A GLCM is a histogram of co-occurring greyscale values at a given offset over an image.'''

# the output is a tuple (dissimilarity,correlation)
def glcm_feature(patch):
    glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
    return(greycoprops(glcm, 'dissimilarity')[0, 0],greycoprops(glcm, 'correlation')[0, 0])
    