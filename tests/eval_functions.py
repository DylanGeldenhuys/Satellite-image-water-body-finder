
# Author: Dylan Geldenhuys



from sklearn.neighbors import NearestNeighbors
import numpy as np


# Extract boundaries using simons code for extracting edges of segment
def get_boundary(mask):
    edges = []
    for j in range(mask.shape[0] - 1):
        for i in range(mask.shape[1] - 1):
            if (mask[j, i] != mask[j, i + 1] 
            or mask[j, i] != mask[j, i - 1] 
            or mask[j, i] != mask[j + 1, i]
            or mask[j, i] != mask[j - 1, i]):
                edges.append([j,i])
    return(edges)

'''measures the distance from each reference boundary (50x50cm) cell to the nearest extracted
boundary cell, which provides an indication of how close in geographical space the extracted
boundaries are to the actual boundary.'''
def MAEi(extracted_boundary, reference_boundary):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(extracted_boundary)
    distances, indices = nbrs.kneighbors(reference_boundary)
    return(np.average(distances))

'''MAEj measures the distance from each extracted boundary cell to the nearest reference boundary
cells and is an indicator of boundaries within a water body (caused by oversemgentation)''' 
def MAEj(reference_boundary, extracted_boundary):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(reference_boundary)
    distances, indices = nbrs.kneighbors(extracted_boundary)
    return(np.average(distances)) 

# calculates the MAEi and MAEj for a pair of mask images (refrence and predicted)
def eval_forImage(reference_image,extracted_image):
    reference_boundary = get_boundary(reference_image)
    extracted_boundary = get_boundary(extracted_image)
    return('MAEi:{}'.format(MAEi(extracted_boundary, reference_boundary)),'MAEj:{}'.format(MAEj(reference_boundary, extracted_boundary)))