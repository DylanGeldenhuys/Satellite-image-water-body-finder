gfrom sklearn.neighbors import NearestNeighbors
import numpy as np


def MAEi(extracted_boundary, reference_boundary):
    '''measures the distance from each reference boundary (50x50cm) cell to the nearest extracted
    boundary cell, which provides an indication of how close in geographical space the extracted
    boundaries are to the actual boundary.'''

    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='ball_tree').fit(extracted_boundary)
    distances = nbrs.kneighbors(reference_boundary)
    return(np.average(distances[0]))


def MAEj(reference_boundary, extracted_boundary):
    '''MAEj measures the distance from each extracted boundary cell to the nearest reference boundary
    cells and is an indicator of boundaries within a water body (caused by oversemgentation)'''

    nbrs = NearestNeighbors(
        n_neighbors=1, algorithm='ball_tree').fit(reference_boundary)
    distances = nbrs.kneighbors(extracted_boundary)
    return(np.average(distances[0]))
