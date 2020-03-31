import numpy as np


def extract_variance(data):
    """Test."""

    shape = data.shape
    return(np.var(data[0:shape[0], 0:shape[1]]))
