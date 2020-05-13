from scipy.ndimage.morphology import binary_closing, binary_opening

from .utilities import reduce_noise, smooth


def process(arr):
    """Performs post process functions on the binary prediction image.

    This includes two noise reduction techniques and a smoothing function.

    Parameters
    ----------
    arr : ndarray
        2D prediction numpy array.

    Returns
    -------
    out: ndarray
        Processed prediction array.
    """
    prediction_close = binary_closing(arr, iterations=2)
    prediction_open = binary_opening(prediction_close)
    prediction_noise_reduction = reduce_noise(prediction_open, 300, 6000)
    return smooth(prediction_noise_reduction)
