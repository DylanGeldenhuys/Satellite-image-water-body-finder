from skimage.measure import shannon_entropy


def extract_entropy(window):
    return (shannon_entropy(window))
