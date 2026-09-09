"""GREEN rPPG extractor."""

import numpy as np
from scipy import signal as sig


def green_algorithm(rgb_raw):
    """Extract the z-scored, detrended green-channel rPPG signal."""
    green = sig.detrend(rgb_raw[:, 1], type="linear")
    return (green - np.mean(green)) / (np.std(green) + 1e-8)