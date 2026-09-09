"""POS
Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). 
Algorithmic principles of remote PPG. 
IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
"""

import math

import numpy as np
from scipy import signal

from config import POS_HIGH_HZ, POS_LOW_HZ, POS_WIN_SEC
from preprocessing.detrend import detrend
from preprocessing.filters import band_edges
from preprocessing.processing import _process_video


def pos_algorithm(frames, fs):
    WinSec = POS_WIN_SEC
    RGB = _process_video(frames)
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.asmatrix(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.asmatrix(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    low, high = band_edges(fs, POS_LOW_HZ, POS_HIGH_HZ)
    b, a = signal.butter(1, [low, high], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP

