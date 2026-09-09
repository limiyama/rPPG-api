"""Filtering functions for rPPG signals."""

import numpy as np
from scipy import signal as sig

from config import BAND_MAX_NYQUIST_RATIO, HR_HIGH_HZ, HR_LOW_HZ


def band_edges(fps, low_hz, high_hz):
    """Bordas normalizadas seguras para qualquer butter/filtfilt do pipeline.

    Trava o topo em 90% do Nyquist: acima disso o Butterworth fica
    numericamente instável e o pico da FFT pode cair em ruído de borda —
    algo que a 24 fps nunca aparecia e a 8 fps aparece.
    """
    nyq = 0.5 * fps
    low = low_hz / nyq
    high = min(high_hz / nyq, BAND_MAX_NYQUIST_RATIO)
    if not 0 < low < high:
        raise ValueError(f"Banda inválida para fps={fps:.2f}: [{low_hz}, {high_hz}] Hz")
    return low, high


def effective_band_hz(fps, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ):
    """Banda cardíaca em Hz efetivamente aplicada pelo bandpass_filter.

    É o que compute_hr_fft precisa usar para procurar o pico dentro da
    passagem real do filtro — ver Seção 4.3.

    `spectral_snr` é a exceção deliberada: usa a banda literal, porque lá o
    número de bins define a escala dos limiares de qualidade.
    """
    return low_hz, min(high_hz, BAND_MAX_NYQUIST_RATIO * 0.5 * fps)


def bandpass_filter(x, fps, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ, order=4):
    """Apply the existing Butterworth cardiac band-pass filter."""
    low, high = band_edges(fps, low_hz, high_hz)
    b, a = sig.butter(order, [low, high], btype="band")
    return sig.filtfilt(b, a, x)


def moving_average_smooth(x, window=3):
    """Smooth each RGB channel with the existing moving-average method."""
    if window <= 1:
        return x
    kernel = np.ones(window) / window
    return np.array([np.convolve(x[:, channel], kernel, mode="same")
                     for channel in range(x.shape[1])]).T
