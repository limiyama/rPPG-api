"""Heart-rate calculation."""

import numpy as np
from scipy.fft import rfft, rfftfreq

from preprocessing.filters import effective_band_hz


def compute_hr_fft(filtered_signal, fps):
    """Estimate heart rate using the unchanged FFT procedure.

    O pico é procurado na mesma banda efetiva que o bandpass_filter aplicou:
    se o filtro corta em 3.6 Hz e a busca fosse até 4.0 Hz, o argmax poderia
    cair em ruído atenuado fora da passagem (Seção 4.3).
    """
    n = len(filtered_signal)
    windowed = filtered_signal * np.hanning(n)
    freqs = rfftfreq(n, d=1.0 / fps)
    mags = np.abs(rfft(windowed))
    low, high = effective_band_hz(fps)
    valid = (freqs >= low) & (freqs <= high)
    if not valid.any():
        raise ValueError(
            f"Faixa espectral vazia a {fps:.1f} frames/s — captura curta demais."
        )
    return float(freqs[valid][np.argmax(mags[valid])] * 60.0)
