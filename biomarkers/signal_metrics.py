"""Signal quality metrics."""

import numpy as np
from scipy.fft import rfft, rfftfreq

from config import HR_HIGH_HZ, HR_LOW_HZ
from preprocessing.filters import effective_band_hz


def spectral_snr(filtered_signal, fps):
    """Razão entre o pico do espectro e a média da banda cardíaca.

    É o SNR que alimenta `qualidade.snr`, portado literalmente do
    `rppg_core.py` em produção: os limiares SNR_BOA = 3.0 e SNR_MEDIA = 2.0
    são calibrados nesta razão LINEAR. Não troque por `compute_signal_metrics`,
    cujo `snr` está em dB (10·log10) e vive numa escala completamente
    diferente — a mesma captura sai 11.9 aqui e 2.0 lá, ou seja, "boa" vira
    "ruim" sem que nenhum limiar tenha sido tocado.

    Um pulso limpo concentra energia numa frequência só (razão alta),
    enquanto ruído de movimento e luz espalha energia pela banda inteira
    (razão perto de 1).

    Esta é a ÚNICA função do pipeline que usa a banda literal HR_LOW_HZ..
    HR_HIGH_HZ em vez de `effective_band_hz`, e é de propósito: a razão é uma
    normalização (pico contra a média da banda), não uma busca de pico, e o
    valor dos limiares depende de quantos bins entram nessa média. Medir na
    banda efetiva dá um SNR ~12% menor de forma sistemática, o que desloca a
    fronteira boa/media sem que ninguém tenha mexido em SNR_BOA.
    """
    n = len(filtered_signal)
    freqs = rfftfreq(n, d=1.0 / fps)
    magnitudes = np.abs(rfft(filtered_signal * np.hanning(n)))
    band = magnitudes[(freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)]
    if len(band) == 0:
        return 0.0
    media = float(np.mean(band))
    return 0.0 if media <= 0 else float(np.max(band) / media)


def compute_signal_metrics(filtered_signal, fps):
    """Calculate the unchanged temporal and spectral signal metrics."""
    std = np.std(filtered_signal)
    amplitude = np.max(filtered_signal) - np.min(filtered_signal)
    energy = np.sum(filtered_signal ** 2)
    signal = filtered_signal - np.mean(filtered_signal)
    std_signal = np.std(signal)
    if std_signal > 0:
        signal = signal / std_signal

    frequencies = rfftfreq(len(signal), d=1.0 / fps)
    fft_magnitude = (2.0 / len(signal)) * np.abs(rfft(signal))
    low_hz, high_hz = effective_band_hz(fps)
    heart_band = (frequencies >= low_hz) & (frequencies <= high_hz)
    fft_band = fft_magnitude[heart_band]
    fft_peak = np.max(fft_band) if len(fft_band) else 0.0

    if len(fft_band):
        peak_idx = np.argmax(fft_band)
        signal_window = np.zeros_like(fft_band, dtype=bool)
        start = max(0, peak_idx - 2)
        end = min(len(fft_band), peak_idx + 3)
        signal_window[start:end] = True
        signal_power = np.sum(fft_band[signal_window] ** 2)
        noise_band = fft_band[~signal_window]
        noise_power = np.sum(noise_band ** 2)
        total_power = signal_power + noise_power
        spectral_concentration = signal_power / total_power if total_power > 0 else 0.0
        snr = (10 * np.log10(signal_power / noise_power) if noise_power > 0
               else (np.inf if signal_power > 0 else 0.0))
        mean_noise = np.mean(noise_band) if len(noise_band) else 0.0
        peak_ratio = (fft_peak / mean_noise if mean_noise > 0
                      else (np.inf if fft_peak > 0 else 0.0))
    else:
        snr = 0.0
        peak_ratio = 0.0
        spectral_concentration = 0.0

    return {"std": std, "amplitude": amplitude, "energy": energy,
            "fft_peak": fft_peak, "snr": snr, "peak_ratio": peak_ratio,
            "spectral_concentration": spectral_concentration}
