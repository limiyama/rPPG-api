"""Fusão dos métodos de extração e das ROIs em um único sinal rPPG."""

import numpy as np

from config import METHOD_WEIGHTS, ROI_WEIGHTS, SMOOTH_WINDOW
from extractors.chrom import chrom_algorithm
from extractors.green import green_algorithm
from extractors.ica import ica_algorithm
from extractors.pos import pos_algorithm
from preprocessing.filters import bandpass_filter, moving_average_smooth


def _zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)


def _align_polarity(reference, other):
    """CHROM, POS, ICA e GREEN têm convenções de sinal próprias; sem alinhar
    a polaridade os sinais se cancelam na soma ponderada."""
    correlation = np.corrcoef(reference, other)[0, 1]
    return -other if np.isfinite(correlation) and correlation < 0 else other


def combine_roi_and_methods(roi_signals, fps):
    combined_per_roi = []
    roi_weights = []

    for roi_name, rgb_raw in roi_signals.items():
        rgb = moving_average_smooth(rgb_raw, window=SMOOTH_WINDOW)

        signals = {
            "chrom": chrom_algorithm(rgb, fps),
            "pos": pos_algorithm(rgb, fps),
            "green": green_algorithm(rgb),
            "ica": ica_algorithm(rgb, fps),
        }

        # CHROM sai mais curto (overlap-add em janelas): alinhar comprimentos.
        min_len = min(len(s) for s in signals.values())
        z = {name: _zscore(s[:min_len]) for name, s in signals.items()}

        reference = z["chrom"]
        for name in ("pos", "green", "ica"):
            z[name] = _align_polarity(reference, z[name])

        combined_per_roi.append(sum(METHOD_WEIGHTS[n] * z[n] for n in z))
        roi_weights.append(ROI_WEIGHTS[roi_name])

    min_len = min(len(s) for s in combined_per_roi)
    final = np.average(
        np.vstack([s[:min_len] for s in combined_per_roi]),
        axis=0,
        weights=roi_weights,
    )
    return bandpass_filter(final, fps)
