"""Pipeline de rPPG sobre médias RGB já extraídas pelo navegador."""

import math

import numpy as np

from biomarkers.heart_rate import compute_hr_fft
from biomarkers.hrv import compute_hrv
from biomarkers.signal_metrics import spectral_snr
from biomarkers.signal_quality import assess_signal_quality
from config import ABSOLUTE_MIN_FPS, MIN_VALID_FRAMES, ROI_NAMES
from extractors.chrom import chrom_window_length
from extractors.combine import combine_roi_and_methods


def _json_safe(value):
    """Converte escalares numpy em tipos que o json.dumps aceita — ver Seção 10."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def analyze_signals(fps, roi_signals, n_frames_recebidos):
    """Recebe {roi: [[r,g,b], ...]} + fps e devolve o dict do contrato."""
    faltando = [roi for roi in ROI_NAMES if roi not in roi_signals]
    if faltando:
        raise ValueError(f"ROI ausente no payload: {', '.join(faltando)}")

    n_frames = len(roi_signals[ROI_NAMES[0]])
    if any(len(roi_signals[roi]) != n_frames for roi in ROI_NAMES):
        raise ValueError("As ROIs chegaram com quantidades diferentes de frames.")

    if fps < ABSOLUTE_MIN_FPS:
        raise ValueError(
            f"Taxa de captura baixa demais ({fps:.1f} quadros/s) para estimar a "
            "frequência cardíaca. Refaça o escaneamento com o rosto dentro da "
            "área indicada e boa iluminação."
        )

    # O CHROM exige uma janela que cresce com o fps — ver Seção 4.4.
    min_frames = max(MIN_VALID_FRAMES, chrom_window_length(fps))
    if n_frames < min_frames:
        raise ValueError(
            f"Só {n_frames} quadros com o rosto visível — são necessários pelo "
            f"menos {min_frames}. Refaça o escaneamento mantendo o rosto dentro "
            "da área indicada."
        )

    signals = {roi: np.asarray(roi_signals[roi], dtype=np.float64)
               for roi in ROI_NAMES}
    for roi, array in signals.items():
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f"ROI '{roi}' não veio como uma lista de [r, g, b].")

    filtered = combine_roi_and_methods(signals, fps)
    hr_bpm = compute_hr_fft(filtered, fps)
    # Razão pico/média LINEAR, como em produção — é a escala em que
    # SNR_BOA/SNR_MEDIA foram calibrados. Ver spectral_snr().
    snr = spectral_snr(filtered, fps)

    valid_ratio = n_frames / max(n_frames_recebidos, n_frames)
    qualidade = assess_signal_quality(
        fps=fps, valid_ratio=valid_ratio, snr=snr, hr_bpm=hr_bpm,
    )

    return {
        "hr_bpm": round(float(hr_bpm), 1),
        "qualidade": qualidade,
        "n_frames": int(n_frames),
        "n_frames_recebidos": int(n_frames_recebidos),
        "fps_estimado": round(float(fps), 2),
        # Não chega ao painel do médico hoje (regra 2 da Seção 1) — ver 8.1.
        "hrv": _json_safe(compute_hrv(filtered, fps)),
    }
