"""Critérios de aceitação do serviço de rPPG."""

import json
import math

import numpy as np
import pytest
from scipy.fft import rfft, rfftfreq

from analysis.analyze_signals import analyze_signals
from biomarkers.signal_metrics import spectral_snr
from biomarkers.signal_quality import assess_signal_quality
from config import HR_HIGH_HZ, HR_LOW_HZ, ROI_NAMES
from extractors.chrom import chrom_window_length
from extractors.combine import combine_roi_and_methods

FPS = 8.0
N_FRAMES = 240
HR_HZ = 1.2  # 72 bpm


def _synthetic_roi_signals(n_frames=N_FRAMES, fps=FPS, hr_hz=HR_HZ, seed=0):
    """Senoide de 1.2 Hz sobre uma base RGB plausível, nas três ROIs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / fps
    pulse = np.sin(2 * np.pi * hr_hz * t)

    signals = {}
    for offset, roi in enumerate(ROI_NAMES):
        base = np.array([160.0, 120.0, 110.0]) + offset
        # O pulso modula sobretudo o verde, como num sinal real.
        gains = np.array([0.6, 2.0, 0.4])
        frames = base + np.outer(pulse, gains)
        frames += rng.normal(0.0, 0.05, size=frames.shape)
        signals[roi] = frames.tolist()
    return signals


# 1. Sinal sintético, sem servidor.
def test_hr_de_sinal_sintetico_a_8_fps():
    result = analyze_signals(FPS, _synthetic_roi_signals(), N_FRAMES)
    assert result["hr_bpm"] == pytest.approx(72.0, abs=2.0)


def test_chrom_cresce_a_janela_a_8_fps():
    # Seção 4.1: a 8 fps a janela de 1.6 s (14) não passa no padlen do
    # filtfilt de ordem 3; tem que crescer para 24 (3 s).
    assert chrom_window_length(8.0) == 24
    assert chrom_window_length(24.0) == 40  # acima de ~14 fps nada muda


# 2. json.dumps não levanta e não emite Infinity/NaN.
def test_resultado_serializa_em_json_valido():
    result = analyze_signals(FPS, _synthetic_roi_signals(), N_FRAMES)
    encoded = json.dumps(result)
    assert "Infinity" not in encoded
    assert "NaN" not in encoded
    assert "hrv" in result  # é o hrv que carrega os round(np.float64)


# 3. Pisos rígidos.
def test_poucos_frames_levanta_value_error():
    signals = _synthetic_roi_signals(n_frames=20)
    with pytest.raises(ValueError, match="Refaça o escaneamento"):
        analyze_signals(FPS, signals, 20)


def test_fps_baixo_levanta_value_error():
    signals = _synthetic_roi_signals()
    with pytest.raises(ValueError, match="Taxa de captura baixa demais"):
        analyze_signals(3.0, signals, N_FRAMES)


def test_roi_ausente_levanta_value_error():
    signals = _synthetic_roi_signals()
    del signals["testa"]
    with pytest.raises(ValueError, match="ROI ausente"):
        analyze_signals(FPS, signals, N_FRAMES)


# 4. Os três níveis, direto em assess_signal_quality.
def test_nivel_boa():
    q = assess_signal_quality(fps=8, valid_ratio=0.95, snr=4.0, hr_bpm=72.0)
    assert q["nivel"] == "boa"
    assert q["avisos"] == []
    assert q["confiavel"] is True


def test_nivel_media():
    q = assess_signal_quality(fps=8, valid_ratio=0.55, snr=4.0, hr_bpm=72.0)
    assert q["nivel"] == "media"
    # O critério 4 da Seção 12 afirma confiavel: true aqui, mas o
    # rppg_core.py de produção amarra confiavel a nivel == "boa" — e a
    # Seção 7 manda usar a versão de produção quando as duas divergem.
    assert q["confiavel"] is False


def test_nivel_ruim():
    q = assess_signal_quality(fps=8, valid_ratio=0.30, snr=1.0, hr_bpm=72.0)
    assert q["nivel"] == "ruim"
    assert q["confiavel"] is False


def _spectral_snr_referencia(filtered, fps):
    """Cópia literal do spectral_snr do rppg_core.py que roda no Render.

    Fica aqui em vez de importar o arquivo original de propósito: é o
    oráculo do teste abaixo e precisa continuar existindo mesmo depois que
    o backend antigo sumir do disco.
    """
    n = len(filtered)
    freqs = rfftfreq(n, d=1.0 / fps)
    fft_vals = np.abs(rfft(filtered * np.hanning(n)))
    fft_valid = fft_vals[(freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)]
    if len(fft_valid) == 0:
        return 0.0
    media = float(np.mean(fft_valid))
    return 0.0 if media <= 0 else float(np.max(fft_valid) / media)


# O SNR tem que ficar na escala de produção (razão linear), não em dB.
def test_spectral_snr_bate_com_o_de_producao():
    """Trava a escala do SNR contra a implementação que roda no Render.

    Os limiares SNR_BOA/SNR_MEDIA são calibrados nessa razão pico/média
    linear. Se alguém trocar por compute_signal_metrics()["snr"] (dB), este
    teste quebra — que é exatamente o ponto.
    """
    signals = {roi: np.asarray(v, dtype=np.float64)
               for roi, v in _synthetic_roi_signals().items()}
    filtered = combine_roi_and_methods(signals, FPS)

    assert spectral_snr(filtered, FPS) == pytest.approx(
        _spectral_snr_referencia(filtered, FPS), rel=1e-9)


def test_snr_do_contrato_esta_na_escala_linear():
    # Um pulso limpo dá razão na casa das dezenas; em dB daria ~15-18.
    result = analyze_signals(FPS, _synthetic_roi_signals(), N_FRAMES)
    assert result["qualidade"]["snr"] > 25.0


def test_confiavel_segue_producao():
    # Produção: confiavel = (nivel == "boa"). "media" NÃO é confiável.
    assert assess_signal_quality(8, 0.95, 4.0, 72.0)["confiavel"] is True
    assert assess_signal_quality(8, 0.55, 4.0, 72.0)["confiavel"] is False
    assert assess_signal_quality(8, 0.30, 1.0, 72.0)["confiavel"] is False


def test_snr_infinito_vira_numero_finito():
    q = assess_signal_quality(fps=8, valid_ratio=0.95, snr=math.inf, hr_bpm=72.0)
    assert q["snr"] == 99.0
    json.dumps(q)


def test_hr_implausivel_so_gera_aviso():
    q = assess_signal_quality(fps=8, valid_ratio=0.95, snr=4.0, hr_bpm=200.0)
    assert q["nivel"] == "boa"
    assert len(q["avisos"]) == 1


# 5. Formato do contrato.
def test_chaves_do_contrato():
    result = analyze_signals(FPS, _synthetic_roi_signals(), N_FRAMES)
    assert {"hr_bpm", "qualidade", "n_frames", "n_frames_recebidos",
            "fps_estimado"} <= set(result)
    assert set(result["qualidade"]) == {
        "nivel", "confiavel", "avisos", "fps", "frames_validos_pct",
        "snr", "max_bpm_detectavel",
    }
    assert result["qualidade"]["nivel"] in ("ruim", "media", "boa")
    # Seção 4.2: teto real da análise, 0.9 * Nyquist a 8 fps.
    assert result["qualidade"]["max_bpm_detectavel"] == 216


def test_captura_degradada_nao_vira_excecao():
    # 240 recebidos, 200 válidos: valid_ratio abaixo do piso de "boa",
    # mas ainda assim sai um hr_bpm com ressalvas.
    signals = _synthetic_roi_signals(n_frames=200)
    result = analyze_signals(FPS, signals, 240)
    assert result["hr_bpm"] > 0
    assert result["qualidade"]["frames_validos_pct"] == pytest.approx(83.3, abs=0.1)
