"""Classificação de confiabilidade da captura (campo `qualidade` do contrato)."""

import math

from config import (
    FPS_BOA, FPS_MEDIA,
    HR_PLAUSIVEL_BPM,
    SNR_BOA, SNR_MEDIA,
    VALID_RATIO_BOA, VALID_RATIO_MEDIA,
)
from preprocessing.filters import effective_band_hz


def assess_signal_quality(fps, valid_ratio, snr, hr_bpm):
    """Classifica a captura sem nunca rejeitá-la.

    "boa" exige os três pisos de cima; qualquer um violado rebaixa para
    "media", e violar os pisos de baixo rebaixa para "ruim". Captura ruim
    não vira erro — vira nivel "ruim" com avisos. Os dois pisos que
    realmente barram estão em analyze_signals().

    `snr` tem que vir de `spectral_snr` (razão pico/média linear): é a
    escala em que SNR_BOA e SNR_MEDIA foram calibrados em produção.
    """
    snr = _finite(snr)

    boa = fps >= FPS_BOA and valid_ratio >= VALID_RATIO_BOA and snr >= SNR_BOA
    media = fps >= FPS_MEDIA and valid_ratio >= VALID_RATIO_MEDIA and snr >= SNR_MEDIA
    nivel = "boa" if boa else ("media" if media else "ruim")

    avisos = []
    if fps < FPS_BOA:
        avisos.append(
            "A câmera entregou menos quadros por segundo do que o esperado."
        )
    if valid_ratio < VALID_RATIO_BOA:
        avisos.append(
            "O rosto saiu da área indicada em parte do escaneamento."
        )
    if snr < SNR_BOA:
        avisos.append(
            "O sinal ficou fraco em relação ao ruído — luz fraca ou movimento."
        )
    # Informativo apenas: não participa da decisão do nível.
    if not (HR_PLAUSIVEL_BPM[0] <= hr_bpm <= HR_PLAUSIVEL_BPM[1]):
        avisos.append(
            "O valor estimado está fora da faixa típica de repouso."
        )

    _, high_hz = effective_band_hz(fps)
    return {
        "nivel": nivel,
        # Como no rppg_core.py que rodava no Render: só "boa" é confiável.
        # Nenhuma tela lê este campo hoje (o painel do médico e a Tela 6
        # usam `nivel` e `avisos`), mas ele vai inteiro pro Firestore.
        "confiavel": nivel == "boa",
        "avisos": avisos,
        "fps": round(float(fps), 2),
        "frames_validos_pct": round(float(valid_ratio) * 100.0, 1),
        "snr": round(snr, 2),
        "max_bpm_detectavel": int(round(high_hz * 60.0)),
    }


def _finite(value):
    """Rede de segurança: JSON não representa inf nem nan (ver Seção 10).

    `spectral_snr` já devolve 0.0 nos casos degenerados, então na prática
    isto não dispara — mas o campo vai direto pro Firestore e um Infinity
    quebraria o JSON.parse do navegador, deixando a tela presa em
    "calculando". Custa uma comparação.
    """
    value = float(value)
    return 99.0 if math.isinf(value) else (0.0 if math.isnan(value) else value)
