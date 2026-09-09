"""CHROM
de Haan, G., & Jeanne, V. (2013).
Robust pulse rate from chrominance-based rPPG.
IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.
"""

import math

import numpy as np
from scipy import signal

from config import CHROM_HIGH_HZ, CHROM_LOW_HZ, CHROM_WIN_SEC
from preprocessing.filters import band_edges
from preprocessing.processing import _process_video

CHROM_FILTER_ORDER = 3


def chrom_window_length(fps, order=CHROM_FILTER_ORDER, win_sec=CHROM_WIN_SEC):
    """Tamanho de janela (par) que o filtfilt interno do CHROM aceita.

    A 8 fps a janela de 1.6 s tem 14 amostras e o filtfilt de ordem 3 exige
    mais de 21. Em vez de quebrar, a janela cresce até o mínimo viável — a
    3 s, a 8 fps. Acima de ~14 fps o valor de win_sec volta a mandar e o
    comportamento fica idêntico ao de hoje.
    """
    padlen = 3 * (2 * order + 1)
    win_l = math.ceil(win_sec * fps)
    if win_l <= padlen:
        win_l = padlen + 2
    if win_l % 2:
        win_l += 1
    return win_l


def chrom_algorithm(frames, FS):
    RGB = _process_video(frames)
    FN = RGB.shape[0]
    low, high = band_edges(FS, CHROM_LOW_HZ, CHROM_HIGH_HZ)
    B, A = signal.butter(CHROM_FILTER_ORDER, [low, high], 'bandpass')

    WinL = chrom_window_length(FS)
    if FN < WinL:
        raise ValueError(
            f"CHROM precisa de ao menos {WinL} frames a {FS:.1f} fps, recebeu {FN}."
        )

    NWin = math.floor((FN-WinL//2)/(WinL//2))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))

        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    BVP = S
    return BVP
