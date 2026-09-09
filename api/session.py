"""Estado por conexão: acumula os frames e mede o fps real de chegada."""

import time

from config import ROI_NAMES


class ScanSession:
    """Espelha o ScanSession do backend antigo.

    O fps é medido no servidor, pelos instantes de chegada das mensagens —
    é a única fonte possível, já que o cliente não informa fps e não pode
    ser alterado.
    """

    def __init__(self):
        self.roi_signals = {roi: [] for roi in ROI_NAMES}
        self.timestamps = []
        self.n_frames_recebidos = 0

    def process_frame_means(self, valid, means):
        """Registra um frame. Inválidos contam no total, mas não no sinal."""
        self.n_frames_recebidos += 1
        if not valid or not means:
            return
        if any(roi not in means for roi in ROI_NAMES):
            return
        for roi in ROI_NAMES:
            self.roi_signals[roi].append([float(c) for c in means[roi][:3]])
        self.timestamps.append(time.monotonic())

    @property
    def n_frames(self):
        return len(self.timestamps)

    def estimate_fps(self):
        """(n_validos - 1) / (t_ultimo - t_primeiro), sobre os frames aceitos.

        Mesma definição do backend antigo: mudar isso desloca todos os
        limiares de qualidade.
        """
        if self.n_frames < 2:
            return 0.0
        span = self.timestamps[-1] - self.timestamps[0]
        return (self.n_frames - 1) / span if span > 0 else 0.0
