"""Configuração do serviço de rPPG."""

# --- Banda cardíaca -------------------------------------------------------
# Valores que a matemática já usava de fato (estavam hardcoded em três
# módulos). Agora são a única fonte da verdade.
HR_LOW_HZ = 0.7    # 42 bpm
HR_HIGH_HZ = 4.0   # 240 bpm
BAND_MAX_NYQUIST_RATIO = 0.9   # ver band_edges(), Seção 4.2

# --- Fusão ----------------------------------------------------------------
METHOD_WEIGHTS = {"chrom": 0.3, "pos": 0.3, "ica": 0.3, "green": 0.1}
ROI_WEIGHTS = {"testa": 0.4, "bochecha_esquerda": 0.3, "bochecha_direita": 0.3}

# Só as CHAVES importam: as ROIs são recortadas no navegador
# (apps/scan-site/src/lib/faceScanner.js). Precisam bater com ROI_WEIGHTS
# e com o que o frontend envia.
ROI_NAMES = ("testa", "bochecha_esquerda", "bochecha_direita")

# --- Janelas dos extratores ----------------------------------------------
CHROM_WIN_SEC = 1.6
CHROM_LOW_HZ, CHROM_HIGH_HZ = 0.7, 2.5
POS_WIN_SEC = 1.6
POS_LOW_HZ, POS_HIGH_HZ = 0.75, 3.0
ICA_LOW_HZ, ICA_HIGH_HZ = 0.7, 2.5
SMOOTH_WINDOW = 3

# --- Pisos rígidos (os únicos que geram erro) -----------------------------
MIN_VALID_FRAMES = 28   # padlen do bandpass_filter de ordem 4
ABSOLUTE_MIN_FPS = 4.0  # ver Seção 4.4

# --- Limiares de qualidade (Seção 7) --------------------------------------
# COPIADOS LITERALMENTE do rppg_core.py em produção. Não são reconstrução:
# são os valores que o painel do médico já usa hoje. Não mexa neles como
# efeito colateral de outra mudança.
#
# Pisos por critério. "boa" exige todos os de cima; qualquer um violado
# rebaixa pra "media", e os limites inferiores rebaixam pra "ruim".
FPS_BOA = 6.0            # 30*6 = 180 bpm detectáveis, cobre folgado o repouso
FPS_MEDIA = 4.0          # 120 bpm; acima disso a taquicardia já faz alias
VALID_RATIO_BOA = 0.70   # % de frames com o rosto na área
VALID_RATIO_MEDIA = 0.40
SNR_BOA = 3.0
SNR_MEDIA = 2.0

NIVEIS = ("ruim", "media", "boa")

# Só informativo: gera um aviso, mas NÃO entra na decisão do nível (o
# rppg_core.py decide só por fps, valid_ratio e snr).
HR_PLAUSIVEL_BPM = (40.0, 180.0)
