# rPPG — Residência Pipos

Serviço WebSocket de fotopletismografia remota (rPPG). Recebe, quadro a quadro,
as **médias RGB de três ROIs faciais já extraídas pelo navegador** e devolve a
**frequência cardíaca** estimada mais um objeto de **qualidade da captura**.

Substitui o backend anterior do Render (`main.py` + `rppg_core.py`), mantendo o
protocolo do WebSocket — o frontend do paciente (`apps/scan-site/`) não muda,
exceto pela URL do serviço (`VITE_SCAN_WS_URL`).

> O servidor **nunca vê imagem**: não roda MediaPipe nem OpenCV. A detecção
> facial e o recorte das ROIs acontecem inteiramente no navegador
> (`apps/scan-site/src/lib/faceScanner.js`). O CLI de pesquisa que analisava
> arquivos `.mp4` foi removido nesta fase.

---

## Como funciona

1. **O navegador envia** as médias R, G, B de `testa`, `bochecha_esquerda` e
   `bochecha_direita`, a ~8 quadros/s, por 30 s.
2. **O servidor mede o fps real** pelos instantes de chegada das mensagens
   (`api/session.py`).
3. **Pré-processamento** — média móvel por canal (`preprocessing/`).
4. **Extração do sinal rPPG** — quatro algoritmos por ROI:
   - **CHROM** (de Haan & Jeanne, 2013)
   - **POS** (Wang et al., 2017)
   - **GREEN** (Verkruysse et al., 2008)
   - **ICA** (Poh et al., 2010)
5. **Combinação** — z-score, alinhamento de polaridade e média ponderada,
   primeiro entre algoritmos (`METHOD_WEIGHTS`) e depois entre as ROIs
   (`ROI_WEIGHTS`).
6. **Filtragem final** — Butterworth passa-banda na faixa cardíaca, com o topo
   travado em 90% do Nyquist (216 bpm a 8 fps).
7. **Biomarcadores** — FC pelo pico da FFT, SNR espectral, HRV (SDNN, RMSSD,
   pNN50).
8. **Qualidade** — `biomarkers/signal_quality.py` classifica a captura em
   `boa` / `media` / `ruim` a partir de fps, proporção de quadros válidos e SNR.

---

## Contrato do WebSocket

Rota: `/ws/scan`. **Abrir a conexão já é o "start"** — não existe mensagem de
início.

Cliente → servidor, a cada quadro:

```json
{"type": "frame_data", "valid": true,
 "means": {"testa": [r, g, b],
           "bochecha_esquerda": [r, g, b],
           "bochecha_direita": [r, g, b]}}
```

Quadros sem rosto chegam como `{"type": "frame_data", "valid": false}` (sem
`means`). Ao fim dos 30 s o cliente envia `{"type": "stop"}` e espera a
resposta.

Servidor → cliente, uma única vez:

```json
{
  "type": "result",
  "hr_bpm": 72.3,
  "qualidade": {
    "nivel": "boa",
    "confiavel": true,
    "avisos": [],
    "fps": 8.02,
    "frames_validos_pct": 96.4,
    "snr": 3.8,
    "max_bpm_detectavel": 216
  },
  "n_frames": 231,
  "n_frames_recebidos": 240,
  "fps_estimado": 8.02,
  "hrv": {"SDNN_ms": 48.1, "RMSSD_ms": 39.7, "pNN50_%": 12.5,
          "n_batimentos_detectados": 36}
}
```

Em caso de erro: `{"type": "error", "message": "<texto em português>"}`.

O cliente atual lê apenas `hr_bpm` e `qualidade` (este último gravado inteiro no
Firestore); `hrv` e os campos de diagnóstico são devolvidos mas descartados por
ele hoje. O HRV fica no retorno de propósito: custa um `find_peaks` sobre o
sinal já filtrado, e no dia em que o frontend puder repassá-lo o campo já
está lá.

Há também `GET /health`, usado só para um ping externo que evite o cold start do
Render.

---

## Instalação e execução local

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Para apontar o app do paciente ao servidor local:
`VITE_SCAN_WS_URL=ws://localhost:8000/ws/scan`.

Testes:

```bash
pip install pytest
pytest
```

---

## Deploy (Render)

- **Build:** `pip install -r requirements.txt`
- **Start:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- Manter a mesma URL (`residencia-api.onrender.com`): o frontend a tem
  hardcoded como fallback de produção.

`numpy<2` é obrigatório: `ica.py` e `pos.py` usam `np.asmatrix` e aritmética de
`np.matrix`, sem testes sob NumPy 2.x.

---

## Configuração

Tudo em [config.py](config.py): banda cardíaca, pesos de fusão e de ROI, janelas
dos extratores, pisos rígidos e limiares de qualidade.

Dois pontos que costumam confundir:

- **`ROI_NAMES` é só um contrato de nomes.** A geometria das ROIs mora no
  frontend; ajustar índices de landmark aqui não teria efeito nenhum.
- **Os limiares de qualidade** (`FPS_BOA`, `VALID_RATIO_BOA`, `SNR_BOA`, …) são
  os valores de produção que o painel do médico já usa. Não os mexa como efeito
  colateral de outra mudança.

---

## Arquitetura do projeto

```
rPPG/
├── api/
│   ├── main.py                 # FastAPI + CORS + /health + rota /ws/scan
│   ├── ws.py                   # handler do WebSocket
│   └── session.py              # ScanSession: acumula frames e mede fps
│
├── analysis/
│   └── analyze_signals.py      # função pura: (fps, roi_signals, n) -> dict
│
├── extractors/
│   ├── chrom.py  pos.py  green.py  ica.py
│   └── combine.py              # fusão método/ROI
│
├── preprocessing/
│   ├── filters.py              # bandpass_filter, band_edges, média móvel
│   ├── detrend.py
│   └── processing.py
│
├── biomarkers/
│   ├── heart_rate.py           # FC pelo pico da FFT
│   ├── hrv.py                  # SDNN, RMSSD, pNN50
│   ├── signal_metrics.py       # SNR espectral e afins
│   └── signal_quality.py       # o objeto `qualidade` do contrato
│
├── config.py
├── requirements.txt
└── tests/
    └── test_analyze_signals.py
```

`analysis/analyze_signals.py` é pura e não sabe nada de transporte — é o que
tornaria uma eventual migração para REST pequena: só `api/` mudaria.

---

## Duas mudanças de comportamento em relação ao backend antigo

1. **A janela do CHROM cresce a 8 fps** (1.6 s → 3 s). A de 1.6 s tem 14
   amostras e o `filtfilt` de ordem 3 exige mais de 21 — a 8 fps ela levantava
   exceção. Acima de ~14 fps nada muda.
2. **O teto da banda passa de 0.99 para 0.9 do Nyquist** (≈237 → 216 bpm a
   8 fps), tirando o filtro da região numericamente instável.
   `max_bpm_detectavel` passa a reportar esse teto real.

---

## Referências

- de Haan, G., & Jeanne, V. (2013). *Robust pulse rate from chrominance-based rPPG*. IEEE Transactions on Biomedical Engineering, 60(10), 2878–2886.
- Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017). *Algorithmic principles of remote PPG*. IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491.
- Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). *Remote plethysmographic imaging using ambient light*. Optics Express, 16(26), 21434–21445.
- Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). *Non-contact, automated cardiac pulse measurements using video imaging and blind source separation*. Optics Express, 18(10), 10762–10774.
