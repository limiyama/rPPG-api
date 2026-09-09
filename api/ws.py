"""Rota /ws/scan — o protocolo é o do frontend e não pode mudar (Seção 1)."""

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from analysis.analyze_signals import analyze_signals
from api.session import ScanSession

logger = logging.getLogger(__name__)
router = APIRouter()

MENSAGEM_ERRO_INTERNO = (
    "Não foi possível calcular o resultado do escaneamento. "
    "Tente escanear novamente."
)


@router.websocket("/ws/scan")
async def scan(websocket: WebSocket):
    await websocket.accept()
    session = ScanSession()
    try:
        while True:
            msg = await websocket.receive_json()
            tipo = msg.get("type")

            if tipo == "frame_data":
                session.process_frame_means(msg.get("valid"), msg.get("means"))
                continue

            if tipo == "stop":
                await _send_result(websocket, session)
                return
            # Qualquer outro tipo é ignorado — o cliente atual não envia outros.
    except WebSocketDisconnect:
        # O paciente fechou a aba ou a rede caiu no meio: nada a fazer.
        return


async def _send_result(websocket, session):
    try:
        result = analyze_signals(
            fps=session.estimate_fps(),
            roi_signals=session.roi_signals,
            n_frames_recebidos=session.n_frames_recebidos,
        )
    except ValueError as error:
        # Piso rígido atingido: a mensagem já é escrita para o paciente ler.
        await websocket.send_json({"type": "error", "message": str(error)})
        return
    except Exception:
        # Falha inesperada da matemática: loga o detalhe, mostra texto genérico.
        logger.exception("Falha ao analisar a sessão de rPPG")
        await websocket.send_json({"type": "error", "message": MENSAGEM_ERRO_INTERNO})
        return

    await websocket.send_json({"type": "result", **result})
