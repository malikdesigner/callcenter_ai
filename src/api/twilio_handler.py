"""
Twilio Media Streams integration for real phone call support.

Call flow:
  1. Someone dials your Twilio number
  2. Twilio POSTs to /twilio/voice  → we return TwiML to open a Media Stream
  3. Twilio opens WebSocket to /twilio/stream
  4. We stream audio bidirectionally in G.711 μ-law (8kHz mono)
  5. CallHandler (VAD → STT → LLM → TTS) runs exactly as it does for browser calls

Audio format chain:
  Inbound  (caller → us):  mulaw 8kHz  → linear16 → resample 16kHz → float32 → CallHandler
  Outbound (us → caller):  EdgeTTS MP3 → PCM 8kHz  → mulaw          → base64  → Twilio
"""

import asyncio
import base64
import io
import json
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from scipy.signal import resample_poly

from src.pipeline.call_handler import CallHandler

router = APIRouter()


# ── G.711 μ-law codec ─────────────────────────────────────────────────────────
# audioop is stdlib on Python ≤3.12; use a pure-numpy fallback on 3.13+.

try:
    import audioop as _audioop

    def _ulaw2lin(data: bytes) -> bytes:
        return _audioop.ulaw2lin(data, 2)

    def _lin2ulaw(data: bytes) -> bytes:
        return _audioop.lin2ulaw(data, 2)

except ImportError:
    def _ulaw2lin(data: bytes) -> bytes:
        u = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32)
        sign  = (u >> 7) & 1
        exp   = (u >> 4) & 7
        mant  = u & 0x0F
        linear = (((mant << 1) | 1) << (exp + 2)).astype(np.int32)
        linear = np.where(sign, -linear, linear)
        return linear.clip(-32768, 32767).astype(np.int16).tobytes()

    def _lin2ulaw(data: bytes) -> bytes:
        BIAS, CLIP = 132, 32635
        pcm  = np.frombuffer(data, dtype=np.int16).astype(np.int32)
        sign = (pcm < 0).astype(np.int32)
        pcm  = np.clip(np.abs(pcm) + BIAS, 0, CLIP)
        safe = np.where(pcm > 0, pcm, 1)
        exp  = np.clip(np.floor(np.log2(safe)).astype(np.int32) - 4, 0, 7)
        mant = (pcm >> (exp + 1)) & 0x0F
        return (~((sign << 7) | (exp << 4) | mant) & 0xFF).astype(np.uint8).tobytes()


# ── Audio conversion helpers ──────────────────────────────────────────────────

def mulaw8k_to_float32_16k(mulaw_bytes: bytes) -> bytes:
    """Twilio mulaw 8 kHz → float32 16 kHz PCM bytes (what CallHandler expects)."""
    linear16 = _ulaw2lin(mulaw_bytes)
    pcm = np.frombuffer(linear16, dtype=np.int16).astype(np.float32) / 32768.0
    pcm_16k = resample_poly(pcm, 2, 1)          # 8 kHz → 16 kHz
    return pcm_16k.astype(np.float32).tobytes()


def mp3_to_mulaw8k(mp3_bytes: bytes) -> bytes:
    """EdgeTTS MP3 bytes → mulaw 8 kHz bytes (what Twilio expects)."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    seg = seg.set_channels(1).set_frame_rate(8000).set_sample_width(2)
    return _lin2ulaw(seg.raw_data)


def _chunk(data: bytes, size: int = 160) -> list:
    """Split mulaw into 20 ms chunks (160 bytes × 8000 Hz = 20 ms)."""
    return [data[i : i + size] for i in range(0, len(data), size)]


# ── TwiML webhook ─────────────────────────────────────────────────────────────

@router.post("/twilio/voice")
async def twilio_voice_webhook(request: Request):
    """
    Twilio calls this HTTP endpoint when someone dials your number.
    Returns TwiML that instructs Twilio to open a Media Stream WebSocket to us.

    Query params:
      lang=en  (default) | ur | bi
    """
    form = await request.form()
    lang = request.query_params.get("lang", "en")
    from_number = form.get("From", "unknown")
    call_sid    = form.get("CallSid", "unknown")
    logger.info(f"[Twilio] Incoming call from {from_number} | CallSid={call_sid} | lang={lang}")

    # Reconstruct WSS URL from the request host (ngrok sets x-forwarded-host)
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host", "localhost")
    )
    ws_url = f"wss://{host}/twilio/stream"

    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{ws_url}">
            <Parameter name="language" value="{lang}" />
        </Stream>
    </Connect>
</Response>"""
    return Response(content=twiml, media_type="text/xml")


# ── Media Stream WebSocket ────────────────────────────────────────────────────

@router.websocket("/twilio/stream")
async def twilio_media_stream(websocket: WebSocket):
    """
    Bidirectional Twilio Media Streams WebSocket handler.

    Twilio sends JSON frames:
      {"event":"connected"}
      {"event":"start","start":{"streamSid":"...","customParameters":{"language":"en"}}}
      {"event":"media","media":{"payload":"<base64 mulaw>"}}
      {"event":"stop"}

    We send audio back:
      {"event":"media","streamSid":"...","media":{"payload":"<base64 mulaw>"}}
    """
    await websocket.accept()

    session_id: str        = uuid.uuid4().hex[:8]
    stream_sid: Optional[str] = None
    handler:    Optional[CallHandler] = None

    logger.info(f"[Twilio] WS connected | session={session_id}")

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _send_audio(mp3_bytes: bytes):
        """Convert MP3 → mulaw chunks and push to Twilio."""
        if not mp3_bytes or not stream_sid:
            return
        try:
            mulaw = mp3_to_mulaw8k(mp3_bytes)
            for chunk in _chunk(mulaw):
                await websocket.send_text(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                }))
        except Exception as exc:
            logger.error(f"[Twilio] Audio send error: {exc}")

    # ── main event loop ───────────────────────────────────────────────────────

    try:
        async for raw in websocket.iter_text():
            msg   = json.loads(raw)
            event = msg.get("event")

            # ── 1. Stream started ─────────────────────────────────────────────
            if event == "start":
                start_data = msg["start"]
                stream_sid = start_data["streamSid"]
                params     = start_data.get("customParameters", {})
                language   = params.get("language", "en")
                logger.info(f"[Twilio] Stream start | sid={stream_sid} | lang={language}")

                handler = CallHandler(session_id=session_id, language=language, phone_mode=True)
                handler.__class__.load_models()

                # Play greeting — blocks briefly (1–3 s TTS), then we resume the loop
                async for kind, payload in handler.start_call():
                    if kind == "audio":
                        await _send_audio(payload)

            # ── 2. Inbound audio chunk ────────────────────────────────────────
            elif event == "media":
                if not handler or not handler.is_active:
                    continue

                mulaw_bytes = base64.b64decode(msg["media"]["payload"])
                pcm_bytes   = mulaw8k_to_float32_16k(mulaw_bytes)

                async for kind, payload in handler.process_audio_chunk(pcm_bytes):
                    if kind == "audio":
                        await _send_audio(payload)

                # End call if agent decided to hang up
                if not handler.is_active:
                    break

            # ── 3. Stream / call ended by Twilio ─────────────────────────────
            elif event == "stop":
                logger.info(f"[Twilio] Stream stopped | session={session_id}")
                break

    except WebSocketDisconnect:
        logger.info(f"[Twilio] WS disconnected | session={session_id}")
    except Exception as exc:
        logger.exception(f"[Twilio] Stream error | session={session_id}: {exc}")
    finally:
        if handler:
            await handler.end_call()
        logger.info(f"[Twilio] Session ended | session={session_id}")
