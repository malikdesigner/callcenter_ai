"""
FastAPI server — WebSocket endpoint for real-time call audio,
REST endpoints for appointment management and dashboard data.
"""

import json
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.appointment.models import Appointment, create_tables, engine
from src.pipeline.call_handler import CallHandler
from sqlmodel import Session, select

app = FastAPI(title="AI Hospital Receptionist", version="1.0.0")

_active_calls: Dict[str, CallHandler] = {}


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    create_tables()
    logger.info("Database ready.")
    CallHandler.load_models()


# ── Static files ───────────────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── WebSocket call endpoint ────────────────────────────────────────────────────

@app.websocket("/ws/call/{session_id}")
async def call_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info(f"[WS] Connected: {session_id}")

    handler = CallHandler(session_id=session_id)
    _active_calls[session_id] = handler

    async def send(item):
        """Dispatch a (type, payload) tuple to the browser."""
        kind, payload = item
        if kind == "audio":
            await websocket.send_bytes(payload)
        elif kind == "json":
            await websocket.send_json(payload)

    try:
        # Stream greeting sentences as soon as they're ready
        async for chunk in handler.start_call():
            kind, payload = chunk
            if kind == "audio":
                await websocket.send_bytes(payload)
            elif kind == "json":
                await websocket.send_json(payload)

        while handler.is_active:
            try:
                data = await websocket.receive()
            except WebSocketDisconnect:
                break

            if "bytes" in data:
                async for item in handler.process_audio_chunk(data["bytes"]):
                    await send(item)

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("type") == "end_call":
                    break

        try:
            await websocket.send_json({"type": "call_ended"})
        except Exception:
            pass

    except WebSocketDisconnect:
        logger.info(f"[WS] Disconnected: {session_id}")
    except Exception as e:
        logger.exception(f"[WS] Error in {session_id}: {e}")
    finally:
        await handler.end_call()
        _active_calls.pop(session_id, None)
        logger.info(f"[WS] Session cleaned up: {session_id}")


# ── REST: Appointments dashboard ───────────────────────────────────────────────

@app.get("/api/appointments")
def list_appointments():
    with Session(engine) as session:
        rows = session.exec(select(Appointment)).all()
    return [
        {
            "id": a.id,
            "patient_name": a.patient_name,
            "patient_phone": a.patient_phone,
            "doctor": a.doctor_name,
            "department": a.department,
            "date": str(a.appointment_date),
            "time": a.appointment_time,
            "reason": a.reason,
            "status": a.status,
            "created_at": str(a.created_at),
        }
        for a in rows
    ]


@app.get("/api/appointments/{appt_id}")
def get_appointment(appt_id: int):
    with Session(engine) as session:
        appt = session.get(Appointment, appt_id)
    if not appt:
        raise HTTPException(status_code=404, detail="Appointment not found")
    return appt


@app.patch("/api/appointments/{appt_id}/cancel")
def cancel_appointment(appt_id: int):
    with Session(engine) as session:
        appt = session.get(Appointment, appt_id)
        if not appt:
            raise HTTPException(status_code=404, detail="Not found")
        appt.status = "cancelled"
        session.add(appt)
        session.commit()
    return {"message": f"Appointment #{appt_id} cancelled"}


@app.get("/api/stats")
def stats():
    with Session(engine) as session:
        all_appts = session.exec(select(Appointment)).all()
    total = len(all_appts)
    confirmed = sum(1 for a in all_appts if a.status == "confirmed")
    cancelled = sum(1 for a in all_appts if a.status == "cancelled")
    return {
        "total_appointments": total,
        "confirmed": confirmed,
        "cancelled": cancelled,
        "active_calls": len(_active_calls),
    }
