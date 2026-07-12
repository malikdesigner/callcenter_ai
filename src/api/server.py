"""
FastAPI server — WebSocket endpoint for real-time call audio,
REST endpoints for appointment management and dashboard data.
"""

import json
import os
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.appointment.models import Appointment, Doctor, Department, DoctorSlot, create_tables, engine
from src.appointment.booking import BookingSystem, resolve_department
from src.pipeline.call_handler import CallHandler
from src.api.twilio_handler import router as twilio_router
from sqlmodel import Session, select
from datetime import datetime, date

_active_calls: Dict[str, CallHandler] = {}
_booking = BookingSystem()


def create_app(language: str = None) -> FastAPI:
    """
    Factory that creates a FastAPI app configured for the given language.
    Running two instances in the same process means heavy models
    (Whisper, VAD, TTS) are loaded exactly once via CallHandler class-level state.
    """
    lang = language or os.getenv("APP_LANGUAGE", "en")
    logger.info(f"Server starting in {lang.upper()} mode")

    new_app = FastAPI(title="AI Hospital Receptionist", version="1.0.0")
    new_app.mount("/static", StaticFiles(directory="static"), name="static")
    new_app.include_router(twilio_router)

    # ── Startup ────────────────────────────────────────────────────────────────

    @new_app.on_event("startup")
    async def startup():
        create_tables()
        logger.info("Database ready.")
        CallHandler.load_models()

    # ── Static pages ───────────────────────────────────────────────────────────

    @new_app.get("/")
    async def root():
        if lang == "ur":
            return FileResponse("static/index_ur.html")
        if lang == "ro":
            return FileResponse("static/index_ro.html")
        return FileResponse("static/index.html")

    @new_app.get("/urdu")
    async def urdu_redirect():
        return FileResponse("static/index_ur.html")

    @new_app.get("/dashboard")
    async def dashboard():
        return FileResponse("static/dashboard.html")

    # ── WebSocket call endpoint ────────────────────────────────────────────────

    @new_app.websocket("/ws/call/{session_id}")
    async def call_ws(websocket: WebSocket, session_id: str):
        await websocket.accept()
        logger.info(f"[WS] Connected ({lang}): {session_id}")

        handler = CallHandler(session_id=session_id, language=lang)
        _active_calls[session_id] = handler

        async def send(item):
            kind, payload = item
            if kind == "audio":
                await websocket.send_bytes(payload)
            elif kind == "json":
                await websocket.send_json(payload)

        try:
            await handler.start_call()

            async def receive_loop():
                while handler.is_active:
                    try:
                        data = await websocket.receive()
                    except WebSocketDisconnect:
                        break
                    except RuntimeError: # usually when connection is closed
                        break

                    if "bytes" in data:
                        await handler.process_audio_chunk(data["bytes"])
                    elif "text" in data:
                        try:
                            msg = json.loads(data["text"])
                            if msg.get("type") == "end_call":
                                break
                        except Exception:
                            pass
                await handler.end_call()

            async def send_loop():
                async for item in handler.consume_responses():
                    kind, payload = item
                    try:
                        if kind == "audio":
                            await websocket.send_bytes(payload)
                        elif kind == "json":
                            await websocket.send_json(payload)
                    except Exception:
                        break

            import asyncio
            await asyncio.gather(receive_loop(), send_loop())

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

    # ── REST: Appointments ─────────────────────────────────────────────────────

    @new_app.get("/api/appointments")
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
                "transcript": a.transcript,
                "status": a.status,
                "created_at": str(a.created_at),
            }
            for a in rows
        ]

    @new_app.get("/api/appointments/{appt_id}")
    def get_appointment(appt_id: int):
        with Session(engine) as session:
            appt = session.get(Appointment, appt_id)
        if not appt:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return appt

    @new_app.patch("/api/appointments/{appt_id}/cancel")
    def cancel_appointment(appt_id: int):
        with Session(engine) as session:
            appt = session.get(Appointment, appt_id)
            if not appt:
                raise HTTPException(status_code=404, detail="Not found")
            appt.status = "cancelled"
            session.add(appt)
            session.commit()
        return {"message": f"Appointment #{appt_id} cancelled"}

    @new_app.get("/api/stats")
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

    # ── REST: Departments ──────────────────────────────────────────────────────

    @new_app.get("/api/departments")
    def list_departments():
        with Session(engine) as session:
            return session.exec(select(Department)).all()

    @new_app.post("/api/departments")
    def create_department(dept: Department):
        with Session(engine) as session:
            existing = session.exec(select(Department).where(Department.name == dept.name)).first()
            if existing:
                return existing
            session.add(dept)
            session.commit()
            session.refresh(dept)
        return dept

    @new_app.patch("/api/departments/{dept_id}")
    def update_department(dept_id: int, dept_data: Department):
        with Session(engine) as session:
            dept = session.get(Department, dept_id)
            if not dept:
                raise HTTPException(status_code=404, detail="Not found")
            dept.name = dept_data.name
            dept.description = dept_data.description
            session.add(dept)
            session.commit()
            session.refresh(dept)
        return dept

    @new_app.delete("/api/departments/{dept_id}")
    def delete_department(dept_id: int):
        with Session(engine) as session:
            dept = session.get(Department, dept_id)
            if not dept:
                raise HTTPException(status_code=404, detail="Not found")
            session.delete(dept)
            session.commit()
        return {"message": "Department deleted"}

    # ── REST: Doctors ──────────────────────────────────────────────────────────

    @new_app.get("/api/doctors")
    def list_doctors():
        with Session(engine) as session:
            return session.exec(select(Doctor)).all()

    @new_app.post("/api/doctors")
    def create_doctor(doc: Doctor):
        with Session(engine) as session:
            session.add(doc)
            session.commit()
            session.refresh(doc)
        return doc

    @new_app.patch("/api/doctors/{doc_id}")
    def update_doctor(doc_id: int, doc_data: Doctor):
        with Session(engine) as session:
            doc = session.get(Doctor, doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Not found")
            doc.name = doc_data.name
            doc.department_name = doc_data.department_name
            doc.specialty = doc_data.specialty
            doc.is_active = doc_data.is_active
            doc.fee = doc_data.fee
            session.add(doc)
            session.commit()
            session.refresh(doc)
        return doc

    @new_app.delete("/api/doctors/{doc_id}")
    def delete_doctor(doc_id: int):
        with Session(engine) as session:
            doc = session.get(Doctor, doc_id)
            if not doc:
                raise HTTPException(status_code=404, detail="Doctor not found")
            session.delete(doc)
            session.commit()
        return {"message": "Doctor deleted"}

    # ── REST: Slots ────────────────────────────────────────────────────────────

    @new_app.get("/api/slots")
    def list_slots(doctor: str = None):
        with Session(engine) as session:
            stmt = select(DoctorSlot)
            if doctor:
                stmt = stmt.where(DoctorSlot.doctor_name == doctor)
            return session.exec(stmt).all()

    @new_app.post("/api/slots")
    def create_slot(slot: DoctorSlot):
        def fmt(t):
            if not t or " " in t:
                return t
            try:
                return datetime.strptime(t, "%H:%M").strftime("%I:%M %p")
            except Exception:
                return t
        slot.start_time = fmt(slot.start_time)
        slot.end_time = fmt(slot.end_time)
        with Session(engine) as session:
            session.add(slot)
            session.commit()
            session.refresh(slot)
        return slot

    @new_app.patch("/api/slots/{slot_id}")
    def update_slot(slot_id: int, slot_data: DoctorSlot):
        def fmt(t):
            if not t or " " in t:
                return t
            try:
                return datetime.strptime(t, "%H:%M").strftime("%I:%M %p")
            except Exception:
                return t
        with Session(engine) as session:
            slot = session.get(DoctorSlot, slot_id)
            if not slot:
                raise HTTPException(status_code=404, detail="Slot not found")
            slot.doctor_name = slot_data.doctor_name
            slot.start_time = fmt(slot_data.start_time)
            slot.end_time = fmt(slot_data.end_time)
            session.add(slot)
            session.commit()
            session.refresh(slot)
        return slot

    @new_app.delete("/api/slots/{slot_id}")
    def delete_slot(slot_id: int):
        with Session(engine) as session:
            slot = session.get(DoctorSlot, slot_id)
            if not slot:
                raise HTTPException(status_code=404, detail="Slot not found")
            session.delete(slot)
            session.commit()
        return {"message": "Slot deleted"}

    # ── REST: Hospital Knowledge (timings, fees, policies) ────────────────────

    _KNOWLEDGE_PATH = "data/hospital_knowledge.json"

    @new_app.get("/api/knowledge")
    def get_knowledge():
        try:
            with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Knowledge file not found")

    @new_app.patch("/api/knowledge")
    def update_knowledge(payload: dict):
        try:
            with open(_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                knowledge = json.load(f)
            # Deep-merge only the sections passed in payload
            for section, value in payload.items():
                if isinstance(value, dict) and isinstance(knowledge.get(section), dict):
                    knowledge[section].update(value)
                else:
                    knowledge[section] = value
            with open(_KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
                json.dump(knowledge, f, ensure_ascii=False, indent=2)
            # Invalidate cached knowledge so agent picks up changes immediately
            import src.llm.agent as _agent_mod
            _agent_mod._hospital_knowledge = None
            return {"message": "Knowledge updated"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # ── REST: Manual Booking ───────────────────────────────────────────────────

    @new_app.get("/api/booking/available-slots")
    def get_available_slots(doctor: str, date_str: str):
        try:
            target_date = date.fromisoformat(date_str)
            return _booking.get_available_slots(doctor, target_date)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @new_app.post("/api/booking")
    def create_manual_booking(payload: dict):
        try:
            appt = _booking.book_appointment(
                patient_name=payload['patient_name'],
                patient_phone=payload['patient_phone'],
                doctor_name=payload['doctor'],
                department=payload['department'],
                appointment_date=date.fromisoformat(payload['date']),
                appointment_time=payload['time'],
                reason=payload.get('reason', ''),
                transcript=None,
            )
            return {"status": "success", "appointment_id": appt.id}
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"[API] Manual booking error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    return new_app


# ── Backward-compat: allow `uvicorn src.api.server:app` directly ──────────────
app = create_app()
