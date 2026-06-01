"""
Hospital AI tool definitions and async executors.

The LLM calls these tools to get REAL data from the database instead of
inventing appointment times, doctors, or availability.

Tool call flow:
  1. LLM receives user message + tool definitions
  2. LLM returns a tool_call (name + args) instead of text
  3. Executor runs the tool against PostgreSQL/SQLite
  4. Result injected back as "tool" role message
  5. LLM generates the spoken response using real data
"""

import json
from datetime import date, time, datetime, timedelta
from typing import Any

from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Doctor, DoctorSchedule, Appointment, Patient, FAQ, Department


# ── Tool definitions (OpenAI function-calling format) ─────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_available_slots",
            "description": (
                "Get available appointment time slots for a doctor on a specific date. "
                "Call this whenever the patient wants to book an appointment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {
                        "type": "string",
                        "description": "Full or partial name of the doctor (e.g. 'Dr. Kamran Baig')"
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (e.g. '2026-06-05')"
                    }
                },
                "required": ["doctor_name", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment for the patient. Only call after confirming all details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_name":  {"type": "string"},
                    "patient_phone": {"type": "string"},
                    "doctor_name":   {"type": "string"},
                    "date":          {"type": "string", "description": "YYYY-MM-DD"},
                    "time":          {"type": "string", "description": "HH:MM (24-hour)"},
                    "reason":        {"type": "string", "description": "Patient's complaint"}
                },
                "required": ["patient_name", "patient_phone", "doctor_name", "date", "time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_appointment",
            "description": "Cancel an existing appointment by ID or patient name + doctor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer", "description": "Booking reference number"},
                    "patient_name":   {"type": "string"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reschedule_appointment",
            "description": "Reschedule an existing appointment to a new date/time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "appointment_id": {"type": "integer"},
                    "new_date":       {"type": "string", "description": "YYYY-MM-DD"},
                    "new_time":       {"type": "string", "description": "HH:MM"}
                },
                "required": ["appointment_id", "new_date", "new_time"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_doctors",
            "description": "Find doctors by symptom, specialty, or department name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Symptom or specialty (e.g. 'heart', 'children', 'bone pain')"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_doctor_next_available",
            "description": "Get the next available day and slots for a doctor within the next 7 days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doctor_name": {"type": "string"}
                },
                "required": ["doctor_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hospital_faq",
            "description": "Answer hospital FAQ questions (timings, fees, facilities, policies).",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"}
                },
                "required": ["question"]
            }
        }
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

async def execute_tool(name: str, args: dict, session: AsyncSession) -> str:
    """
    Dispatch a tool call and return a JSON string result.
    All DB access is async via the provided session.
    """
    try:
        if name == "get_available_slots":
            return await _get_available_slots(args, session)
        if name == "book_appointment":
            return await _book_appointment(args, session)
        if name == "cancel_appointment":
            return await _cancel_appointment(args, session)
        if name == "reschedule_appointment":
            return await _reschedule_appointment(args, session)
        if name == "search_doctors":
            return await _search_doctors(args, session)
        if name == "get_doctor_next_available":
            return await _get_doctor_next_available(args, session)
        if name == "hospital_faq":
            return await _hospital_faq(args, session)
        return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        logger.error(f"[Tool] {name} failed: {e}")
        return json.dumps({"error": str(e)})


# ── Individual tool implementations ──────────────────────────────────────────

async def _get_available_slots(args: dict, session: AsyncSession) -> str:
    doctor_name = args.get("doctor_name", "")
    date_str    = args.get("date", "")

    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD."})

    doctor = await _find_doctor(doctor_name, session)
    if not doctor:
        return json.dumps({"error": f"Doctor '{doctor_name}' not found."})

    day_of_week = target_date.weekday()  # 0=Monday

    # Get schedule blocks for this day
    result = await session.execute(
        select(DoctorSchedule).where(
            DoctorSchedule.doctor_id == doctor.id,
            DoctorSchedule.day_of_week == day_of_week,
        )
    )
    schedules = result.scalars().all()
    if not schedules:
        return json.dumps({
            "available": False,
            "doctor": doctor.name,
            "date": date_str,
            "message": f"{doctor.name} is not available on {target_date.strftime('%A')}."
        })

    # Get already-booked slots for this doctor+date
    booked = await session.execute(
        select(Appointment.appointment_time).where(
            Appointment.doctor_id == doctor.id,
            Appointment.appointment_date == target_date,
            Appointment.status == "confirmed",
        )
    )
    booked_times = {r[0] for r in booked}

    # Generate slots
    slots = []
    now = datetime.now()
    for sched in schedules:
        current = datetime.combine(target_date, sched.start_time)
        end     = datetime.combine(target_date, sched.end_time)
        while current < end:
            if current.time() not in booked_times:
                if target_date > now.date() or current > now:
                    slots.append(current.strftime("%H:%M"))
            current += timedelta(minutes=sched.slot_minutes)

    return json.dumps({
        "doctor": doctor.name,
        "department": doctor.specialization,
        "date": date_str,
        "day": target_date.strftime("%A"),
        "fee": doctor.consultation_fee,
        "available_slots": slots[:8],  # cap at 8
        "total_available": len(slots),
    })


async def _book_appointment(args: dict, session: AsyncSession) -> str:
    patient_name  = args.get("patient_name", "").strip()
    patient_phone = args.get("patient_phone", "").strip()
    doctor_name   = args.get("doctor_name", "").strip()
    date_str      = args.get("date", "").strip()
    time_str      = args.get("time", "").strip()
    reason        = args.get("reason", "").strip()

    # Validate inputs
    if not all([patient_name, patient_phone, doctor_name, date_str, time_str]):
        missing = [k for k, v in {
            "patient_name": patient_name, "patient_phone": patient_phone,
            "doctor_name": doctor_name, "date": date_str, "time": time_str
        }.items() if not v]
        return json.dumps({"error": f"Missing required fields: {', '.join(missing)}"})

    phone_digits = "".join(c for c in patient_phone if c.isdigit())
    if len(phone_digits) < 10:
        return json.dumps({"error": "Phone number must be at least 10 digits."})

    try:
        appt_date = date.fromisoformat(date_str)
        appt_time = time.fromisoformat(time_str)
    except ValueError as e:
        return json.dumps({"error": f"Invalid date/time format: {e}"})

    doctor = await _find_doctor(doctor_name, session)
    if not doctor:
        return json.dumps({"error": f"Doctor '{doctor_name}' not found."})

    # Check if slot is still free
    conflict = await session.execute(
        select(Appointment).where(
            Appointment.doctor_id == doctor.id,
            Appointment.appointment_date == appt_date,
            Appointment.appointment_time == appt_time,
            Appointment.status == "confirmed",
        )
    )
    if conflict.scalars().first():
        return json.dumps({"error": "This slot is already booked. Please choose another time."})

    # Upsert patient
    existing_patient = await session.execute(
        select(Patient).where(Patient.phone == phone_digits)
    )
    patient = existing_patient.scalars().first()
    if not patient:
        patient = Patient(name=patient_name, phone=phone_digits)
        session.add(patient)
        await session.flush()

    appt = Appointment(
        patient_id=patient.id,
        doctor_id=doctor.id,
        appointment_date=appt_date,
        appointment_time=appt_time,
        status="confirmed",
        reason=reason or None,
    )
    session.add(appt)
    await session.commit()
    await session.refresh(appt)

    return json.dumps({
        "success": True,
        "appointment_id": appt.id,
        "patient_name": patient_name,
        "doctor": doctor.name,
        "department": doctor.specialization,
        "date": appt_date.strftime("%A, %d %B %Y"),
        "time": appt_time.strftime("%I:%M %p"),
        "fee": doctor.consultation_fee,
    })


async def _cancel_appointment(args: dict, session: AsyncSession) -> str:
    appt_id      = args.get("appointment_id")
    patient_name = args.get("patient_name", "").strip()

    if appt_id:
        result = await session.execute(
            select(Appointment).where(Appointment.id == appt_id)
        )
        appt = result.scalars().first()
    elif patient_name:
        result = await session.execute(
            select(Appointment)
            .join(Patient)
            .where(
                Patient.name.ilike(f"%{patient_name}%"),
                Appointment.status == "confirmed",
            )
            .order_by(Appointment.created_at.desc())
        )
        appt = result.scalars().first()
    else:
        return json.dumps({"error": "Provide appointment_id or patient_name."})

    if not appt:
        return json.dumps({"error": "Appointment not found."})

    appt.status = "cancelled"
    await session.commit()
    return json.dumps({"success": True, "appointment_id": appt.id, "message": "Appointment cancelled."})


async def _reschedule_appointment(args: dict, session: AsyncSession) -> str:
    appt_id  = args.get("appointment_id")
    new_date = args.get("new_date", "")
    new_time = args.get("new_time", "")

    result = await session.execute(select(Appointment).where(Appointment.id == appt_id))
    appt = result.scalars().first()
    if not appt:
        return json.dumps({"error": "Appointment not found."})

    try:
        appt.appointment_date = date.fromisoformat(new_date)
        appt.appointment_time = time.fromisoformat(new_time)
    except ValueError as e:
        return json.dumps({"error": str(e)})

    await session.commit()
    return json.dumps({
        "success": True,
        "appointment_id": appt.id,
        "new_date": new_date,
        "new_time": new_time,
    })


async def _search_doctors(args: dict, session: AsyncSession) -> str:
    query = args.get("query", "").lower().strip()

    # Load departments and match keywords
    depts_result = await session.execute(select(Department))
    departments  = depts_result.scalars().all()

    matched_dept_ids = []
    for dept in departments:
        if (query in dept.name.lower() or
            query in dept.display.lower() or
            query in dept.description.lower()):
            matched_dept_ids.append(dept.id)

    if matched_dept_ids:
        doctors_result = await session.execute(
            select(Doctor).where(
                Doctor.department_id.in_(matched_dept_ids),
                Doctor.is_active == True,
            )
        )
    else:
        doctors_result = await session.execute(
            select(Doctor).where(
                Doctor.specialization.ilike(f"%{query}%"),
                Doctor.is_active == True,
            )
        )

    doctors = doctors_result.scalars().all()
    if not doctors:
        return json.dumps({"found": False, "message": f"No doctors found for '{query}'."})

    return json.dumps({
        "found": True,
        "doctors": [
            {
                "name": d.name,
                "specialization": d.specialization,
                "available_days": d.available_days,
                "fee": d.consultation_fee,
            }
            for d in doctors
        ]
    })


async def _get_doctor_next_available(args: dict, session: AsyncSession) -> str:
    doctor_name = args.get("doctor_name", "")
    doctor = await _find_doctor(doctor_name, session)
    if not doctor:
        return json.dumps({"error": f"Doctor '{doctor_name}' not found."})

    today = date.today()
    for offset in range(1, 8):
        check_date = today + timedelta(days=offset)
        day_of_week = check_date.weekday()

        sched_result = await session.execute(
            select(DoctorSchedule).where(
                DoctorSchedule.doctor_id == doctor.id,
                DoctorSchedule.day_of_week == day_of_week,
            )
        )
        schedules = sched_result.scalars().all()
        if not schedules:
            continue

        booked_result = await session.execute(
            select(Appointment.appointment_time).where(
                Appointment.doctor_id == doctor.id,
                Appointment.appointment_date == check_date,
                Appointment.status == "confirmed",
            )
        )
        booked_times = {r[0] for r in booked_result}

        slots = []
        for sched in schedules:
            current = datetime.combine(check_date, sched.start_time)
            end     = datetime.combine(check_date, sched.end_time)
            while current < end:
                if current.time() not in booked_times:
                    slots.append(current.strftime("%H:%M"))
                current += timedelta(minutes=sched.slot_minutes)

        if slots:
            return json.dumps({
                "doctor": doctor.name,
                "next_available_date": check_date.isoformat(),
                "day": check_date.strftime("%A, %d %B"),
                "first_slots": slots[:4],
            })

    return json.dumps({"error": f"{doctor.name} has no availability in the next 7 days."})


async def _hospital_faq(args: dict, session: AsyncSession) -> str:
    question = args.get("question", "").lower()

    result = await session.execute(select(FAQ))
    faqs   = result.scalars().all()

    # Simple keyword match — replace with vector search later
    best_match = None
    best_score = 0
    for faq in faqs:
        score = sum(
            1 for word in question.split()
            if word in faq.question.lower() or word in faq.question_ur
        )
        if score > best_score:
            best_score = score
            best_match = faq

    if best_match and best_score > 0:
        return json.dumps({
            "answer_en": best_match.answer,
            "answer_ur": best_match.answer_ur,
            "category":  best_match.category,
        })

    return json.dumps({"answer_en": "I don't have specific information about that. Please call us at +92-42-1234567.", "answer_ur": "اس بارے میں معلومات نہیں ہے۔ براہ کرم +92-42-1234567 پر رابطہ کریں۔"})


# ── Helper ────────────────────────────────────────────────────────────────────

async def _find_doctor(name: str, session: AsyncSession) -> Doctor | None:
    result = await session.execute(
        select(Doctor).where(Doctor.name.ilike(f"%{name}%"), Doctor.is_active == True)
    )
    return result.scalars().first()
