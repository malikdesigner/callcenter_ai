"""
Appointment booking logic.
Manages doctors, time slots, and database operations.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger
from sqlmodel import Session, select

from src.appointment.models import Appointment, engine

# ── Hospital configuration ────────────────────────────────────────────────────

DOCTORS: Dict[str, List[str]] = {
    "general":      ["Dr. Ahmed Khan",   "Dr. Sarah Malik"],
    "cardiology":   ["Dr. Ali Raza"],
    "orthopedic":   ["Dr. Fatima Noor"],
    "pediatric":    ["Dr. Usman Siddiq", "Dr. Nadia Shah"],
    "gynecology":   ["Dr. Amna Tariq"],
    "neurology":    ["Dr. Kamran Baig"],
}

MORNING_SLOTS = [
    "09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM",
    "11:00 AM", "11:30 AM",
]
AFTERNOON_SLOTS = [
    "02:00 PM", "02:30 PM", "03:00 PM", "03:30 PM",
    "04:00 PM", "04:30 PM",
]
ALL_SLOTS = MORNING_SLOTS + AFTERNOON_SLOTS

DEPARTMENT_ALIASES: Dict[str, str] = {
    "general medicine": "general",
    "general physician": "general",
    "gp": "general",
    "heart": "cardiology",
    "cardiac": "cardiology",
    "bone": "orthopedic",
    "ortho": "orthopedic",
    "child": "pediatric",
    "children": "pediatric",
    "kids": "pediatric",
    "women": "gynecology",
    "neuro": "neurology",
    "brain": "neurology",
}


def resolve_department(raw: str) -> Optional[str]:
    """Normalise a department name from free text."""
    key = raw.lower().strip()
    if key in DOCTORS:
        return key
    return DEPARTMENT_ALIASES.get(key)


# ── Booking system ─────────────────────────────────────────────────────────────

class BookingSystem:

    # ── Availability ──────────────────────────────────────────────────────────

    def get_booked_slots(self, doctor_name: str, target_date: date) -> set:
        with Session(engine) as session:
            rows = session.exec(
                select(Appointment).where(
                    Appointment.doctor_name == doctor_name,
                    Appointment.appointment_date == target_date,
                    Appointment.status == "confirmed",
                )
            ).all()
        return {r.appointment_time for r in rows}

    def get_available_slots(self, doctor_name: str, target_date: date) -> List[str]:
        """Return free time slots for a doctor on a given date."""
        booked = self.get_booked_slots(doctor_name, target_date)
        return [s for s in ALL_SLOTS if s not in booked]

    def get_next_available(
        self, department: str, days_ahead: int = 7
    ) -> List[dict]:
        """
        Find first available slots in a department within the next N days.
        Returns a list of {doctor, date, slots} dicts.
        """
        dept_key = resolve_department(department) or department.lower()
        doctors = DOCTORS.get(dept_key, [])
        results = []

        for offset in range(1, days_ahead + 1):
            check_date = date.today() + timedelta(days=offset)
            if check_date.weekday() == 6:   # Skip Sundays
                continue
            for doctor in doctors:
                slots = self.get_available_slots(doctor, check_date)
                if slots:
                    results.append(
                        {
                            "doctor": doctor,
                            "date": check_date.strftime("%A, %B %d %Y"),
                            "date_iso": check_date.isoformat(),
                            "slots": slots[:4],
                        }
                    )
        return results

    # ── Booking ───────────────────────────────────────────────────────────────

    def book_appointment(
        self,
        patient_name: str,
        patient_phone: str,
        doctor_name: str,
        department: str,
        appointment_date: date,
        appointment_time: str,
        reason: str = "",
    ) -> Appointment:
        available = self.get_available_slots(doctor_name, appointment_date)
        if appointment_time not in available:
            raise ValueError(
                f"Slot {appointment_time} is not available for {doctor_name} "
                f"on {appointment_date}. Available: {available}"
            )

        appt = Appointment(
            patient_name=patient_name,
            patient_phone=patient_phone,
            doctor_name=doctor_name,
            department=department,
            appointment_date=appointment_date,
            appointment_time=appointment_time,
            reason=reason,
        )

        with Session(engine) as session:
            session.add(appt)
            session.commit()
            session.refresh(appt)

        logger.info(
            f"[Booking] #{appt.id} — {patient_name} with {doctor_name} "
            f"on {appointment_date} at {appointment_time}"
        )
        return appt

    # ── Cancellation ─────────────────────────────────────────────────────────

    def cancel_appointment(self, appointment_id: int) -> bool:
        with Session(engine) as session:
            appt = session.get(Appointment, appointment_id)
            if appt and appt.status == "confirmed":
                appt.status = "cancelled"
                session.add(appt)
                session.commit()
                logger.info(f"[Booking] #{appointment_id} cancelled")
                return True
        return False

    # ── Query ─────────────────────────────────────────────────────────────────

    def find_by_phone(self, phone: str) -> List[Appointment]:
        with Session(engine) as session:
            return session.exec(
                select(Appointment).where(
                    Appointment.patient_phone == phone,
                    Appointment.status == "confirmed",
                )
            ).all()

    def list_all(self) -> List[Appointment]:
        with Session(engine) as session:
            return session.exec(select(Appointment)).all()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def department_list(self) -> str:
        return ", ".join(
            dept.title() for dept in DOCTORS
        )

    def doctors_in_department(self, department: str) -> List[str]:
        dept_key = resolve_department(department) or department.lower()
        return DOCTORS.get(dept_key, [])
