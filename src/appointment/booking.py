"""
Appointment booking logic.
Manages doctors, time slots, and database operations.
"""

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

from loguru import logger
from sqlmodel import Session, select

from src.appointment.models import Appointment, Doctor, Department, DoctorSlot, engine

# ── Hospital configuration ────────────────────────────────────────────────────

# DOCTORS dictionary is now legacy, using DB.

# Slots are now generated dynamically per doctor.

DEPARTMENT_ALIASES: Dict[str, str] = {
    # General
    "general medicine": "general",
    "general physician": "general",
    "general practitioner": "general",
    "gp": "general",
    "family doctor": "general",
    # Eye / Ophthalmology → routed to general
    "eye": "general",
    "eyes": "general",
    "vision": "general",
    "ophthalmology": "general",
    "ophthalmologist": "general",
    "eye surgery": "general",
    "eye problem": "general",
    "eye check": "general",
    "sight": "general",
    # Cardiology
    "heart": "cardiology",
    "cardiac": "cardiology",
    "chest": "cardiology",
    "chest pain": "cardiology",
    "cardiologist": "cardiology",
    # Orthopedic
    "bone": "orthopedic",
    "bones": "orthopedic",
    "joint": "orthopedic",
    "joints": "orthopedic",
    "fracture": "orthopedic",
    "ortho": "orthopedic",
    "orthopedics": "orthopedic",
    "knee": "orthopedic",
    "back pain": "orthopedic",
    "spine": "orthopedic",
    # Pediatric
    "child": "pediatric",
    "children": "pediatric",
    "kids": "pediatric",
    "baby": "pediatric",
    "infant": "pediatric",
    "paediatric": "pediatric",
    # Gynecology
    "women": "gynecology",
    "gynae": "gynecology",
    "gynaecology": "gynecology",
    "pregnancy": "gynecology",
    "maternity": "gynecology",
    # Neurology
    "neuro": "neurology",
    "brain": "neurology",
    "headache": "neurology",
    "migraine": "neurology",
    "neurologist": "neurology",
}


def resolve_department(query: str) -> Optional[str]:
    """Resolves a raw query to a canonical department name using aliases."""
    q = query.lower().strip()
    if q in DEPARTMENT_ALIASES:
        return DEPARTMENT_ALIASES[q]
    
    # Partial matching: "eye" in "eye surgery"
    for alias, canonical in DEPARTMENT_ALIASES.items():
        if alias in q or q in alias:
            return canonical
            
    # Check if key matches a real department in DB
    with Session(engine) as session:
        dept = session.exec(select(Department).where(Department.name == q)).first()
        if dept:
            return q
            
    return None


# ── Booking system ─────────────────────────────────────────────────────────────

class BookingSystem:

    # ── Availability ──────────────────────────────────────────────────────────

    def _generate_slots(self, start_str: str, end_str: str, interval_min: int = 30) -> List[str]:
        """Generate a list of time strings between start and end (inclusive, 30-min steps)."""
        slots = []
        try:
            current = datetime.strptime(start_str, "%I:%M %p")
            end = datetime.strptime(end_str, "%I:%M %p")
            
            while current <= end:
                slots.append(current.strftime("%I:%M %p"))
                current += timedelta(minutes=interval_min)
        except Exception as e:
            logger.error(f"[Booking] Error generating slots: {e}")
            return ["09:00 AM", "09:30 AM", "10:00 AM", "10:30 AM", "11:00 AM", "11:30 AM"] # Fallback

        return slots

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
        """Return free time slots for a doctor on a given date by aggregating all their shifts."""
        booked = self.get_booked_slots(doctor_name, target_date)
        
        # Fetch all shifts/slots defined for this doctor
        with Session(engine) as session:
            doctor_shifts = session.exec(select(DoctorSlot).where(DoctorSlot.doctor_name == doctor_name)).all()
        
        if not doctor_shifts:
            # Default fallback if no slots defined yet
            possible_slots = self._generate_slots("09:00 AM", "05:00 PM")
        else:
            # Aggregate all possible slots from all shifts
            possible_slots = []
            for shift in doctor_shifts:
                possible_slots.extend(self._generate_slots(shift.start_time, shift.end_time))
            # Sort and remove duplicates if any overlap occurred
            possible_slots = sorted(list(set(possible_slots)), key=lambda x: datetime.strptime(x, "%I:%M %p"))

        # Filter out slots in the past if the date is today
        now = datetime.now()
        is_today = (target_date == now.date())
        
        available = []
        for slot in possible_slots:
            if slot in booked:
                continue
            
            if is_today:
                # Convert slot (e.g. "09:00 AM") to datetime for comparison
                slot_time = datetime.strptime(slot, "%I:%M %p").time()
                current_time = now.time()
                # Use a 15-min buffer (can't book something that starts in less than 15 mins)
                buffer_time = (datetime.combine(date.min, current_time) + timedelta(minutes=15)).time()
                
                if slot_time < buffer_time:
                    continue
            
            available.append(slot)
            
        return available

    def get_next_available(
        self, department: str, days_ahead: int = 7
    ) -> List[dict]:
        """
        Find first available slots in a department within the next N days.
        Returns a list of {doctor, date, slots} dicts.
        """
        dept_key = resolve_department(department) or department.lower()
        with Session(engine) as session:
            doctors = session.exec(select(Doctor).where(Doctor.department_name == dept_key, Doctor.is_active == True)).all()
            doctor_names = [d.name for d in doctors]
        results = []

        for offset in range(1, days_ahead + 1):
            check_date = date.today() + timedelta(days=offset)
            if check_date.weekday() == 6:   # Skip Sundays
                continue
            for doctor in doctor_names:
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
        transcript: Optional[str] = None,
    ) -> Appointment:
        # ── Defensive Placeholder Check ──
        placeholders = {"patient name", "unknown", "n/a", "none", "phone number", "placeholder"}
        if patient_name.lower() in placeholders or len(patient_name) < 2:
            raise ValueError(f"Invalid patient name provided: '{patient_name}'")
        
        # Phone: 10–13 digits covers local (03XX-XXXXXXX=11) and country-code (923XX-XXXXXXX=12)
        clean_phone = "".join(filter(str.isdigit, patient_phone))
        if not (10 <= len(clean_phone) <= 13) or patient_phone.lower() in placeholders:
            raise ValueError(f"Invalid phone number: '{patient_phone}' ({len(clean_phone)} digits — expected 10–13)")

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
            transcript=transcript,
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
        with Session(engine) as session:
            depts = session.exec(select(Department)).all()
            return ", ".join(d.name.title() for d in depts)

    def doctors_in_department(self, department: str) -> List[str]:
        dept_key = resolve_department(department) or department.lower()
        with Session(engine) as session:
            doctors = session.exec(select(Doctor).where(Doctor.department_name == dept_key, Doctor.is_active == True)).all()
            return [d.name for d in doctors]
