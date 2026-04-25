"""
Seed the database with initial departments, doctors, and time slots.

Run once to set up the hospital:
    python scripts/seed_db.py

To reseed from scratch, delete data/appointments.db first.
"""
import sys
import os
sys.path.append(os.getcwd())

from sqlmodel import Session, select
from src.appointment.models import Doctor, Department, DoctorSlot, engine, create_tables

# ── Hospital data ──────────────────────────────────────────────────────────────
# Each entry: department_name → list of (doctor_name, start_time, end_time)
# Times use 12-hour format matching the slot system ("09:00 AM" / "05:00 PM").
INITIAL_DATA = {
    "general": [
        ("Dr. Ahmed Khan",  "09:00 AM", "05:00 PM"),
        ("Dr. Sarah Malik", "09:00 AM", "01:00 PM"),  # morning shift only
    ],
    "cardiology": [
        ("Dr. Ali Raza",    "10:00 AM", "04:00 PM"),
    ],
    "orthopedic": [
        ("Dr. Fatima Noor", "09:00 AM", "03:00 PM"),
    ],
    "pediatric": [
        ("Dr. Usman Siddiq","09:00 AM", "05:00 PM"),
        ("Dr. Nadia Shah",  "01:00 PM", "05:00 PM"),  # afternoon shift only
    ],
    "gynecology": [
        ("Dr. Amna Tariq",  "09:00 AM", "02:00 PM"),
    ],
    "neurology": [
        ("Dr. Kamran Baig", "11:00 AM", "05:00 PM"),
    ],
}

DEPARTMENT_DESCRIPTIONS = {
    "general":    "General medicine and outpatient care",
    "cardiology": "Heart and cardiovascular conditions",
    "orthopedic": "Bone, joint, and musculoskeletal conditions",
    "pediatric":  "Children's health and development",
    "gynecology": "Women's health, pregnancy, and maternity",
    "neurology":  "Brain, spine, and nervous system conditions",
}


def seed():
    create_tables()
    with Session(engine) as session:
        # Check if already seeded
        if session.exec(select(Doctor)).first():
            print("Database already seeded. Delete data/appointments.db to reseed.")
            return

        print("Seeding departments, doctors, and slots...")
        for dept_name, doctors in INITIAL_DATA.items():
            # Department
            dept = Department(
                name=dept_name,
                description=DEPARTMENT_DESCRIPTIONS.get(dept_name, ""),
            )
            session.add(dept)
            session.commit()
            session.refresh(dept)

            for doc_name, start_time, end_time in doctors:
                # Doctor record
                doc = Doctor(name=doc_name, department_name=dept_name)
                session.add(doc)

                # Explicit time slot definition (start → end, 30-min intervals)
                slot = DoctorSlot(
                    doctor_name=doc_name,
                    start_time=start_time,
                    end_time=end_time,
                )
                session.add(slot)

        session.commit()
        print("Seeding complete.")
        print("\nDoctors seeded:")
        for dept_name, doctors in INITIAL_DATA.items():
            print(f"  {dept_name.title()}:")
            for doc_name, start, end in doctors:
                print(f"    {doc_name}  ({start} – {end})")


if __name__ == "__main__":
    seed()
