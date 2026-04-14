"""
Seed the database with initial doctors and departments from the hardcoded configuration.
"""
import sys
import os
sys.path.append(os.getcwd())

from sqlmodel import Session, select
from src.appointment.models import Doctor, Department, engine, create_tables

INITIAL_DATA = {
    "general":      ["Dr. Ahmed Khan",   "Dr. Sarah Malik"],
    "cardiology":   ["Dr. Ali Raza"],
    "orthopedic":   ["Dr. Fatima Noor"],
    "pediatric":    ["Dr. Usman Siddiq", "Dr. Nadia Shah"],
    "gynecology":   ["Dr. Amna Tariq"],
    "neurology":    ["Dr. Kamran Baig"],
}

def seed():
    create_tables()
    with Session(engine) as session:
        # Check if already seeded
        if session.exec(select(Doctor)).first():
            print("Database already seeded.")
            return

        print("Seeding departments and doctors...")
        for dept_name, doctors in INITIAL_DATA.items():
            dept = Department(name=dept_name)
            session.add(dept)
            session.commit()
            session.refresh(dept)

            for doc_name in doctors:
                doc = Doctor(name=doc_name, department_name=dept_name)
                session.add(doc)
        
        session.commit()
        print("Seeding complete.")

if __name__ == "__main__":
    seed()
