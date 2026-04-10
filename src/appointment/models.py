"""
SQLite database models for appointments.
Uses SQLModel (SQLAlchemy + Pydantic).
"""

import os
from datetime import date, datetime
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select

DATABASE_PATH = "data/appointments.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

engine = create_engine(DATABASE_URL, echo=False)


class Appointment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    patient_name: str = Field(index=True)
    patient_phone: str
    doctor_name: str = Field(index=True)
    department: str
    appointment_date: date = Field(index=True)
    appointment_time: str          # "09:00 AM"
    reason: Optional[str] = None
    status: str = Field(default="confirmed")   # confirmed | cancelled | completed
    notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)


def create_tables():
    os.makedirs("data", exist_ok=True)
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
