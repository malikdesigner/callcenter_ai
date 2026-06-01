"""
SQLAlchemy ORM models for the hospital AI system.
Uses async-compatible mapped columns.
"""

from datetime import date, datetime, time
from typing import Optional

from sqlalchemy import String, Text, Integer, Boolean, Date, Time, DateTime, Float, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.database import Base


class Department(Base):
    __tablename__ = "departments"

    id:          Mapped[int]          = mapped_column(Integer, primary_key=True)
    name:        Mapped[str]          = mapped_column(String(100), unique=True, index=True)
    display:     Mapped[str]          = mapped_column(String(200), default="")
    display_ur:  Mapped[str]          = mapped_column(String(200), default="")
    description: Mapped[str]          = mapped_column(Text, default="")
    floor:       Mapped[Optional[str]]= mapped_column(String(100), nullable=True)

    doctors: Mapped[list["Doctor"]] = relationship(back_populates="department_rel", lazy="select")


class Doctor(Base):
    __tablename__ = "doctors"

    id:             Mapped[int]          = mapped_column(Integer, primary_key=True)
    name:           Mapped[str]          = mapped_column(String(200), index=True)
    department_id:  Mapped[int]          = mapped_column(ForeignKey("departments.id"))
    specialization: Mapped[str]          = mapped_column(String(300), default="")
    qualification:  Mapped[str]          = mapped_column(String(300), default="")
    available_days: Mapped[str]          = mapped_column(String(100), default="")  # "Mon,Wed,Fri"
    consultation_fee: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    bio_ur:         Mapped[str]          = mapped_column(Text, default="")
    is_active:      Mapped[bool]         = mapped_column(Boolean, default=True)

    department_rel: Mapped["Department"] = relationship(back_populates="doctors")
    schedules:      Mapped[list["DoctorSchedule"]] = relationship(back_populates="doctor", lazy="select")
    appointments:   Mapped[list["Appointment"]]    = relationship(back_populates="doctor_rel", lazy="select")


class DoctorSchedule(Base):
    """
    Weekly recurring schedule for a doctor.
    day_of_week: 0=Monday … 6=Sunday.
    Each row = one daily block (a doctor can have multiple blocks per day).
    """
    __tablename__ = "doctor_schedule"

    id:            Mapped[int]  = mapped_column(Integer, primary_key=True)
    doctor_id:     Mapped[int]  = mapped_column(ForeignKey("doctors.id"), index=True)
    day_of_week:   Mapped[int]  = mapped_column(Integer)   # 0 = Monday, 6 = Sunday
    start_time:    Mapped[time] = mapped_column(Time)
    end_time:      Mapped[time] = mapped_column(Time)
    slot_minutes:  Mapped[int]  = mapped_column(Integer, default=30)

    doctor: Mapped["Doctor"] = relationship(back_populates="schedules")


class Patient(Base):
    __tablename__ = "patients"

    id:         Mapped[int]          = mapped_column(Integer, primary_key=True)
    name:       Mapped[str]          = mapped_column(String(200))
    phone:      Mapped[str]          = mapped_column(String(30), unique=True, index=True)
    created_at: Mapped[datetime]     = mapped_column(DateTime, default=func.now())

    appointments: Mapped[list["Appointment"]] = relationship(back_populates="patient_rel", lazy="select")


class Appointment(Base):
    __tablename__ = "appointments"

    id:               Mapped[int]          = mapped_column(Integer, primary_key=True)
    patient_id:       Mapped[int]          = mapped_column(ForeignKey("patients.id"), index=True)
    doctor_id:        Mapped[int]          = mapped_column(ForeignKey("doctors.id"), index=True)
    appointment_date: Mapped[date]         = mapped_column(Date, index=True)
    appointment_time: Mapped[time]         = mapped_column(Time)
    status:           Mapped[str]          = mapped_column(String(20), default="confirmed")
    reason:           Mapped[Optional[str]]= mapped_column(Text, nullable=True)
    transcript:       Mapped[Optional[str]]= mapped_column(Text, nullable=True)
    created_at:       Mapped[datetime]     = mapped_column(DateTime, default=func.now())

    patient_rel: Mapped["Patient"] = relationship(back_populates="appointments")
    doctor_rel:  Mapped["Doctor"]  = relationship(back_populates="appointments")


class FAQ(Base):
    __tablename__ = "faqs"

    id:       Mapped[int]          = mapped_column(Integer, primary_key=True)
    question: Mapped[str]          = mapped_column(Text)
    answer:   Mapped[str]          = mapped_column(Text)
    question_ur: Mapped[str]       = mapped_column(Text, default="")
    answer_ur:   Mapped[str]       = mapped_column(Text, default="")
    category: Mapped[str]          = mapped_column(String(50), default="general")


class ClaudeUsage(Base):
    """
    Monthly Claude API spend tracker.
    One row per calendar month (YYYY-MM).
    Updated after every successful Claude call — persists across restarts.
    """
    __tablename__ = "claude_usage"

    id:             Mapped[int]   = mapped_column(Integer, primary_key=True)
    month:          Mapped[str]   = mapped_column(String(7), unique=True, index=True)  # "2026-06"
    total_calls:    Mapped[int]   = mapped_column(Integer, default=0)
    total_tokens:   Mapped[int]   = mapped_column(Integer, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float,   default=0.0)
    updated_at:     Mapped[datetime] = mapped_column(DateTime, default=func.now())


class ConversationLog(Base):
    """
    Full audit trail of every conversation turn.
    Stores transcript, intent, tool calls, model used, quality score,
    and whether Claude fallback was triggered — so issues can be analyzed later.
    """
    __tablename__ = "conversation_logs"

    id:               Mapped[int]           = mapped_column(Integer, primary_key=True)
    session_id:       Mapped[str]           = mapped_column(String(100), index=True)
    language:         Mapped[str]           = mapped_column(String(10), default="ur")
    turn_number:      Mapped[int]           = mapped_column(Integer, default=0)

    # STT
    user_transcript:  Mapped[str]           = mapped_column(Text, default="")
    asr_confidence:   Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Intent + state
    detected_intent:  Mapped[str]           = mapped_column(String(50), default="")
    policy_state:     Mapped[str]           = mapped_column(String(50), default="")

    # Tool calls
    tool_called:      Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    tool_result:      Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Response
    agent_response:   Mapped[str]           = mapped_column(Text, default="")
    model_used:       Mapped[str]           = mapped_column(String(50), default="")
    quality_score:    Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    quality_reason:   Mapped[str]           = mapped_column(String(50), default="")
    used_fallback:    Mapped[bool]          = mapped_column(Boolean, default=False)

    # Performance
    latency_ms:       Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at:       Mapped[datetime]      = mapped_column(DateTime, default=func.now())
