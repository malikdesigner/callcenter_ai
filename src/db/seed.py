"""
Seed the database from YAML knowledge base files.
Run once at startup if tables are empty.
"""

import yaml
from pathlib import Path
from datetime import time
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Department, Doctor, DoctorSchedule, FAQ

KB = Path("knowledge_base")

_DAY_MAP = {
    "Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3,
    "Fri": 4, "Sat": 5, "Sun": 6,
}


async def seed(session: AsyncSession):
    """Idempotent: only inserts if tables are empty."""
    dept_count = (await session.execute(select(Department))).scalars().first()
    if dept_count is not None:
        return  # already seeded

    logger.info("[DB] Seeding knowledge base...")

    # ── Departments ────────────────────────────────────────────────────────────
    dept_data = yaml.safe_load((KB / "departments.yaml").read_text(encoding="utf-8"))
    dept_map: dict[str, int] = {}
    for d in dept_data["departments"]:
        dept = Department(
            name=d["name"],
            display=d.get("display", ""),
            display_ur=d.get("display_ur", ""),
            description=d.get("description", ""),
            floor=d.get("floor"),
        )
        session.add(dept)
        await session.flush()
        dept_map[d["name"]] = dept.id
        logger.debug(f"[DB] Department: {dept.name} id={dept.id}")

    # ── Doctors + schedules ────────────────────────────────────────────────────
    doc_data = yaml.safe_load((KB / "doctors.yaml").read_text(encoding="utf-8"))
    for d in doc_data["doctors"]:
        dept_id = dept_map.get(d["department"])
        if not dept_id:
            logger.warning(f"[DB] Unknown department '{d['department']}' for {d['name']}")
            continue

        doctor = Doctor(
            name=d["name"],
            department_id=dept_id,
            specialization=d.get("specialization", ""),
            qualification=d.get("qualification", ""),
            available_days=d.get("available_days", ""),
            consultation_fee=d.get("fee"),
            bio_ur=d.get("bio_ur", ""),
            is_active=True,
        )
        session.add(doctor)
        await session.flush()

        # Build weekly schedule from days + slots
        for day_str in d.get("available_days", "").split(","):
            day_str = day_str.strip()
            day_num = _DAY_MAP.get(day_str)
            if day_num is None:
                continue
            for slot in d.get("slots", []):
                sh, sm = map(int, slot["start"].split(":"))
                eh, em = map(int, slot["end"].split(":"))
                sched = DoctorSchedule(
                    doctor_id=doctor.id,
                    day_of_week=day_num,
                    start_time=time(sh, sm),
                    end_time=time(eh, em),
                    slot_minutes=30,
                )
                session.add(sched)

        logger.debug(f"[DB] Doctor: {doctor.name}")

    # ── FAQs ───────────────────────────────────────────────────────────────────
    faq_data = yaml.safe_load((KB / "faqs.yaml").read_text(encoding="utf-8"))
    for f in faq_data["faqs"]:
        faq = FAQ(
            question=f.get("q", ""),
            answer=f.get("a", ""),
            question_ur=f.get("q_ur", ""),
            answer_ur=f.get("a_ur", ""),
            category=f.get("category", "general"),
        )
        session.add(faq)

    await session.commit()
    logger.info("[DB] Seed complete.")
