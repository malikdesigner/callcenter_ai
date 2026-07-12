"""
WhatsApp notification sender via Twilio.
Sends booking confirmation with fee and payment instructions after appointment is booked.
"""

import os
import threading
from loguru import logger


def _normalize_pk_phone(phone: str) -> str:
    """Convert Pakistani phone number to E.164 format (+92XXXXXXXXXX)."""
    p = phone.strip().replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
    if p.startswith("0092"):
        p = "+" + p[2:]
    elif p.startswith("92") and len(p) >= 12:
        p = "+" + p
    elif p.startswith("0") and len(p) == 11:
        p = "+92" + p[1:]
    elif not p.startswith("+"):
        p = "+92" + p
    return p


def _build_message(
    patient_name: str,
    doctor_name: str,
    appointment_date: str,
    appointment_time: str,
    booking_id: int,
    fee: str,
    payment: dict,
    hospital_name: str,
) -> str:
    fee_line = f"\n💰 *فیس:* {fee}" if fee else ""

    payment_block = ""
    if payment:
        method  = payment.get("method", "")
        account = payment.get("account_number", "")
        title   = payment.get("account_title", "")
        instrs  = payment.get("instructions", "ادائیگی کا اسکرین شاٹ اس نمبر پر بھیجیں")
        parts = []
        if method:  parts.append(f"*طریقہ:* {method}")
        if title:   parts.append(f"*نام:* {title}")
        if account: parts.append(f"*اکاؤنٹ:* {account}")
        if parts:
            payment_block = "\n\n💳 *ادائیگی کی معلومات:*\n" + "\n".join(parts)
            payment_block += f"\n\n📸 {instrs}"

    return (
        f"🏥 *{hospital_name}*\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"✅ *اپوائنٹمنٹ کنفرم ہوگئی*\n\n"
        f"👤 *نام:* {patient_name}\n"
        f"👨‍⚕️ *ڈاکٹر:* {doctor_name}\n"
        f"📅 *تاریخ:* {appointment_date}\n"
        f"🕐 *وقت:* {appointment_time}\n"
        f"🔢 *بکنگ نمبر:* #{booking_id}"
        f"{fee_line}"
        f"{payment_block}\n\n"
        f"وقت پر تشریف لائیں۔ شکریہ 🙏"
    )


def send_whatsapp_confirmation(
    patient_phone: str,
    patient_name: str,
    doctor_name: str,
    appointment_date: str,
    appointment_time: str,
    booking_id: int,
    fee: str = None,
    payment: dict = None,
    hospital_name: str = "City Medical Hospital",
) -> None:
    """Fire-and-forget: sends WhatsApp message in a background thread."""
    threading.Thread(
        target=_send_sync,
        args=(patient_phone, patient_name, doctor_name, appointment_date,
              appointment_time, booking_id, fee, payment, hospital_name),
        daemon=True,
    ).start()


def _send_sync(
    patient_phone, patient_name, doctor_name, appointment_date,
    appointment_time, booking_id, fee, payment, hospital_name,
):
    try:
        from twilio.rest import Client
    except ImportError:
        logger.warning("[WhatsApp] twilio package not installed — pip install twilio")
        return

    account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token  = os.getenv("TWILIO_AUTH_TOKEN", "")
    from_raw    = os.getenv("TWILIO_WHATSAPP_FROM", os.getenv("TWILIO_PHONE_NUMBER", ""))

    if not account_sid or not auth_token or not from_raw:
        logger.warning("[WhatsApp] Twilio credentials missing in .env")
        return

    from_number = f"whatsapp:{from_raw}" if not from_raw.startswith("whatsapp:") else from_raw
    to_number   = f"whatsapp:{_normalize_pk_phone(patient_phone)}"

    body = _build_message(
        patient_name, doctor_name, appointment_date, appointment_time,
        booking_id, fee, payment or {}, hospital_name,
    )

    try:
        client  = Client(account_sid, auth_token)
        message = client.messages.create(from_=from_number, to=to_number, body=body)
        logger.info(f"[WhatsApp] Sent to {to_number} | SID={message.sid}")
    except Exception as e:
        logger.error(f"[WhatsApp] Failed: {e}")
