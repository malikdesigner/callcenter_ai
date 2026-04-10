"""
Component-level tests. Run each section independently to verify setup.
Usage:  python test_components.py [stt|tts|llm|booking|all]
"""

import sys
import os


def test_stt():
    print("\n── STT (Whisper) ─────────────────────────────────")
    import numpy as np
    from src.stt.transcriber import Transcriber

    t = Transcriber()
    # Generate 2 seconds of silence — should return empty string
    dummy = np.zeros(32000, dtype=np.float32)
    result = t.transcribe(dummy)
    print(f"  Silence → '{result}'  (expected empty)")
    print("  ✓ STT working")


def test_tts():
    print("\n── TTS (XTTS v2) ─────────────────────────────────")
    from src.tts.synthesizer import VoiceSynthesizer

    voice = os.getenv("VOICE_SAMPLE_PATH", "voices/receptionist.wav")
    if not os.path.exists(voice):
        print(f"  ✗ Voice sample not found: {voice}")
        print("    Place a WAV file there and re-run.")
        return

    s = VoiceSynthesizer()
    audio = s.synthesize("Hello, this is a test of the voice synthesis.")
    print(f"  Generated {len(audio)} bytes of audio")
    s.synthesize_to_file("Appointment confirmed.", "data/test_output.wav")
    print("  Saved to data/test_output.wav")
    print("  ✓ TTS working")


def test_llm():
    print("\n── LLM (Ollama) ──────────────────────────────────")
    from src.llm.agent import HospitalAgent

    agent = HospitalAgent()
    greeting = agent.get_greeting()
    print(f"  Greeting: {greeting}")

    r1 = agent.process_turn("I need to book an appointment")
    print(f"  Turn 1: {r1['speech']}")

    r2 = agent.process_turn("For cardiology please")
    print(f"  Turn 2: {r2['speech']}")
    print("  ✓ LLM working")


def test_booking():
    print("\n── Booking System ────────────────────────────────")
    from datetime import date, timedelta
    from src.appointment.models import create_tables
    from src.appointment.booking import BookingSystem

    create_tables()
    b = BookingSystem()

    tomorrow = date.today() + timedelta(days=1)
    slots = b.get_available_slots("Dr. Ahmed Khan", tomorrow)
    print(f"  Available slots for Dr. Ahmed Khan tomorrow: {slots[:3]}")

    avail = b.get_next_available("general", days_ahead=3)
    for item in avail[:2]:
        print(f"  {item['doctor']} | {item['date']} | {item['slots'][:2]}")

    appt = b.book_appointment(
        patient_name="Test Patient",
        patient_phone="03001234567",
        doctor_name="Dr. Ahmed Khan",
        department="general",
        appointment_date=tomorrow,
        appointment_time=slots[0],
        reason="Test booking",
    )
    print(f"  Booked: #{appt.id}  {appt.patient_name}  {appt.appointment_time}")

    b.cancel_appointment(appt.id)
    print(f"  Cancelled: #{appt.id}")
    print("  ✓ Booking working")


def test_vad():
    print("\n── VAD (Silero) ──────────────────────────────────")
    import numpy as np
    from src.vad.detector import VADDetector

    vad = VADDetector()
    silence = np.zeros(8000, dtype=np.float32)
    noise   = np.random.normal(0, 0.3, 8000).astype(np.float32)

    print(f"  Silence → speech={vad.is_speech(silence)}")
    print(f"  Noise   → speech={vad.is_speech(noise)}")
    print("  ✓ VAD working")


TESTS = {
    "stt":     test_stt,
    "tts":     test_tts,
    "llm":     test_llm,
    "booking": test_booking,
    "vad":     test_vad,
}

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    if target == "all":
        for name, fn in TESTS.items():
            try:
                fn()
            except Exception as e:
                print(f"  ✗ {name} FAILED: {e}")
    elif target in TESTS:
        TESTS[target]()
    else:
        print(f"Unknown test '{target}'. Choose from: {', '.join(TESTS)} or 'all'")
