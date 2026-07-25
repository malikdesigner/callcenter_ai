# Sara AI — Hospital Receptionist

**Sara** is a real-time AI voice agent for City Medical Hospital. A patient calls (via browser), speaks naturally in English or Urdu, and Sara books their appointment end-to-end — collecting name, phone, symptoms, doctor, date, and time — then writes the confirmed appointment to a database. No forms, no menus, no hold music.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Project Structure](#2-project-structure)
3. [First-Time Setup](#3-first-time-setup)
4. [Configuration (.env)](#4-configuration-env)
5. [Running the Server](#5-running-the-server)
6. [Access URLs](#6-access-urls)
7. [How a Call Works — Full Flow](#7-how-a-call-works--full-flow)
8. [Component Deep-Dive](#8-component-deep-dive)
9. [LLM Fallback Chain](#9-llm-fallback-chain)
10. [Bilingual Mode](#10-bilingual-mode)
11. [Database & Booking](#11-database--booking)
12. [Admin Dashboard](#12-admin-dashboard)
13. [Call Logs](#13-call-logs)
14. [Troubleshooting](#14-troubleshooting)

---

## 1. System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 | Windows 11 |
| Python | 3.10 | 3.11 |
| RAM | 8 GB | 16 GB |
| GPU | None (CPU mode) | NVIDIA 6 GB+ VRAM |
| CUDA | — | 12.1 |
| Disk | 5 GB free | 10 GB free |
| Internet | Required (cloud LLMs) | Required |

**Required external software**

- [Ollama](https://ollama.com/) — local LLM server (used as final fallback)
- A modern browser with microphone access (Chrome, Edge, Firefox)

**Required API keys (at least one LLM)**

- Google Gemini API key (free tier: 15 requests/min) — primary for Urdu
- Groq API key (free tier: generous limits) — fast fallback
- Ollama running locally — always-available final fallback

---

## 2. Project Structure

```
CallAgent/
│
├── main.py                    ← Entry point. Starts all 3 servers in one process.
├── setup.bat                  ← First-time setup script (Windows)
├── start.bat                  ← Daily start script (activates venv + starts Ollama)
├── requirements.txt           ← Python dependencies
├── .env                       ← Your API keys and config (create from .env.example)
│
├── config/
│   └── settings.py            ← Reads .env into a typed Settings object
│
├── src/
│   ├── api/
│   │   ├── server.py          ← FastAPI app factory, WebSocket endpoint, REST APIs
│   │   └── twilio_handler.py  ← Twilio phone integration (optional)
│   │
│   ├── pipeline/
│   │   ├── call_handler.py    ← Core orchestrator: audio → STT → LLM → TTS → browser
│   │   └── stt_filter.py      ← Quality filter: confidence, length, gibberish check
│   │
│   ├── llm/
│   │   ├── agent.py           ← HospitalAgent: booking state machine + LLM calls
│   │   └── intent.py          ← Rule-based intent classifier (no LLM needed)
│   │
│   ├── dialogue/
│   │   ├── fast_engine.py     ← Handles structured inputs (name/phone/date/time) instantly
│   │   ├── policy.py          ← Dialogue state machine and NLG constraints
│   │   └── semantic_validator.py  ← Rejects nonsensical inputs before LLM
│   │
│   ├── stt/
│   │   └── transcriber.py     ← Whisper large-v3 via faster-whisper (GPU/CPU)
│   │
│   ├── tts/
│   │   ├── synthesizer.py     ← Edge-TTS + caching + voice switching
│   │   └── speech_formatter.py ← Cleans LLM output before TTS (removes markdown etc.)
│   │
│   ├── vad/
│   │   └── detector.py        ← Silero VAD: detects when caller is speaking
│   │
│   └── appointment/
│       ├── models.py          ← SQLite schema: Appointment, Doctor, Department, DoctorSlot
│       └── booking.py         ← Slot availability, booking writes, department aliases
│
├── static/
│   ├── index.html             ← English caller UI (port 8000)
│   ├── index_ur.html          ← Urdu caller UI (port 8001)
│   ├── index_select.html      ← Bilingual selector UI (port 8002)
│   └── dashboard.html         ← Admin dashboard
│
├── data/
│   ├── appointments.db        ← SQLite database (auto-created)
│   ├── hospital_knowledge.json ← Hospital info (hours, policies, departments)
│   ├── cache/                 ← Pre-synthesized greeting audio (auto-generated)
│   ├── tts_cache/             ← Persistent TTS audio cache (auto-generated)
│   └── call_logs/             ← Per-call JSON transcripts (auto-generated)
│
├── voices/
│   └── receptionist.wav       ← (Optional) Reference audio for voice cloning
│
└── docs/
    ├── sara-call-flow.html    ← Visual call flow diagrams (English + Urdu)
    └── sara-technical.html    ← Full technical architecture reference
```

---

## 3. First-Time Setup

### Step 1 — Clone / download the project

Place the project folder anywhere, e.g. `E:\Personal\MyProject\CallAgent\`

### Step 2 — Install Ollama

Download from [ollama.com](https://ollama.com/) and install. Then pull a model:

```bash
ollama pull qwen2.5:14b-instruct
```

> A smaller model like `llama3.2:3b` works but gives weaker Urdu output.

### Step 3 — Run the automated setup

Double-click `setup.bat` or run from terminal:

```bat
setup.bat
```

This will:
1. Create a Python virtual environment (`venv/`)
2. Install PyTorch with CUDA 12.1 support
3. Install all dependencies from `requirements.txt`
4. Create the `voices/`, `data/`, and `static/` folders
5. Pull the Ollama fallback model

> **Total download size:** ~5–8 GB (PyTorch + Whisper large-v3 + dependencies)

### Step 4 — Create your `.env` file

Create a file named `.env` in the project root:

```env
# ── LLM: Gemini (primary for Urdu) — get free keys at aistudio.google.com ──
GEMINI_API_KEY=AIza...
GEMINI_API_KEY1=AIza...      # Optional: up to 6 keys rotated automatically
GEMINI_API_KEY2=AIza...
GEMINI_MODEL=gemini-3.5-flash

# ── LLM: Groq (fast fallback) — get free key at console.groq.com ──
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile

# ── LLM: Ollama (local final fallback) ──
OLLAMA_MODEL=qwen2.5:14b-instruct
OLLAMA_HOST=http://localhost:11434

# ── STT: Whisper ──
WHISPER_MODEL=large-v3
WHISPER_DEVICE=cuda          # Use "cpu" if no NVIDIA GPU
WHISPER_COMPUTE_TYPE=int8_float16   # Use "int8" for CPU mode

# ── Audio ──
SAMPLE_RATE=16000
SILENCE_DURATION=1.8         # Seconds of silence before processing speech
VAD_THRESHOLD=0.35

# ── Hospital ──
HOSPITAL_NAME=City Medical Hospital
RECEPTIONIST_NAME=Sara
```

### Step 5 — Add doctors to the database

Start the server (Step 5 below), then open the admin dashboard at `http://localhost:8000/dashboard` to add doctors, departments, and availability slots.

---

## 4. Configuration (.env)

All settings are read by `config/settings.py` using `pydantic-settings`. Environment variables override `.env` file values.

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` … `GEMINI_API_KEY5` | — | Up to 6 Gemini keys, rotated automatically |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model name |
| `GROQ_API_KEY` | — | Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `OLLAMA_MODEL` | `qwen2.5:14b-instruct` | Local Ollama model |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cpu` | `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8_float16` (GPU) or `int8` (CPU) |
| `SILENCE_DURATION` | `1.8` | Seconds of silence to trigger turn processing |
| `VAD_THRESHOLD` | `0.35` | Silero VAD sensitivity (0–1, lower = more sensitive) |
| `HOSPITAL_NAME` | `City Medical Hospital` | Injected into Sara's prompts |
| `RECEPTIONIST_NAME` | `Sara` | Sara's name in prompts |

---

## 5. Running the Server

**Daily start (recommended):**

```bat
start.bat
```

This activates the venv, starts Ollama in the background, and launches the Python server.

**Manual start:**

```bash
# Activate virtual environment
venv\Scripts\activate

# Start Ollama (in a separate terminal or background)
ollama serve

# Start Sara
python main.py
```

**Expected startup output:**

```
=======================================================
  Sara AI — City Medical Hospital
=======================================================
  Warnings (non-fatal):
    Ollama model 'qwen2.5:14b-instruct' found.
=======================================================

Loading models...
[Whisper] Loading large-v3 on cpu (int8)...
[VAD] Silero VAD loaded.
[TTS] Edge-TTS ready. Voice: ur-PK-UzmaNeural (ur)
Sara AI ready:
  English                → http://localhost:8000
  Urdu                   → http://localhost:8001
  Language selector call → http://localhost:8002
Press Ctrl+C to stop.
```

> **First start takes 2–4 minutes** while Whisper downloads and loads. Subsequent starts are fast as the model is cached.

---

## 6. Access URLs

| URL | Purpose | Language |
|---|---|---|
| `http://localhost:8000` | Patient call interface | English |
| `http://localhost:8001` | Patient call interface | Urdu |
| `http://localhost:8002` | Patient call — Sara chooses language | Bilingual |
| `http://localhost:8000/dashboard` | Admin appointment management | English |

All three ports serve the same Python process. Models are loaded once and shared.

---

## 7. How a Call Works — Full Flow

This section traces a single complete call from browser click to database write.

### Phase 1: Call starts

1. Patient opens `http://localhost:8002` and clicks **Start Call**
2. Browser requests microphone permission via the Web Audio API
3. Browser generates a unique `session_id` (timestamp-based) and opens a WebSocket to `ws://localhost:8002/ws/call/{session_id}`
4. Server creates a new `CallHandler` instance for this session
5. `CallHandler.start_call()` synthesizes Sara's greeting and puts it on the response queue
6. Greeting audio streams to the browser and plays

**Sara says:** *"آداب! میں سارہ ہوں، City Medical Hospital کی receptionist۔ آپ سے بات کر کے خوشی ہوئی۔ کیا آپ English میں بات کرنا چاہیں گے یا اردو میں؟"*

---

### Phase 2: Audio capture loop

The browser runs a continuous audio capture loop:

```
Microphone → Web Audio API → ScriptProcessor (2048 samples/callback)
         → Float32Array → ArrayBuffer → WebSocket binary frame → Server
```

The server's `receive_loop()` passes every incoming binary frame to `CallHandler.process_audio_chunk()`.

---

### Phase 3: Voice Activity Detection (VAD)

Each audio chunk is processed by **Silero VAD** — a small neural network that outputs a probability (0–1) of whether the chunk contains human speech.

- **Above threshold (0.35):** chunk is added to the `_speech_buffer`. The `_pre_buffer` (last 300 ms of audio, maintained at all times) is prepended so no onset syllable is lost.
- **Below threshold:** silence is measured. When silence exceeds `_silence_gate` (2.0 s for Urdu, 1.5 s for English), turn processing is triggered.

**Barge-in protection:** While Sara is speaking, her voice echoes into the caller's microphone. The handler tracks a `_playback_end_estimate` (audio duration + 0.6 s margin). Any VAD positive during this window is discarded — Sara cannot accidentally trigger herself.

---

### Phase 4: Whisper transcription

When a silence gap is detected, the accumulated `_speech_buffer` is sent to **Whisper large-v3**:

- Runs in a thread pool so it doesn't block the asyncio event loop
- In bilingual mode (no language chosen yet): `language=None` → auto-detect → returns transcript + audio language code
- In Urdu mode: `language="ur"` → forced Urdu decoding → higher accuracy

**Output:** transcript text + confidence score (log-probability converted to 0–1)

---

### Phase 5: STT Quality Filter

Three checks run on the transcript before it reaches the dialogue layer:

| Check | English threshold | Urdu threshold |
|---|---|---|
| Confidence | ≥ 0.50 | ≥ 0.40 |
| Minimum length | 2 chars | 2 chars |
| Gibberish detection | Heuristic patterns | Heuristic patterns |

**If any check fails:** Sara plays a retry prompt directly (no LLM) → "Could you please repeat that?" → turn ends.

---

### Phase 6: Bilingual language detection (port 8002 only)

If language has not yet been chosen, the agent's bilingual branch runs:

1. `detect_language_preference(transcript)` checks for "english", "urdu", Urdu-script characters, phonetic variants
2. If text is ambiguous, Whisper's audio-level language code overrides (e.g., Whisper heard Urdu audio but transcribed it as "or do" in Latin script)
3. Language is set. The agent yields a short acknowledgment and a `switch_language` action
4. Call handler updates language, TTS voice, silence gate, Fast Engine language — all in-place on the same WebSocket
5. Sara greets in the chosen language and the call continues as if the caller had connected to port 8000 or 8001 directly

---

### Phase 7: Semantic validation & intent detection

Two lightweight checks run before the LLM:

**Semantic validator:** Checks whether the input is meaningful given the current dialogue state. Example: if Sara just asked for a name and the caller responds with a 20-word medical sentence containing no name-like words, this is flagged as invalid → Sara asks again.

**Intent classifier** (rule-based, no LLM):

| Detected intent | Example input |
|---|---|
| `provide_name` | "میرا نام احمد ہے" |
| `provide_phone` | "03001234567" |
| `describe_symptoms` | "بخار اور سر میں درد ہے" |
| `choose_date` | "کل" / "Monday" / "15 June" |
| `choose_time` | "دس بجے" / "10:30 AM" |
| `confirm` | "جی" / "ہاں" / "yes" |
| `deny` | "نہیں" / "no" |

---

### Phase 8: Fast Dialogue Engine (structured inputs)

For structured booking data, the **Fast Dialogue Engine** handles the turn instantly — zero LLM calls, zero latency:

| Dialogue state | What it extracts | Fallback to LLM if |
|---|---|---|
| `collect_name` | 1–4 word name after stripping filler phrases | Input too long, contains digits, or matches non-name vocabulary |
| `collect_phone` | Pakistani mobile number (03xx-xxxxxxx) via regex | No digit sequence found, or length wrong |
| `collect_date` | Day names, relative terms (آج/کل/today/tomorrow), numeric dates | Unrecognisable format |
| `collect_time` | Spoken time with Urdu number words, AM/PM indicators | Parsed time not in DB slot list |
| `confirm_booking` | Affirmative response | Negative or ambiguous → LLM handles |

When the Fast Engine collects the **last required field** (appointment time), it does not wait for the next caller turn. It immediately chains to the LLM with a neutral trigger ("جی"/"okay") — the LLM receives an `ALL_INFO_COLLECTED` hint in the prompt and generates the booking summary right away.

---

### Phase 9: LLM Agent (open-ended turns)

When the Fast Engine cannot handle a turn, it falls through to **HospitalAgent**. The agent:

1. Builds a full system prompt including: Sara's persona, today's date, all active doctors with fees, the booking flow in the correct language, output format rules, and hospital knowledge
2. Injects live context: current `_collected` fields, available DB slots for the chosen doctor/date, and `ALL_INFO_COLLECTED` hint when all fields are present
3. Sends conversation history (last 8 turns) + new user message to the LLM
4. **Streams** the response token-by-token

As tokens arrive, a regex detects sentence boundaries in the JSON `"speech"` field. **Each completed sentence is immediately sent to TTS** — the caller starts hearing Sara's voice before the LLM finishes generating. This is how Sara achieves low perceived latency even with large models.

**LLM output format:**
```json
{
  "speech": "آپ کا نام کیا ہے؟",
  "action": "ask",
  "data": {
    "patient_name": "Ahmed",
    "patient_phone": "03001234567"
  }
}
```

The `data` field is merged into `_collected` after each turn. The state machine derives the next required field purely from what is still missing in `_collected`.

---

### Phase 10: TTS synthesis and streaming

Each sentence from the LLM is synthesized by **Edge-TTS**:

- English: `en-US-AriaNeural`
- Urdu: `ur-PK-UzmaNeural`

Audio is synthesized and **queued immediately** as binary WebSocket frames. The browser's audio player starts each sentence as soon as its bytes arrive.

**Micro-acknowledgment:** Before the LLM responds, a pre-synthesized filler ("جی..." / "Okay..." / "اچھا...") plays from an in-memory cache. This masks the LLM latency — callers hear an immediate response.

**Audio cache:** TTS output is cached on disk by an MD5 hash of (text + voice + tone). Repeated phrases (greetings, retry prompts) are served from disk with zero synthesis time.

---

### Phase 11: Confirmation and booking

When all six fields are collected (name, phone, reason, doctor, date, time), the LLM generates a booking summary. After the summary, the handler automatically appends a hardcoded confirmation line:

> *"آپ کی appointment fee Rs. 1500 ہے۔ confirm کرنے کے لیے WhatsApp پر payment screenshot بھیجیں۔ کیا میں appointment confirm کروں؟"*

This line is **never generated by the LLM** — it is fetched directly from the database (doctor fee) and injected by the call handler. This prevents the LLM from hallucinating incorrect fees.

When the caller confirms:

1. `BookingSystem.book_appointment()` writes the appointment to `data/appointments.db`
2. Sara reads out the booking ID
3. `_flow_state` transitions to `post_booking`
4. Sara asks if there is anything else
5. **Inactivity monitor:** if the caller goes silent for 10 seconds after booking, Sara says goodbye and closes the WebSocket

---

### Summary diagram

```
Browser mic → [WebSocket] → CallHandler
                               │
                           [VAD — Silero]
                               │ silence gap detected
                           [Whisper STT]
                               │ transcript + confidence
                           [STT Quality Filter]
                               │ passes
                           [Semantic Validator]
                               │
                           [Intent Classifier]
                               │
                    ┌──────────┴──────────┐
                    │                     │
              [Fast Engine]         [LLM Agent]
           (structured data)    (open-ended turns)
                    │                     │
                    └──────────┬──────────┘
                               │ response text (streaming)
                           [Edge-TTS]
                               │ audio bytes (streaming)
                           [WebSocket] → Browser speaker
                               │
                          (on confirm)
                           [SQLite DB] ← appointment written
```

---

## 8. Component Deep-Dive

### Voice Activity Detection (`src/vad/detector.py`)
- Model: Silero VAD (< 1 MB neural network)
- Input: 30 ms audio frames at 16 kHz
- Output: speech probability (0–1)
- State is maintained across chunks within a call; reset between calls
- Threshold: configurable via `VAD_THRESHOLD` (default 0.35)

### Speech-to-Text (`src/stt/transcriber.py`)
- Model: Whisper `large-v3` via `faster-whisper` (CTranslate2 backend)
- GPU: `int8_float16` quantisation (half VRAM, negligible accuracy loss)
- CPU: `int8` quantisation
- Runs in `asyncio` thread pool — never blocks the event loop
- Returns: transcript text, confidence score, detected language code

### Fast Dialogue Engine (`src/dialogue/fast_engine.py`)
- No LLM. Pure rule-based extraction with regex and keyword matching
- Handles: name, phone (Pakistani formats), date (Urdu/English day names, relative terms), time (Urdu number words, AM/PM, digit patterns)
- After time extraction: validates against real DB slots, snaps to nearest slot within 30 minutes
- Response latency: < 50 ms

### LLM Agent (`src/llm/agent.py`)
- System prompt built fresh each turn (includes live doctor list, today's date, available slots)
- History capped at 16 messages (8 turns) to control token cost
- Doctor name auto-extraction from speech text when LLM omits it from `data` field
- State machine: derives current booking step from `_collected` — self-correcting if a field is overwritten

### Call Handler (`src/pipeline/call_handler.py`)
- Owns the `asyncio.Queue` response queue (audio bytes + JSON control messages)
- Queue uses `None` sentinel to signal stream end — never exits loop on `_is_active` flag (to avoid dropping queued items)
- `_inactivity_monitor`: background task, fires farewell + close after 10 s silence post-booking
- Pre-buffer: 300 ms rolling buffer prepended to every speech chunk

---

## 9. LLM Fallback Chain

The system tries LLM providers in order, automatically falling back if one fails:

```
Gemini (primary, Urdu)
  ↓ 503 / timeout (12 s per key) / quota exceeded
Groq — llama-3.3-70b-versatile (fast fallback)
  ↓ timeout (10 s) / quota exceeded
Ollama — qwen2.5:14b-instruct (local, always available)
  ↓ GPU out of VRAM
Ollama on CPU (slow but guaranteed)
```

**Key details:**
- Up to **6 Gemini keys** rotate automatically; exhausted keys are skipped for the rest of the day
- **12-second timeout** per Gemini key — prevents 30–60 s hangs during Google service overload
- **10-second timeout** on Groq — fast enough that callers barely notice the fallback
- For **English calls**, Gemini is skipped entirely (Groq → Ollama), as Ollama gives good English output and Gemini keys are preserved for Urdu

---

## 10. Bilingual Mode

Port 8002 serves the bilingual selector. This is the recommended entry point for a hospital that serves both English and Urdu patients.

**How it works:**

1. Sara greets in mixed Urdu/English (each language synthesized with its own TTS voice)
2. Whisper transcribes the caller's language choice with `language=None` (auto-detect)
3. Text + audio-level language signal are combined to determine the choice
4. A `switch_language` action updates the call in-place — no disconnect, no reload:
   - Agent language updated
   - **Fast Engine language updated** (critical — it has its own copy)
   - TTS voice switched
   - Silence gate adjusted (Urdu needs 2.0 s; English 1.5 s)
5. Sara greets in the chosen language and the call proceeds normally

The caller experience: a seamless transition. The WebSocket connection never drops.

---

## 11. Database & Booking

**Database:** SQLite, stored at `data/appointments.db`  
**ORM:** SQLModel (Pydantic + SQLAlchemy)

### Tables

| Table | Purpose |
|---|---|
| `Appointment` | Confirmed bookings: patient name, phone, doctor, department, date, time, reason, status, transcript |
| `Doctor` | Active doctors: name, department, specialty, fee |
| `Department` | Hospital departments |
| `DoctorSlot` | Doctor availability: start_time and end_time per doctor |

### Slot availability logic

`BookingSystem.get_available_slots(doctor, date)`:
1. Reads the doctor's `DoctorSlot` (e.g., 09:00 AM – 05:00 PM)
2. Generates 30-minute slots within that range
3. Subtracts slots already booked in `Appointment` for that doctor+date
4. Returns available time strings for the next 5 days

### Booking write

`BookingSystem.book_appointment()` inserts an `Appointment` record with `status="confirmed"`. It returns a booking ID (e.g., `APT-00042`) that Sara reads to the caller.

---

## 12. Admin Dashboard

Access: `http://localhost:8000/dashboard`

The dashboard provides a web UI to:
- View all appointments (with filter by date, doctor, status)
- Add / edit / deactivate doctors
- Manage departments
- Set doctor availability slots
- Edit hospital knowledge (hours, policies, WhatsApp number) — changes take effect on the next LLM call without restarting the server

All dashboard operations use the REST API endpoints defined in `src/api/server.py`:

| Method | Endpoint | Action |
|---|---|---|
| GET | `/api/appointments` | List appointments |
| GET | `/api/doctors` | List doctors |
| POST | `/api/doctors` | Add doctor |
| GET | `/api/departments` | List departments |
| GET | `/api/slots` | Get doctor slots |
| GET | `/api/knowledge` | Get hospital knowledge |
| PATCH | `/api/knowledge` | Update hospital knowledge |

---

## 13. Call Logs

Every call is saved to `data/call_logs/{date}_{session_id}.json` when the WebSocket closes.

A log file contains:
```json
{
  "session_id": "session-1783965396574",
  "language": "ur",
  "start_time": "2026-07-13T14:23:16",
  "end_time": "2026-07-13T14:27:42",
  "flow_state": "post_booking",
  "collected": {
    "patient_name": "Ahmed Khan",
    "patient_phone": "03001234567",
    "reason": "بخار اور سر درد",
    "doctor_name": "Dr. Fatima",
    "appointment_date": "2026-07-14",
    "appointment_time": "10:00 AM"
  },
  "transcript": [
    {"role": "assistant", "text": "آداب! میں سارہ ہوں..."},
    {"role": "user", "text": "اردو"},
    ...
  ]
}
```

Call logs are the primary debugging tool when a call did not complete a booking. Check `flow_state` to see where the conversation ended and `collected` to see what data was gathered.

---

## 14. Troubleshooting

### Server won't start

**"CUDA not available" warning** — Either `WHISPER_DEVICE=cpu` is set, or CUDA is not installed. CPU mode works fine but transcription takes ~3–5 s instead of ~0.4 s.

**"Port already in use"** — `main.py` auto-kills processes on ports 8000–8002 at startup. If it fails, run:
```bat
netstat -ano | findstr ":8000"
taskkill /PID <pid> /F
```

**"Ollama model not found"** — Run `ollama pull qwen2.5:14b-instruct`

### Calls not working

**Microphone not detected** — The browser must be served over `localhost` or `https`. HTTP on a remote IP blocks microphone access. If accessing from another device on the network, use a reverse proxy with TLS, or access via `http://localhost`.

**Sara not responding** — Check terminal logs. Common causes:
- Whisper is still loading (wait for "Loading models..." to complete)
- Groq/Gemini quota exhausted → Ollama will take over but is slower
- VAD threshold too high — caller's audio is too quiet

**Sara responds in English after bilingual switch** — This would indicate `_fast_engine.language` was not updated. Check that `call_handler.py` includes `self.agent._fast_engine.language = lang` in the `switch_language` action handler.

### LLM issues

**Gemini 503 errors** — Normal during Google service overload. The system automatically times out after 12 s per key and falls through to Groq. No action needed.

**Slow responses** — If Gemini is exhausted and Groq is also busy, Ollama handles the call. Ollama on GPU takes ~2–3 s for first token; on CPU ~10–20 s. Consider adding more Gemini keys or increasing Groq quota.

**LLM hallucinating doctor fees** — The fee is never generated by the LLM. It is fetched from `Doctor.fee` in the database and injected by `_play_confirm_reminder()` in the call handler. If the fee is wrong, update it in the admin dashboard.

---

## Technical Documentation

For a complete technical explanation of every component, design decision, and implementation detail, see:

- **[docs/sara-call-flow.html](docs/sara-call-flow.html)** — Visual side-by-side flow diagrams for English and Urdu calls
- **[docs/sara-technical.html](docs/sara-technical.html)** — Full technical architecture reference (17 sections)
