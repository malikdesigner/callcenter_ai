"""
Quick test — run this to see which Gemini models your API key can actually use.
Usage: python test_gemini.py
"""
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY2") or os.getenv("GEMINI_API_KEY")

print(f"Using key: ...{API_KEY[-6:]}\n")

# ── Step 1: List available models via REST (no SDK needed) ────────────────────
print("=" * 60)
print("AVAILABLE MODELS ON YOUR ACCOUNT:")
print("=" * 60)
import httpx
try:
    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": API_KEY},
        timeout=10,
    )
    data = r.json()
    available = []
    for m in data.get("models", []):
        name = m.get("name", "")
        if "gemini" in name.lower():
            short = name.replace("models/", "")
            available.append(short)
            print(f"  {short}  —  {m.get('displayName','')}")
    if not available:
        print(f"  (none found — raw response: {data})")
except Exception as e:
    print(f"  ERROR: {e}")
    available = []

# ── Step 2: Test each model with a live call ──────────────────────────────────
print("\n" + "=" * 60)
print("LIVE CALL TEST (OpenAI-compat endpoint):")
print("=" * 60)

from openai import OpenAI

candidates = [
    # Latest
    "gemini-3.5-flash",
    "gemini-3.0-flash",
    # Previous generation
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
]

# Add any models found from the list above that aren't already in candidates
for m in available:
    if m not in candidates:
        candidates.append(m)

oa_client = OpenAI(
    api_key=API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

working = []
for model in candidates:
    try:
        resp = oa_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        answer = resp.choices[0].message.content
        print(f"  ✅ {model:40s} → {answer!r}")
        working.append(model)
    except Exception as e:
        err = str(e)
        code = "429" if "429" in err else ("404" if "404" in err else "ERR")
        short_err = err[:100].replace("\n", " ")
        print(f"  ❌ {model:40s} → [{code}] {short_err}")

# ── Step 3: Native Google SDK test ────────────────────────────────────────────
print("\n" + "=" * 60)
print("NATIVE SDK TEST (google-genai):")
print("=" * 60)
try:
    from google import genai as google_genai
    g_client = google_genai.Client(api_key=API_KEY)
    response = g_client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say OK"
    )
    print(f"  ✅ Native SDK gemini-2.0-flash → {response.text!r}")
except ImportError:
    print("  (google-genai not installed — skipping native SDK test)")
    print("  Install with: pip install -U google-genai")
except Exception as e:
    print(f"  ❌ Native SDK error: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY:")
print("=" * 60)
if working:
    print(f"  Working models: {working}")
    print(f"\n  → Set this in .env:  GEMINI_MODEL={working[0]}")
else:
    print("  ❌ No models worked with this key.")
    print("  → This key has no free-tier quota in your region.")
    print("  → Try: a different Gmail account, a VPN, or enable billing.")
