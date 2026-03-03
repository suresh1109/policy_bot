# PolicyBot — Offline OCR + Gemini AI Pipeline

## How it works now

### Document Verification (Dual Pipeline)
1. User uploads document (JPG/PNG/WEBP/GIF/BMP/TIFF/PDF/DOCX/TXT)
2. **Step 1 — Offline OCR** (pytesseract + OpenCV, NO API needed):
   - OpenCV: quality check (blur/dark/brightness detection)
   - OpenCV: 5 preprocessed versions (Otsu, adaptive, denoised, sharpened, upscaled)
   - pytesseract: OCR with 4 psm configs, picks best result
   - Regex: extract DOB, ID type, name
   - Compare DOB with stated age (±2 years tolerance)
3. **Step 2 — Gemini Vision** (fallback, only if OCR fails AND API key present):
   - Only triggered if OCR produced no text (truly unreadable image)
   - Uses GEMINI_API_KEY or GEMINI_API_KEY_1..5
4. After verification → **Gemini AI takes conversation control** for all Q&A

### Supported Document Types
| Type | Doc | Extract |
|------|-----|---------|
| 🪪 Gov ID | Aadhaar, PAN, DL, Passport, Voter ID | DOB, ID type |
| 🏥 Health | Lab reports, prescriptions | Conditions, doctor, diagnosis |
| 🚗 Vehicle | RC Book, vehicle insurance | Reg no, model, year |
| 📋 Policy | Previous insurance policy | Policy no, coverage, claims |

### .env setup
```
GEMINI_API_KEY=your_key_here          # OR
GEMINI_API_KEY_1=key1
GEMINI_API_KEY_2=key2
```
If no Gemini key → OCR still works fully for verification.
Gemini needed only for AI conversation responses.

## Files changed
- `app.py` — new dual-pipeline upload route
- `models/ocr_verifier.py` — full offline OCR engine (PyPDF2→pypdf fixed)
- `models/doc_verifier.py` — Gemini Vision fallback (all key formats)
- `models/conversation_engine.py` — name-skip bug fixed
- `models/database.py` — reset_session_profile added
- `templates/index.html` — all doc types in dropdown, 5-step overlay
- `static/css/main.css` — OCR badge, verif engine badge
- `static/js/app.js` — 5-step animation, all formats, engine display
