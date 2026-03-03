# 🛡️ PolicyBot AI v3 — Advanced Insurance Advisor

**Full-stack AI insurance advisor with Gemini Vision document verification.**

## ✨ What's in v3

| Feature | Details |
|---|---|
| 🔍 **Gemini Vision Verification** | Real AI image analysis — reads ID, extracts DOB, compares with stated age |
| 📄 **Document Quality Check** | Detects blurry/dark/cropped/unreadable documents |
| 🔘 **Smart Radio Options** | Auto shown after upload failure with "Upload Clear / Continue Without" |
| ⏳ **Bot Waits for Result** | Never advances step until vision analysis completes |
| 🗑️ **Auto-delete Documents** | All uploads deleted at Step 15 (farewell) |
| 📋 **Strict 15-step Flow** | Insurance type → Name → Age → ID → Verify → ... → Delete → Farewell |
| 🔐 **Secure by Design** | ID numbers NEVER stored — only verification status |
| 🎯 **Animated Options** | Neon glow radio/multi buttons with ripple + pulse |
| ⭐ **5-star Ratings** | Stored in DB with comments |
| 👨‍💼 **Human Escalation** | Phone + callback time, stored as lead |
| 🎉 **Farewell Overlay** | Confetti animation + secure deletion notice |
| 🔒 **Admin Login** | Session-based auth with ADMIN_ID + ADMIN_PASSWORD |
| 📊 **9 Admin Sections** | Dashboard, Users, Chats, Verifications, Leads, Ratings, Docs, Fraud, API Keys |
| 📈 **Verification Analytics** | Track verified/failed/skipped per user |

## 🚀 Quick Start

```bash
# 1. Copy env file and add your Gemini key
cp .env.example .env
# Edit .env and add GEMINI_API_KEY_1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
```

Open **http://localhost:5000** for the chat.
Open **http://localhost:5000/admin/login** for the admin panel.

## 🔑 Get Free Gemini API Key
→ https://aistudio.google.com/app/apikey

## 🚂 Deploy to Railway (Free)
1. Push to GitHub
2. Go to railway.app → New Project → Deploy from GitHub
3. Add environment variables in Railway:
   - `GEMINI_API_KEY_1` (required)
   - `ADMIN_ID` / `ADMIN_PASSWORD`
   - `SECRET_KEY` (random string)

## 📄 Document Verification Logic

```
User uploads ID
    ↓
Gemini Vision analyzes image
    ↓
Is it a valid Government ID?
    → No  → "Not a valid ID" + options
    ↓
Is image quality good?
    → No  → "Blurry/Cropped" + options
    ↓
Is DOB readable?
    → No  → "Can't read DOB" + options
    ↓
Does DOB match stated age? (±2 year tolerance)
    → No  → "Age mismatch" + options
    → Yes → ✅ VERIFIED
```

After failure, user sees radio buttons:
- "Upload Clear Document" → retry
- "Continue Without Verification" → mark NOT VERIFIED, proceed

## 🔒 Privacy Rules
- Aadhaar / PAN numbers NEVER stored
- Only verification STATUS is stored (verified/failed/skipped)
- Uploaded documents AUTO-DELETED at end of conversation
- YYYY birth year stored only (no full DOB)
