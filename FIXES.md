# PolicyBot Patch 3 — Bug Fixes

## CRITICAL FIX: Name step no longer skipped
**Root cause:** `_next()` was checking `profile.get("insurance_type")` which was 
already populated from the previous DB session. So when user selected insurance,
the engine saw insurance_type already set AND name already set (from old session),
causing it to jump straight to age/doc_upload.

**Fix:** `_next()` now only uses `extracted` (data from the CURRENT message),
NOT from profile. Stage only advances if the required field was extracted THIS turn.

## CRITICAL FIX: Session data reset on page load
`startBot()` now sends `is_new_chat: true` to `/api/chat` which calls
`db.reset_session_profile(user_id)` — clears name/age/city/insurance_type etc.
Prevents stale data from previous sessions appearing in sidebar or affecting flow.

## CRITICAL FIX: Vision API "temporarily busy" error
**Root cause:** `_call_vision()` was only checking `GEMINI_API_KEY_1..4`.
If your .env has `GEMINI_API_KEY` (no number), it was never found → no keys → api_error.

**Fix:** Now checks GEMINI_API_KEY first, then GEMINI_API_KEY_1 through GEMINI_API_KEY_5.
Also fixed the content parts format for Gemini Vision (uses `inline_data` dict format).

## NEW: Additional image formats supported
Upload now accepts: JPG, JPEG, PNG, WEBP, GIF, BMP, TIFF, TIF, PDF
(previously only JPG, PNG, WEBP, PDF)

## HOW TO APPLY
Replace these files in your policybot-2.2/ folder:
- app.py
- models/conversation_engine.py
- models/doc_verifier.py
- models/database.py
- models/lead_manager.py
- templates/index.html
- static/css/main.css
- static/js/app.js
