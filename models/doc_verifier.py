"""
DocumentVerifier v5 — Full fix
KEY FIXES:
  - Accepts GEMINI_API_KEY, GEMINI_API_KEY_1..4, GEMINI_API_KEY_5 (any format)
  - Correct Gemini Vision multimodal content format (Part objects)
  - Handles JPEG, PNG, WEBP, GIF, BMP, TIFF, PDF
  - Robust JSON extraction (strips markdown fences, finds JSON blocks)
  - Clear per-failure-type error messages
  - Console log: "Document uploaded for {session_id}"
  - Never stores ID numbers — only DOB year and boolean
"""
import os, base64, re, datetime, json, logging

log = logging.getLogger("DocVerifier")


class DocumentVerifier:

    VISION_PROMPT = """Analyze this document image as a document verification assistant.

TASK — answer ALL fields honestly:
1. Is this a valid Indian Government ID? (Aadhaar / PAN Card / Driving License / Passport / Voter ID)
2. Is the image quality good enough to read? (check: blur, darkness, cropping, completeness)
3. Can you clearly read the Date of Birth (DOB) from this document?
4. What exact ID type is it?

PRIVACY RULES — STRICTLY FOLLOW:
- DO NOT output any Aadhaar number, PAN number, DL number or ID number
- DO NOT output the person's full name
- ONLY output Date of Birth if clearly visible
- Be honest about image quality problems

IMAGE QUALITY DEFINITIONS:
  good       = sharp, well-lit, fully visible, all text readable
  blurry     = text not in focus / shaky
  dark       = under-exposed / too dark
  cropped    = document edges cut off
  incomplete = part of document missing
  unreadable = cannot read content

OUTPUT ONLY VALID JSON — NO other text, NO markdown fences, NO explanation:
{
  "is_valid_id": true or false,
  "id_type": "Aadhaar" | "PAN" | "Driving License" | "Passport" | "Voter ID" | "Unknown" | "Not an ID",
  "image_quality": "good" | "blurry" | "dark" | "cropped" | "incomplete" | "unreadable",
  "dob_visible": true or false,
  "dob": "DD/MM/YYYY" or null,
  "dob_confidence": "high" | "medium" | "low" | "none",
  "notes": "brief note — no ID numbers allowed"
}"""

    BAD_QUALITIES = {"blurry","dark","cropped","incomplete","unreadable"}
    QUALITY_TIPS  = {
        "blurry":     "📸 Tip: Hold your phone steady and tap the screen to focus before capturing.",
        "dark":       "💡 Tip: Move to a brighter area or turn on your phone flashlight.",
        "cropped":    "📐 Tip: Make sure all 4 corners of the ID are visible in the frame.",
        "incomplete": "📋 Tip: Ensure the complete document is visible in the photo.",
        "unreadable": "🔍 Tip: Place the document flat and photograph from directly above.",
    }

    def __init__(self, gemini_manager):
        self.gemini = gemini_manager

    # ─── Main verify pipeline ──────────────────────────────────────────────
    def verify(self, file_path, file_bytes, file_ext, doc_type,
               stated_age, user_id, session_id=""):
        sid = session_id or user_id
        print(f"[CONSOLE] Document uploaded for {sid}")
        log.info(f"[UPLOAD] Document uploaded for {sid} | ext={file_ext} | size={len(file_bytes)}")

        b64  = base64.b64encode(file_bytes).decode("utf-8")
        mime = self._mime(file_ext)

        analysis = self._call_vision(b64, mime, file_ext)

        if analysis is None:
            return self._result(
                "api_error", False,
                "I'm sorry 😊 Our verification service is temporarily busy. Please try again in a moment.",
                options=["Upload Document Again", "Continue Without Verification"]
            )

        is_valid  = analysis.get("is_valid_id", False)
        id_type   = analysis.get("id_type", "Unknown")
        quality   = analysis.get("image_quality", "unreadable")
        dob_vis   = analysis.get("dob_visible", False)
        dob_raw   = analysis.get("dob")
        dob_conf  = analysis.get("dob_confidence", "none")
        notes     = analysis.get("notes", "")

        log.info(f"[VERIFY] valid={is_valid} id={id_type} quality={quality} dob_visible={dob_vis} conf={dob_conf}")

        if not is_valid or id_type in ("Unknown","Not an ID",""):
            return self._result(
                "not_valid_id", False,
                "I'm sorry 😊 This doesn't look like a valid Government ID. "
                "Please upload Aadhaar, PAN, Driving License, Passport, or Voter ID.",
                quality=quality, id_type=id_type,
                options=["Upload Correct Document","Continue Without Verification"]
            )

        if quality in self.BAD_QUALITIES:
            tip = self.QUALITY_TIPS.get(quality, "Please try taking a clearer photo.")
            return self._result(
                "low_quality", False,
                f"I'm sorry 😊 I couldn't read your document clearly — it appears to be {quality}. {tip}",
                quality=quality, id_type=id_type,
                options=["Upload Clear Document","Continue Without Verification"]
            )

        if not dob_vis or dob_conf in ("none","low") or not dob_raw:
            return self._result(
                "dob_not_found", False,
                "I'm sorry 😊 I couldn't clearly read the Date of Birth on your document. "
                "Please upload a clearer version or a different ID.",
                quality=quality, id_type=id_type,
                options=["Upload Clear Document","Continue Without Verification"]
            )

        matched, dob_year = self._check_age(dob_raw, stated_age)

        if matched is None:
            return self._result(
                "dob_parse_error", False,
                "I'm sorry 😊 I found a date on your document but couldn't calculate age from it. "
                "Please try a different ID.",
                quality=quality, id_type=id_type,
                options=["Upload Clear Document","Continue Without Verification"]
            )

        if matched:
            return self._result(
                "verified", True,
                f"✅ Your {id_type} has been verified successfully! 👍 "
                "Your identity is confirmed — let's continue!",
                quality=quality, id_type=id_type, dob_year=dob_year, notes=notes
            )
        else:
            return self._result(
                "age_mismatch", False,
                "I couldn't verify your document. The age on your ID doesn't match what you told me. "
                "Please upload a different ID or continue without verification.",
                quality=quality, id_type=id_type,
                options=["Upload Different Document","Continue Without Verification"]
            )

    # ─── Gemini Vision call — supports all key formats ─────────────────────
    def _call_vision(self, b64, mime, file_ext):
        try:
            import google.generativeai as genai
        except ImportError:
            log.error("[VERIFY] google-generativeai not installed")
            return None

        # Collect all API keys — supports GEMINI_API_KEY, GEMINI_API_KEY_1..5
        keys = []
        # Try plain key first
        k = os.environ.get("GEMINI_API_KEY","").strip()
        if k:
            keys.append(k)
        # Then numbered keys
        for i in range(1, 6):
            k = os.environ.get(f"GEMINI_API_KEY_{i}","").strip()
            if k and k not in keys:
                keys.append(k)

        if not keys:
            log.error("[VERIFY] No GEMINI_API_KEY found in environment. "
                      "Set GEMINI_API_KEY or GEMINI_API_KEY_1 in .env")
            return None

        log.info(f"[VERIFY] Found {len(keys)} API key(s), trying Vision analysis…")

        for attempt, key in enumerate(keys):
            try:
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-2.0-flash")

                # Build content parts correctly for Gemini Vision
                if file_ext.lower() == ".pdf":
                    # PDF: inline_data part
                    content = [
                        {"inline_data": {"mime_type": "application/pdf", "data": b64}},
                        self.VISION_PROMPT,
                    ]
                else:
                    # Image: inline_data part (correct format for all image types)
                    content = [
                        {"inline_data": {"mime_type": mime, "data": b64}},
                        self.VISION_PROMPT,
                    ]

                response = model.generate_content(content)
                raw = response.text.strip()
                log.debug(f"[VERIFY] Raw response (key {attempt+1}): {raw[:300]}")

                return self._parse_json(raw)

            except Exception as e:
                err = str(e).lower()
                if any(x in err for x in ["quota","429","resource_exhausted","rate_limit","limit exceeded"]):
                    log.warning(f"[VERIFY] Key {attempt+1} quota exceeded, rotating…")
                    continue
                elif any(x in err for x in ["api_key","permission","invalid","credentials","not valid"]):
                    log.warning(f"[VERIFY] Key {attempt+1} invalid/permission error")
                    continue
                elif any(x in err for x in ["timeout","deadline","connection","unavailable"]):
                    log.warning(f"[VERIFY] Key {attempt+1} timeout/connection error")
                    continue
                else:
                    log.error(f"[VERIFY] Unexpected error with key {attempt+1}: {e}")
                    # Don't continue — unexpected errors may not be key-specific
                    return None

        log.error("[VERIFY] All API keys failed")
        return None

    # ─── JSON extraction (handles markdown fences, extracts JSON block) ────
    def _parse_json(self, raw):
        if not raw:
            return None
        # Strip markdown fences
        text = re.sub(r'```(?:json)?\s*', '', raw).strip()
        text = re.sub(r'```\s*$', '', text).strip()

        # Try full parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting first {...} block (greedy)
        m = re.search(r'\{[\s\S]*\}', text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass

        # Try fixing common Gemini response issues (true/false as strings)
        text2 = text.replace("True","true").replace("False","false").replace("None","null")
        try:
            return json.loads(text2)
        except json.JSONDecodeError:
            pass

        log.warning(f"[VERIFY] Could not parse JSON from: {raw[:200]}")
        return None

    # ─── Age check ─────────────────────────────────────────────────────────
    def _check_age(self, dob_str, stated_age):
        if not dob_str or stated_age is None:
            return None, ""
        dob = None
        formats = [
            "%d/%m/%Y","%d-%m-%Y","%Y-%m-%d","%d/%m/%y","%d.%m.%Y",
            "%m/%d/%Y","%d %B %Y","%d %b %Y","%B %d, %Y","%b %d, %Y",
            "%Y/%m/%d","%d %m %Y",
        ]
        for fmt in formats:
            try:
                dob = datetime.datetime.strptime(dob_str.strip(), fmt).date()
                break
            except ValueError:
                continue

        if not dob:
            ym = re.search(r'\b(19\d{2}|20[0-2]\d)\b', dob_str)
            if ym:
                birth_year = int(ym.group())
                calc_age   = datetime.date.today().year - birth_year
                try:
                    return abs(calc_age - int(stated_age)) <= 2, str(birth_year)
                except (ValueError,TypeError):
                    return None, ""
            return None, ""

        calc_age = (datetime.date.today() - dob).days // 365
        try:
            return abs(calc_age - int(stated_age)) <= 2, str(dob.year)
        except (ValueError,TypeError):
            return None, ""

    # ─── MIME type for ALL supported image formats ─────────────────────────
    def _mime(self, ext):
        return {
            ".jpg":"image/jpeg",".jpeg":"image/jpeg",
            ".png":"image/png",".webp":"image/webp",
            ".gif":"image/gif",".bmp":"image/bmp",
            ".tiff":"image/tiff",".tif":"image/tiff",
            ".pdf":"application/pdf",
        }.get(ext.lower(), "image/jpeg")

    def _result(self, status, verified, message, quality="", id_type="",
                dob_year="", notes="", options=None):
        return {
            "status":status, "verified":verified, "message":message,
            "quality":quality, "doc_type_found":id_type,
            "dob_year":dob_year, "notes":notes,
            "options":options or [],
            "option_type":"radio" if options else "none",
        }

    @staticmethod
    def delete_file(fp):
        try:
            if fp and os.path.exists(fp):
                os.remove(fp); return True
        except Exception as e:
            log.error(f"[CLEANUP] {e}")
        return False

    @staticmethod
    def delete_user_uploads(user_id, db):
        docs = db.get_user_documents(user_id)
        deleted = sum(1 for d in docs if DocumentVerifier.delete_file(d.get("file_path","")))
        db.delete_user_documents(user_id)
        log.info(f"[CLEANUP] Deleted {deleted} files for {user_id}")
        return deleted
