"""
PolicyBot v4 — Flask Backend
Offline OCR verification (pytesseract + OpenCV) → Gemini AI handoff
Document auto-delete after conversation ends
"""
import os, uuid, logging
from functools import wraps
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, Response
from flask_cors import CORS
from dotenv import load_dotenv

from models.gemini_manager      import GeminiManager
from models.rag_engine          import RAGEngine
from models.database            import Database
from models.conversation_engine import ConversationEngine
from models.lead_manager        import LeadManager
from models.fraud_checker       import FraudChecker
from models.doc_verifier        import DocumentVerifier
from models.ocr_verifier        import OCRVerifier, ocr_available
from models.policy_kb           import PolicyKB, ALLOWED_EXTS as KB_EXTS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("PolicyBot")

load_dotenv()

app  = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "policybot-v3-secret-2024")
CORS(app)

# ── Singletons ────────────────────────────────────────────────────────────────
gemini     = GeminiManager()
rag        = RAGEngine()
db         = Database()
lead_mgr   = LeadManager(db)
fraud      = FraudChecker()
ocr        = OCRVerifier()                # ← Offline OCR (primary)
verifier   = DocumentVerifier(gemini)    # ← Gemini Vision (fallback only)
policy_kb  = PolicyKB(db, gemini, ocr)   # ← Policy Knowledge Base engine

# Log OCR availability on startup
_ocr_status = ocr_available()
log.info(f"[STARTUP] OCR status: {_ocr_status}")

# Seed master policy data into KB (only if no master exists yet)
try:
    seeded = db.kb_seed_master()
    log.info(f"[STARTUP] Policy KB master data: {'seeded' if seeded else 'already exists'}")
except Exception as _e:
    log.warning(f"[STARTUP] KB seed warning: {_e}")

ADMIN_ID   = os.getenv("ADMIN_ID",       "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "policybot2024")

# All supported image + document formats
ALLOWED_EXTS = {".jpg",".jpeg",".png",".webp",".gif",".bmp",".tiff",".tif",
                ".pdf",".docx",".doc",".txt"}

# ── Auth decorator ────────────────────────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated

# ── Pages ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index(): return render_template("index.html")

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        d = request.form
        if d.get("admin_id") == ADMIN_ID and d.get("password") == ADMIN_PASS:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Invalid credentials. Please try again."
    return render_template("admin_login.html", error=error)

@app.route("/admin/logout")
def admin_logout(): session.pop("admin_logged_in", None); return redirect(url_for("admin_login"))

@app.route("/admin")
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard(): return render_template("admin.html")

# ── Chat API ──────────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    d            = request.json or {}
    message      = (d.get("message") or "").strip()
    user_id      = d.get("user_id")   or str(uuid.uuid4())
    session_id   = d.get("session_id") or str(uuid.uuid4())
    lang         = d.get("language", "English")
    selected_opt = d.get("selected_option")
    is_new_chat  = d.get("is_new_chat", False)

    if not message and not selected_opt:
        return jsonify({"error": "Empty message"}), 400

    final_msg = selected_opt or message

    # Handle session init ping (from page load — just resets profile, no reply needed)
    if final_msg == "__session_init__":
        if is_new_chat:
            db.reset_session_profile(user_id)
            log.info(f"[SESSION] Init reset for {user_id}")
        return jsonify({"status":"success","reply":"","stage":"insurance_type",
                        "options":[],"option_type":"none","progress":7,
                        "stage_label":"Insurance Type","profile_updated":False})

    # ── Session isolation ─────────────────────────────────────────────────────
    # is_new_chat is sent on EVERY message in a fresh session (first real message)
    # This guarantees reset even if the __session_init__ ping was lost/raced
    fresh = False
    if is_new_chat:
        db.reset_session_profile(user_id)   # wipes profile + history + selected_options
        profile = {"onboarding_stage": "insurance_type", "user_id": user_id}
        history = []                         # ← EMPTY history — no old context bleeds through
        log.info(f"[SESSION] Fresh session reset for {user_id}")
        fresh = True
    else:
        profile = db.get_user_profile(user_id)
        if not profile:
            profile = {"onboarding_stage": "insurance_type", "user_id": user_id}
        elif not profile.get("onboarding_stage"):
            profile["onboarding_stage"] = "insurance_type"
        history = db.get_chat_history(user_id, limit=20)

    engine = ConversationEngine(gemini, rag, db)
    result = engine.process(
        user_id=user_id, session_id=session_id, message=final_msg,
        history=history, profile=profile, language=lang,
        fresh_session=fresh,
    )

    # Store button selection in DB if came from option click
    if selected_opt and profile.get("onboarding_stage"):
        db.store_option_selection(user_id, profile["onboarding_stage"], "option_click", selected_opt)

    lead_mgr.detect(user_id, final_msg)

    db.store_chat(
        user_id=user_id, message=final_msg,
        bot_reply=result["reply"], module=result.get("module","general"),
        session_id=session_id, language=lang
    )

    # Trigger document cleanup on farewell
    if result.get("trigger_cleanup"):
        _cleanup_user_docs(user_id)

    return jsonify({
        "status":          "success",
        "user_id":         user_id,
        "session_id":      session_id,
        "reply":           result["reply"],
        "options":         result.get("options", []),
        "option_type":     result.get("option_type", "none"),
        "stage":           result.get("stage", ""),
        "progress":        result.get("progress", 0),
        "stage_label":     result.get("stage_label", ""),
        "show_rating":     result.get("show_rating", False),
        "show_escalation": result.get("show_escalation", False),
        "show_farewell":   result.get("show_farewell", False),
        "show_upload":     result.get("show_upload", False),
        "confidence":      result.get("confidence", ""),
        "profile_updated": result.get("profile_updated", False),
        "lock_chat":       result.get("lock_chat", False),
        "module":          result.get("module", "general"),
    })

# ── Upload + Verify API ───────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    """
    DUAL PIPELINE:
    Step 1: Offline OCR (pytesseract + OpenCV) — no API needed
            → quality check, text extraction, DOB parse, age compare
    Step 2: If OCR succeeds → mark verified, hand control to Gemini AI
            If OCR fails (blurry/no text) → offer retry or skip
            If Gemini API key available → optionally use as fallback
    Supported: JPG PNG WEBP GIF BMP TIFF PDF DOCX TXT
    """
    doc_type   = request.form.get("doc_type", "gov_id")
    user_id    = request.form.get("user_id", "")
    session_id = request.form.get("session_id", user_id)
    file       = request.files.get("file")

    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    # ── Validate extension ──────────────────────────────────────────────────
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        friendly_list = "JPG, PNG, WEBP, GIF, BMP, TIFF, PDF, DOCX, TXT"
        return jsonify({"error": f"'{ext}' not supported. Use: {friendly_list}"}), 400

    file_bytes = file.read()
    if len(file_bytes) > 15 * 1024 * 1024:  # 15MB limit
        return jsonify({"error": "File too large. Maximum 15MB."}), 400
    if len(file_bytes) < 500:
        return jsonify({"error": "File appears empty or corrupted."}), 400

    # ── Save to disk ────────────────────────────────────────────────────────
    safe_name = f"{uuid.uuid4()}{ext}"
    save_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, safe_name)
    with open(save_path, "wb") as fout:
        fout.write(file_bytes)
    db.store_document(file.filename, save_path, doc_type, user_id)
    log.info(f"[UPLOAD] {len(file_bytes)} bytes saved → {save_path} | type={doc_type} user={user_id}")

    # ════════════════════════════════════════════════════════════════
    # GOV ID — OFFLINE OCR FIRST, GEMINI FALLBACK
    # ════════════════════════════════════════════════════════════════
    gov_id_types = {"aadhaar","pan","driving_license","passport","voter_id","gov_id"}
    if doc_type in gov_id_types:
        profile    = db.get_user_profile(user_id)
        stated_age = profile.get("age")

        # ── STEP 1: Offline OCR (primary — no API key needed) ───────────────
        log.info(f"[PIPELINE] Step 1: Offline OCR | user={user_id} | stated_age={stated_age}")
        stated_name = profile.get("name", "") or ""
        v_result = ocr.verify_gov_id(
            file_path    = save_path,
            file_bytes   = file_bytes,
            file_ext     = ext,
            stated_age   = stated_age,
            user_id      = user_id,
            session_id   = session_id,
            stated_name  = stated_name,
        )
        ocr_failed = v_result["status"] in ("no_ocr_engine","no_text","api_error")
        log.info(f"[PIPELINE] OCR result: status={v_result['status']} verified={v_result['verified']}")

        # ── STEP 2: Gemini Vision fallback if OCR couldn't extract text ─────
        # Only use Gemini if: OCR produced no text AND Gemini key is available
        gemini_keys_present = any(
            os.environ.get(k,"") for k in
            ["GEMINI_API_KEY","GEMINI_API_KEY_1","GEMINI_API_KEY_2","GEMINI_API_KEY_3"]
        )
        if ocr_failed and gemini_keys_present:
            log.info(f"[PIPELINE] Step 2: Gemini Vision fallback | user={user_id}")
            try:
                v_result = verifier.verify(
                    file_path  = save_path,
                    file_bytes = file_bytes,
                    file_ext   = ext,
                    doc_type   = doc_type,
                    stated_age = stated_age,
                    user_id    = user_id,
                    session_id = session_id,
                )
                log.info(f"[PIPELINE] Gemini fallback: {v_result['status']}")
            except Exception as ge:
                log.error(f"[PIPELINE] Gemini fallback failed: {ge}")
                # Keep the OCR result

        # ── Store verification result (never store ID numbers) ───────────────
        if v_result["verified"]:
            db.update_verification(user_id, "gov_id_verified", 1)
            db.update_verification(user_id, "doc_type_found",
                                   v_result.get("doc_type_found",""))
            db.upsert_user_profile(user_id, {"onboarding_stage":"collect_gender"})
            next_stage = "collect_gender"
            log.info(f"[PIPELINE] ✅ VERIFIED → collect_gender | user={user_id}")
        else:
            db.update_verification(user_id, "gov_id_verified", 0)
            next_stage = "doc_upload"
            log.info(f"[PIPELINE] ❌ NOT VERIFIED: {v_result['status']} | user={user_id}")

        return jsonify({
            "status":         "success",
            "verified":       v_result["verified"],
            "doc_type":       doc_type,
            "v_status":       v_result["status"],
            "quality":        v_result.get("quality",""),
            "doc_type_found": v_result.get("doc_type_found",""),
            "engine":         v_result.get("engine","offline_ocr"),
            "reply":          v_result["message"],
            "options":        v_result.get("options",[]),
            "option_type":    v_result.get("option_type","radio"),
            "next_stage":     next_stage,
            "handoff_to_gemini": v_result.get("handoff_to_gemini", False),
        })

    # ════════════════════════════════════════════════════════════════
    # CONDITION REPORT — health report uploaded from condition_report_upload stage
    # Also handles: optional_health_check, vehicle_doc_upload, life_docs,
    #               travel_declare, property_history
    # ════════════════════════════════════════════════════════════════
    elif doc_type in ("health_report", "condition_report",
                      "vehicle_insurance", "rc_book",
                      "life_doc", "travel_doc", "property_doc"):

        profile = db.get_user_profile(user_id)
        current_stage = profile.get("onboarding_stage", "")

        # ── Determine which OCR analyzer to call ─────────────────────────
        if doc_type in ("health_report", "condition_report"):
            log.info(f"[PIPELINE] Condition/Health report OCR | user={user_id}")
            result = ocr.analyze_health_report(file_bytes, ext, user_id)

            conditions_found = result.get("conditions", [])
            if conditions_found:
                existing = profile.get("medical_conditions", "")
                new_conds = ", ".join(conditions_found)
                merged = ", ".join(filter(None, [existing, new_conds]))
                db.upsert_user_profile(user_id, {
                    "medical_conditions":        merged,
                    "medical_proof_uploaded":    1,
                    "condition_report_uploaded": 1,
                    "condition_report_result":   new_conds,
                })
            else:
                db.upsert_user_profile(user_id, {
                    "medical_proof_uploaded":    1,
                    "condition_report_uploaded": 1,
                    "condition_report_result":   "No conditions detected",
                })

            # ── Always advance to collect_budget after condition report ────
            db.upsert_user_profile(user_id, {"onboarding_stage": "collect_budget"})

            reply_msg = result.get("message", "✅ Health report analyzed 👍")
            if conditions_found:
                reply_msg = (f"✅ Got it! I found {', '.join(conditions_found)} in your report. "
                             f"I'll factor this into your plan recommendation 🏥")
            else:
                reply_msg = "✅ Health report received and analyzed! No conditions found. Continuing..."

            return jsonify({
                "status":           "success",
                "verified":         result.get("success", True),
                "reply":            reply_msg,
                "conditions_found": conditions_found,
                "doctor":           result.get("doctor", ""),
                "next_stage":       "collect_budget",
                "options":          ["Under ₹500","₹500–₹1,000","₹1,000–₹2,000","₹2,000–₹5,000","Above ₹5,000"],
                "option_type":      "radio",
                "handoff_to_gemini": True,
            })

        elif doc_type in ("vehicle_insurance", "rc_book"):
            log.info(f"[PIPELINE] Vehicle doc OCR | user={user_id}")
            result = ocr.analyze_vehicle_doc(file_bytes, ext, user_id)

            db.upsert_user_profile(user_id, {
                "vehicle_doc_uploaded": 1,
                **({"vehicle_number": result["vehicle_no"]} if result.get("vehicle_no") else {}),
                "onboarding_stage": "collect_budget",
            })

            return jsonify({
                "status":       "success",
                "verified":     result.get("success", True),
                "reply":        result.get("message", "✅ Vehicle document analyzed! Let's continue."),
                "vehicle_no":   result.get("vehicle_no", ""),
                "next_stage":   "collect_budget",
                "options":      ["Under ₹500","₹500–₹1,000","₹1,000–₹2,000","₹2,000–₹5,000","Above ₹5,000"],
                "option_type":  "radio",
                "handoff_to_gemini": True,
            })

        else:
            # life_doc, travel_doc, property_doc — generic OCR + advance
            log.info(f"[PIPELINE] Supporting doc OCR ({doc_type}) | user={user_id}")
            policy_text = ocr.extract_policy_text_for_rag(file_bytes, ext)
            doc_field_map = {
                "life_doc":     "life_docs",
                "travel_doc":   "travel_declare",
                "property_doc": "property_history",
            }
            field = doc_field_map.get(doc_type, "life_docs")
            db.upsert_user_profile(user_id, {
                field: "document_uploaded",
                "onboarding_stage": "collect_budget",
            })

            return jsonify({
                "status":       "success",
                "verified":     bool(policy_text),
                "reply":        f"✅ Document received and analyzed! Let's continue 😊",
                "next_stage":   "collect_budget",
                "options":      ["Under ₹500","₹500–₹1,000","₹1,000–₹2,000","₹2,000–₹5,000","Above ₹5,000"],
                "option_type":  "radio",
                "handoff_to_gemini": True,
            })

    # ════════════════════════════════════════════════════════════════
    # PREVIOUS POLICY DOC (legacy — kept for backward compat)
    # ════════════════════════════════════════════════════════════════
    elif doc_type == "prev_policy":
        log.info(f"[PIPELINE] Previous policy OCR | user={user_id}")
        result = ocr.analyze_policy_doc(file_bytes, ext, user_id)

        return jsonify({
            "status": "success",
            "verified": result.get("success", True),
            "reply": result.get("message","✅ Previous policy analyzed 👍"),
            "policy_no": result.get("policy_no",""),
            "coverage": result.get("coverage",""),
            "has_claims": result.get("has_claims", False),
            "options":[], "option_type":"none", "next_stage":"",
            "handoff_to_gemini": result.get("handoff_to_gemini", True),
        })

    # ════════════════════════════════════════════════════════════════
    # ADMIN POLICY PDF — extract text for RAG
    # ════════════════════════════════════════════════════════════════
    elif doc_type == "policy_pdf":
        policy_text = ocr.extract_policy_text_for_rag(file_bytes, ext)
        if policy_text:
            rag.add_document(save_path, file.filename)
            return jsonify({
                "status":"success","verified":True,
                "reply":f"✅ Policy document indexed ({len(policy_text)} chars extracted).",
                "options":[],"option_type":"none","next_stage":""
            })
        else:
            return jsonify({
                "status":"success","verified":False,
                "reply":"⚠️ Could not extract text from policy document. Try a text-based PDF.",
                "options":[],"option_type":"none","next_stage":""
            })

    else:
        return jsonify({
            "status": "success", "verified": False,
            "reply": "✅ Document received.",
            "options": [], "option_type": "none", "next_stage": ""
        })

# ── Cleanup helper ─────────────────────────────────────────────────────────────
def _cleanup_user_docs(user_id: str):
    """Delete uploaded files from disk, mark DB records as deleted."""
    count = DocumentVerifier.delete_user_uploads(user_id, db)
    log.info(f"[CLEANUP] Deleted {count} document(s) for user {user_id}")
    return count

@app.route("/api/cleanup", methods=["POST"])
def cleanup():
    """Manual trigger for document cleanup (called on farewell)."""
    d = request.json or {}
    user_id = d.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    count = _cleanup_user_docs(user_id)
    return jsonify({"status": "success", "deleted_files": count,
                    "message": f"Deleted {count} uploaded document(s) securely."})

# ── Rating API ─────────────────────────────────────────────────────────────────
@app.route("/api/rating", methods=["POST"])
def rating():
    d     = request.json or {}
    score = int(d.get("rating", 0))
    uid   = d.get("user_id")
    if 1 <= score <= 5:
        db.store_rating(uid, score, d.get("comment",""))
    return jsonify({"status": "success", "message": "Rating saved. Thank you! 🎉"})

# ── Escalation API ─────────────────────────────────────────────────────────────
@app.route("/api/escalate", methods=["POST"])
def escalate():
    d = request.json or {}
    uid = d.get("user_id")
    phone = d.get("phone","")
    time_ = d.get("best_time","")
    plan  = d.get("plan_name","")
    db.store_escalation(uid, phone, time_, plan)
    lead_mgr.mark(uid, plan, "high", "escalated")
    return jsonify({"status": "success",
                    "reply": "✅ A human advisor will contact you soon! 📞"})

# ── Profile API ────────────────────────────────────────────────────────────────
@app.route("/api/profile")
def get_profile():
    uid = request.args.get("user_id")
    return jsonify({"status": "success", "profile": db.get_user_profile(uid) or {}})

# ── Lead API ───────────────────────────────────────────────────────────────────
@app.route("/api/lead", methods=["POST"])
def lead():
    d = request.json or {}
    lead_mgr.mark(d.get("user_id"), d.get("plan_name"),
                       d.get("interest_level","medium"), d.get("action","interested"))
    return jsonify({"status": "success"})

# ── Admin APIs ─────────────────────────────────────────────────────────────────
@app.route("/api/admin/analytics")
@admin_required
def api_analytics(): return jsonify({"status":"success","analytics":db.get_analytics()})

@app.route("/api/admin/users")
@admin_required
def api_users():
    q = request.args.get("q",""); limit = int(request.args.get("limit",50)); offset = int(request.args.get("offset",0))
    return jsonify({"status":"success","users":db.search_users(q,limit,offset),"total":db.count_users(q)})

@app.route("/api/admin/chats")
@admin_required
def api_chats():
    return jsonify({"status":"success","chats":db.search_chats(request.args.get("user_id"),request.args.get("q",""),int(request.args.get("limit",50)))})

@app.route("/api/admin/leads")
@admin_required
def api_leads(): return jsonify({"status":"success","leads":db.get_leads()})

@app.route("/api/admin/ratings")
@admin_required
def api_ratings(): return jsonify({"status":"success","ratings":db.get_ratings()})

@app.route("/api/admin/documents")
@admin_required
def api_docs(): return jsonify({"status":"success","documents":db.get_documents()})

@app.route("/api/admin/documents/<int:doc_id>/toggle", methods=["POST"])
@admin_required
def toggle_doc(doc_id): db.toggle_document(doc_id); return jsonify({"status":"success"})

@app.route("/api/admin/documents/<int:doc_id>", methods=["DELETE"])
@admin_required
def delete_doc(doc_id): db.delete_document(doc_id); return jsonify({"status":"success"})

@app.route("/api/admin/fraud-alerts")
@admin_required
def api_fraud():
    users  = db.get_all_users_raw()
    alerts = []
    for u in users:
        r = fraud.check(dict(u))
        if r["risk_level"] != "LOW":
            r["user_name"] = u["name"] if u["name"] else "Unknown"
            r["user_id"]   = u["user_id"]
            alerts.append(r)
    return jsonify({"status":"success","alerts":alerts})

@app.route("/api/admin/api-usage")
@admin_required
def api_key_usage(): return jsonify({"status":"success","usage":gemini.get_key_usage()})

@app.route("/api/admin/export/users")
@admin_required
def export_users():
    import csv, io
    users = db.get_all_users_raw()
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["user_id","name","age","gender","city","insurance_type","budget_range","family","medical","id_verified"])
    for u in users:
        u = dict(u)
        w.writerow([u.get("user_id",""), u.get("name",""), u.get("age",""), u.get("gender",""),
                    u.get("city",""), u.get("insurance_type",""), u.get("budget_range",""),
                    u.get("family_members",""), u.get("medical_conditions",""),
                    "Yes" if u.get("gov_id_verified") else "No"])
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=policybot_users.csv"})

# ── Policy Knowledge Base Admin APIs ──────────────────────────────────────────
@app.route("/api/admin/kb/documents", methods=["GET"])
@admin_required
def kb_list_docs():
    return jsonify({"status": "success", "documents": db.kb_get_all_docs()})

@app.route("/api/admin/kb/documents/<int:doc_id>", methods=["GET"])
@admin_required
def kb_get_doc(doc_id):
    doc   = db.kb_get_doc(doc_id)
    plans = db.kb_get_plans(doc_id)
    versions = db.kb_get_versions(doc_id)
    return jsonify({"status": "success", "document": doc, "plans": plans, "versions": versions})

@app.route("/api/admin/kb/upload", methods=["POST"])
@admin_required
def kb_upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    file_bytes = file.read()

    safe_name = f"kb_{uuid.uuid4()}{ext}"
    save_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "kb")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, safe_name)
    with open(save_path, "wb") as fout:
        fout.write(file_bytes)

    result = policy_kb.process_upload(file.filename, file_bytes, save_path, "admin")
    status_code = 200 if result["success"] else 422

    # Also add to RAG engine for semantic search
    if result["success"] and result.get("doc_id"):
        try:
            rag.add_document(save_path, file.filename)
        except Exception as e:
            log.warning(f"[KB] RAG add warning: {e}")

    return jsonify(result), status_code

@app.route("/api/admin/kb/documents/<int:doc_id>/update", methods=["POST"])
@admin_required
def kb_update_doc(doc_id):
    file = request.files.get("file")
    note = request.form.get("note", "Admin update")
    if not file or not file.filename:
        return jsonify({"error": "No file provided"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    file_bytes = file.read()

    safe_name = f"kb_{uuid.uuid4()}{ext}"
    save_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads", "kb")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, safe_name)
    with open(save_path, "wb") as fout:
        fout.write(file_bytes)

    result = policy_kb.process_update(doc_id, file.filename, file_bytes, save_path, note)
    return jsonify(result)

@app.route("/api/admin/kb/documents/<int:doc_id>/download", methods=["GET"])
@admin_required
def kb_download_doc(doc_id):
    """Download the original uploaded policy document."""
    import mimetypes
    from flask import send_file
    doc = db.kb_get_doc(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    file_path = doc.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not available on server"}), 404
    original_name = doc.get("filename", os.path.basename(file_path))
    mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return send_file(file_path, mimetype=mime,
                     as_attachment=True, download_name=original_name)

@app.route("/api/admin/kb/documents/<int:doc_id>/reextract", methods=["POST"])
@admin_required
def kb_reextract_doc(doc_id):
    """Force re-extraction of plans from an already-uploaded document."""
    doc = db.kb_get_doc(doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    file_path = doc.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        return jsonify({"error": "File not on server — please re-upload"}), 404
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        ext = os.path.splitext(file_path)[1].lower()
        # Re-run extraction pipeline
        text = policy_kb._extract_text(file_bytes, ext, file_path)
        if not text:
            return jsonify({"success": False, "message": "⚠️ Could not extract text even after re-try."}), 422
        plans = policy_kb._extract_plans_via_ai(text)
        if not plans:
            return jsonify({"success": False, "message": "⚠️ AI could not find plan data in document."}), 422
        db.kb_store_plans(doc_id, plans, is_master=0)
        db.kb_update_doc_status(doc_id, "active")
        return jsonify({
            "success": True,
            "message": f"✅ Re-extracted {len(plans)} plan(s) successfully.",
            "plans": plans,
        })
    except Exception as e:
        log.error(f"[KB] Re-extract error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/admin/kb/documents/<int:doc_id>", methods=["DELETE"])
@admin_required
def kb_delete_doc(doc_id):
    doc = db.kb_get_doc(doc_id)
    if doc and doc.get("file_path") and os.path.exists(doc["file_path"]):
        try:
            os.remove(doc["file_path"])
        except Exception:
            pass
    db.kb_delete_doc(doc_id)
    return jsonify({"status": "success", "message": "Document and associated plans deleted."})

@app.route("/api/admin/kb/plans/<int:plan_id>/toggle", methods=["POST"])
@admin_required
def kb_toggle_plan(plan_id):
    db.kb_toggle_plan(plan_id)
    return jsonify({"status": "success"})

@app.route("/api/admin/kb/plans/<int:plan_id>", methods=["DELETE"])
@admin_required
def kb_delete_plan(plan_id):
    db.kb_delete_plan(plan_id)
    return jsonify({"status": "success"})

@app.route("/api/admin/kb/analytics")
@admin_required
def kb_analytics():
    return jsonify({"status": "success", "analytics": db.kb_get_analytics()})

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "service": "PolicyBot v3",
        "version": "3.0.0",
        "keys_configured": len(gemini.keys),
        "key_statuses": [
            {"key": f"Key {s['key_index']}", "status": s['status'],
             "requests": s['requests'], "cooldown_remaining": s.get('cooldown_remaining', 0)}
            for s in gemini.get_key_usage()
        ]
    })

@app.route("/api/admin/gemini/health")
@admin_required
def gemini_health():
    """Detailed per-key health check — triggers actual API test calls."""
    results = gemini.health_check()
    usage   = gemini.get_key_usage()
    return jsonify({"status": "success", "health": results, "usage": usage})

@app.route("/api/admin/gemini/status")
@admin_required
def gemini_status():
    """Quick status without test calls — just shows cooldown state."""
    return jsonify({"status": "success", "usage": gemini.get_key_usage()})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)