"""
ConversationEngine v6 — Insurance-type-aware condition checks + document upload
NEW IN v6:
  - After collect_medical: branches based on insurance type AND conditions selected
  - Health Insurance + condition → condition_report_upload → condition_report_wait → collect_budget
  - Health Insurance + None → optional_health_check (skippable) → collect_budget
  - Vehicle Insurance → vehicle_history → (if doc needed) vehicle_doc_upload → collect_budget
  - Life/Term Insurance → life_docs → collect_budget
  - Travel Insurance → travel_declare → collect_budget
  - Property Insurance → property_history → collect_budget
  - All new upload stages are LOCKED (only /api/upload or skip exits)
  - After condition doc branch → returns to main recommendation module
"""
import json, re

SYSTEM_PROMPT = """You are PolicyBot, a warm AI Insurance Advisor for India.

PERSONALITY: Friendly, concise (2-3 sentences max), emoji-natural 😊 👍 ✨
Never use markdown headers or bullet symbols in chat messages.
Address user by first name when known.

LANGUAGE RULE (CRITICAL — NO EXCEPTIONS):
- Every prompt contains a LANGUAGE field (English / Tamil / Hindi).
- You MUST reply in that exact language ONLY.
- If user writes in Tamil but LANGUAGE=English → reply in English.
- If LANGUAGE=Tamil → reply entirely in Tamil script.
- If LANGUAGE=Hindi → reply entirely in Hindi script.
- NEVER mix languages in a single reply.

CRITICAL LANGUAGE RULE:
- The UI language is specified in every prompt as LANGUAGE field.
- ALWAYS reply in that exact language — English, Tamil, or Hindi.
- NEVER mix languages. If user writes in Tamil but LANGUAGE=English → reply in English.
- If LANGUAGE=Tamil → reply entirely in Tamil.
- If LANGUAGE=Hindi → reply entirely in Hindi.
- Language of user's message does NOT change your reply language. Only LANGUAGE field matters.

STRICT STEP ORDER — DO NOT BREAK UNDER ANY CIRCUMSTANCE:
1. Insurance type (radio shown)
2. NAME — ALWAYS ask "Nice choice 👍 May I know your name?" after insurance type
3. Age — ask "Thanks [name]! How old are you?"
4. Gov ID upload — tell user to use upload widget on the left sidebar
5. Verification wait — ONLY say waiting message
6. Gender (radio shown)
7. City
8. Family members (multi-select shown)
9. Medical conditions (multi-select shown)
   → If condition exists (Health): ask to upload health/condition report
   → If None (Health): optional general health check (skippable)
   → Vehicle: ask about previous policy / accident history
   → Life/Term: ask about medical history / income proof
   → Travel: ask about health or trip declarations
   → Property: ask about previous damage / claim history
10. Budget (radio shown)
11. Recommend 2-3 plans with full details
12. Explain selected plan
13. Ask about human advisor
14. Ask for 1-5 star rating
15. Farewell

ABSOLUTE RULES:
- After insurance_type → NEXT is ALWAYS name. NO EXCEPTIONS
- Never say document verified unless backend confirmed it
- Never recommend before budget is collected
- At verify_wait / condition_report_wait: ONLY say waiting message, ask NO questions
- Skippable stages: user types "skip" / "later" / "no thanks" to bypass
- At recommendation: ALWAYS show plans as radio button options (handled by backend)
- At explain_plan: end ALWAYS with "Would you like to apply now or compare other plans?"
- GOV ID VERIFICATION: BOTH name AND age must match for verified status
- RECOMMENDATION MEMORY: Plans are shown ONLY ONCE per session. If already shown, NEVER re-list them.
  If user says "none", "ok thanks", "no" at recommendation → ask: Apply now? Speak to advisor? Change budget?
  Do NOT re-list the plans.
"""

_NOT_A_NAME = {
    "health insurance","term life insurance","term / life insurance","vehicle insurance",
    "travel insurance","property insurance","accident insurance","health","term","life",
    "vehicle","travel","property","accident","insurance","yes","no","okay","ok",
    "skip","later","continue","hello","hi","hey","good","fine","sure","next",
    "none","male","female","other","myself","only me","family","spouse","children",
    "parents","full family","aadhaar","pan","passport","driving license","voter id",
    "diabetes","blood pressure","heart disease","asthma","cancer","upload","verify",
}

_SKIP_WORDS = ["skip","later","without","bypass","no id","continue without",
               "don't have","dont have","no thanks","not now","maybe later"]


class ConversationEngine:

    STEPS = [
        "insurance_type",          # 1
        "collect_name",            # 2
        "collect_age",             # 3
        "doc_upload",              # 4
        "verify_wait",             # 5  locked
        "collect_gender",          # 6
        "collect_city",            # 7
        "collect_family",          # 8
        "collect_medical",         # 9
        # ── Condition-specific branches (inserted after collect_medical) ──
        "condition_report_upload", # 9a health condition report
        "condition_report_wait",   # 9b locked
        "optional_health_check",   # 9c optional (skippable)
        "vehicle_history",         # 9d vehicle prev policy / accident
        "vehicle_doc_upload",      # 9e vehicle doc upload (skippable)
        "life_docs",               # 9f life/term medical or income doc
        "travel_declare",          # 9g travel health / trip declaration
        "property_history",        # 9h property damage / claim
        # ─────────────────────────────────────────────────────────────────
        "collect_budget",          # 10
        "recommendation",          # 11
        "explain_plan",            # 12
        "ask_escalation",          # 13
        "ask_rating",              # 14
        "farewell",                # 15
    ]

    PROGRESS = {
        "insurance_type":7,  "collect_name":12,    "collect_age":18,
        "doc_upload":24,     "verify_wait":29,      "collect_gender":35,
        "collect_city":41,   "collect_family":47,   "collect_medical":53,
        "condition_report_upload":57, "condition_report_wait":60,
        "optional_health_check":57,
        "vehicle_history":57, "vehicle_doc_upload":60,
        "life_docs":57,       "travel_declare":57,   "property_history":57,
        "collect_budget":65,  "recommendation":76,   "explain_plan":84,
        "ask_escalation":91,  "ask_rating":96,        "farewell":100,
    }
    LABELS = {
        "insurance_type":"Insurance Type",  "collect_name":"Your Name",
        "collect_age":"Your Age",           "doc_upload":"ID Upload",
        "verify_wait":"Verifying ID",       "collect_gender":"Gender",
        "collect_city":"Your City",         "collect_family":"Family",
        "collect_medical":"Medical",
        "condition_report_upload":"Health Report",
        "condition_report_wait":"Analyzing Report",
        "optional_health_check":"Health Check",
        "vehicle_history":"Vehicle History",
        "vehicle_doc_upload":"Vehicle Docs",
        "life_docs":"Life Documents",
        "travel_declare":"Travel Declare",
        "property_history":"Property History",
        "collect_budget":"Budget",          "recommendation":"Recommendations",
        "explain_plan":"Plan Details",      "ask_escalation":"Human Advisor",
        "ask_rating":"Rating",              "farewell":"Done ✅",
    }

    # Stages that are "upload-locked" — only /api/upload or skip exits them
    _UPLOAD_LOCKED = {"verify_wait", "condition_report_wait"}

    # Stages that are skippable upload stages
    _SKIPPABLE_UPLOAD = {"optional_health_check", "vehicle_doc_upload",
                         "life_docs", "travel_declare", "property_history"}

    def __init__(self, gemini, rag, db):
        self.gemini = gemini
        self.rag    = rag
        self.db     = db

    # ══════════════════════════════════════════════════════
    # MAIN
    # ══════════════════════════════════════════════════════
    def process(self, user_id, session_id, message, history,
                profile, language="English", fresh_session=False):

        if fresh_session:
            self.db.reset_session_profile(user_id)
            profile = {"onboarding_stage": "insurance_type", "user_id": user_id}

        stage = profile.get("onboarding_stage", "insurance_type")
        if stage not in self.STEPS:
            stage = "insurance_type"

        message = message.strip()
        extracted = self._extract(message, stage)
        next_stage = self._next(stage, message, extracted, profile, user_id)

        if extracted:
            profile.update(extracted)
            self.db.upsert_user_profile(user_id, extracted)

        if next_stage != stage:
            self.db.upsert_user_profile(user_id, {"onboarding_stage": next_stage})
            profile["onboarding_stage"] = next_stage

        if stage == "ask_rating":
            rv = self._extract_rating(message)
            if rv:
                self.db.store_rating(user_id, rv)

        if stage == "recommendation":
            ps = self._detect_plan(message)
            if ps:
                self.db.upsert_user_profile(user_id, {"selected_plan": ps})
                profile["selected_plan"] = ps

        rag_ctx = ""
        kb_recs  = []
        kb_reply = ""

        # ══════════════════════════════════════════════════════════════════════
        # RECOMMENDATION MEMORY — Plans shown ONCE per session only
        # ══════════════════════════════════════════════════════════════════════
        # Detect profile-change events that allow fresh recommendations
        _budget_changed  = (stage == "collect_budget" and extracted.get("budget_range"))
        _family_changed  = (stage == "collect_family" and extracted.get("family_members"))
        _profile_changed = _budget_changed or _family_changed

        if _profile_changed:
            # Profile changed → reset plans_shown so new recs can fire
            self.db.clear_plans_shown(user_id)
            profile["plans_shown"] = 0

        _plans_mem       = self.db.get_plans_shown(user_id)
        _shown_names     = _plans_mem["plan_names"]
        # Validate stored plan names match current insurance type — clear if stale
        _current_ins = (profile.get("insurance_type") or "").lower()
        _plans_stale = False
        if _shown_names and _current_ins:
            # Detect obvious type mismatch: vehicle plans shown for health insurance etc.
            _plan_text = " ".join(_shown_names).lower()
            _is_vehicle_plan = any(w in _plan_text for w in ["motor","vehicle","comprehensive motor","automobile"])
            _is_health_plan  = any(w in _plan_text for w in ["health","optima","family","bupa","care plan","diabetes"])
            _is_term_plan    = any(w in _plan_text for w in ["protect","term","jeevan","iprotect"])
            if "vehicle" in _current_ins and (_is_health_plan or _is_term_plan):
                _plans_stale = True
            elif "health" in _current_ins and (_is_vehicle_plan or _is_term_plan):
                _plans_stale = True
            elif ("term" in _current_ins or "life" in _current_ins) and (_is_vehicle_plan or _is_health_plan):
                _plans_stale = True
        if _plans_stale:
            self.db.clear_plans_shown(user_id)
            _plans_mem = {"shown": False, "plan_names": []}
            _shown_names = []
        _plans_already   = _plans_mem["shown"] and not _profile_changed and not _plans_stale

        # Pass plans_already flag to build_prompt via profile
        profile["_plans_already_shown"] = _plans_already

        if next_stage == "recommendation":
            rag_ctx = self.rag.get_context(self._rag_query(profile), self.gemini)

            if not _plans_already:
                # ── FIRST TIME: run KB and generate recommendations ────────────
                try:
                    from models.policy_kb import PolicyKB
                    _pkb = PolicyKB(self.db, self.gemini)
                    profile_with_uid = dict(profile)
                    profile_with_uid["user_id"] = user_id
                    kb_recs = _pkb.get_recommendations(profile_with_uid, top_n=3)
                    if kb_recs:
                        kb_reply = _pkb.format_recommendation_text(kb_recs, profile_with_uid)
                        plan_names = [item["plan"].get("plan_name","") for item in kb_recs
                                      if item["plan"].get("plan_name")]
                        profile["_kb_plan_options"] = plan_names
                        # Mark plans as shown in DB
                        self.db.mark_plans_shown(user_id, plan_names)
                        # Store recommendations in DB
                        for item in kb_recs:
                            p = item["plan"]
                            self.db.store_recommendation(user_id, {
                                "name":          p.get("plan_name",""),
                                "premium":       p.get("premium_range",""),
                                "coverage":      p.get("coverage_amount",""),
                                "waiting_period": p.get("waiting_period",""),
                                "reason":        item.get("reason",""),
                            })
                except Exception as _kbe:
                    import logging
                    logging.getLogger("PolicyBot").warning(f"[KB] Recommendation error: {_kbe}")
            else:
                # ── ALREADY SHOWN: set plan names for options but NO new rec reply ──
                profile["_kb_plan_options"] = _shown_names

        elif next_stage == "explain_plan":
            rag_ctx = self.rag.get_context(self._rag_query(profile), self.gemini)

        prompt   = self._build_prompt(message, history, profile, rag_ctx, next_stage, language)
        ai_reply = self.gemini.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=500)

        # ── Fallback reply if AI fails or returns empty ──────────────────────────
        # Treat Gemini error strings same as empty (prevents error text showing as bot reply)
        _gemini_errors = [
            "sorry, our server is busy",
            "i'm having trouble connecting",
            "i encountered an issue",
            "please check your api keys",
        ]
        _is_ai_error = (not ai_reply or not ai_reply.strip() or
                        any(err in (ai_reply or "").lower() for err in _gemini_errors))
        if _is_ai_error:
            _name = profile.get("name", "")
            _ins  = (profile.get("insurance_type") or "insurance")
            _cond = profile.get("medical_conditions", "")
            _ins_l = _ins.lower()
            _fallback_map = {
                "collect_medical":   (
                    f"Do you or any family members have any pre-existing injuries, "
                    "disabilities or occupational hazards? ⚡ (Select all that apply)"
                    if "accident" in (_ins or "").lower() else
                    f"Do you or any family members have any health history or lifestyle habits to declare? 📋 "
                    "(Select all that apply)"
                    if any(w in (_ins or "").lower() for w in ["term","life"]) else
                    f"Any medical conditions that may affect your travel coverage? ✈️ "
                    "(Select all that apply, or choose None)"
                    if "travel" in (_ins or "").lower() else
                    f"Do you or any family members have any medical conditions? 🏥 (Select all that apply)"
                ),
                "collect_budget":    f"Almost there{', ' + _name if _name else ''}! 🎯 What is your monthly budget for insurance premiums? 💰",
                "collect_family":    f"Who would you like to include in your insurance coverage{', ' + _name if _name else ''}? 👨‍👩‍👧‍👦",
                "collect_city":      f"Which city do you live in{', ' + _name if _name else ''}? 🏙️",
                "collect_gender":    f"What is your gender{', ' + _name if _name else ''}? 😊",
                "ask_escalation":    f"Would you like to speak with a human advisor, {_name or 'there'}? 😊",
                "ask_rating":        f"Thanks for chatting with me{', ' + _name if _name else ''}! 😊 Could you rate our conversation from 1 to 5 stars? ✨",
                "farewell":          f"🎉 Thank you {_name or ''}! Your insurance journey is complete. Wishing you a safe and secure future!",
            }
            ai_reply = _fallback_map.get(next_stage, "")

        # ── Build final reply ──────────────────────────────────────────────────
        if next_stage == "recommendation":
            if kb_reply and not _plans_already:
                # Fresh recommendation — use KB reply
                reply = kb_reply
            elif _plans_already:
                # Plans already shown — AI handles "none/ok thanks/repeat" gracefully
                reply = ai_reply or ""
            else:
                reply = ai_reply or ""

        elif next_stage == "explain_plan":
            reply = ai_reply or ""
            if reply and "apply now" not in reply.lower() and "compare other plans" not in reply.lower():
                reply += "\n\nWould you like to apply now or compare other plans?"
        else:
            reply = ai_reply
        options, opt_type = self._options(next_stage, profile)

        # Show upload card only for stages where user needs to upload a file
        show_upload = next_stage in (
            "doc_upload", "verify_wait",
            "condition_report_upload", "condition_report_wait",
            "optional_health_check",
            "vehicle_doc_upload",   # vehicle_history itself is radio-only (no upload needed)
            "life_docs",
        )
        # travel_declare and property_history are radio-only (no upload widget needed)
        # vehicle_history is radio-only — upload comes AFTER if needed (vehicle_doc_upload)

        _is_farewell    = next_stage == "farewell"
        _lock_chat      = _is_farewell  # frontend should lock all input

        return {
            "reply":              reply,
            "stage":              next_stage,
            "options":            [] if _is_farewell else options,
            "option_type":        "none" if _is_farewell else opt_type,
            "profile_updated":    bool(extracted),
            "show_rating":        next_stage == "ask_rating",
            "show_escalation":    next_stage == "ask_escalation",
            "show_farewell":      _is_farewell,
            "show_upload":        show_upload,
            "confidence":         self._confidence(profile, next_stage),
            "progress":           self.PROGRESS.get(next_stage, 7),
            "stage_label":        self.LABELS.get(next_stage, ""),
            "module":             self._module(next_stage),
            "plans_mentioned":    [],
            "trigger_cleanup":    _is_farewell,
            "lock_chat":          _lock_chat,
        }

    # ══════════════════════════════════════════════════════
    # STAGE TRANSITIONS
    # ══════════════════════════════════════════════════════
    # Words that mean "I want to end this session now"
    _WIND_UP_WORDS = [
        "wind up","wind-up","windup","wrap up","wrap-up","wrapup",
        "end session","close session","end chat","close chat",
        "finish","done talking","bye","goodbye","cya","see ya",
        "thank you bye","thanks bye","thats all","that's all","all done",
        "exit","quit","im done","i'm done","no more","i'm good","im good",
        "good bye","goodbye","not interested","no thanks",
        "maybe later","not now","wind",
    ]

    def _next(self, stage, message, extracted, profile, user_id=""):
        msg = message.lower().strip()
        ins = (profile.get("insurance_type") or "").lower()
        is_skip = any(w in msg for w in _SKIP_WORDS)

        # ── Global wind-up detection — works from ANY stage ────────────────
        # Skip early stages where "wind" could be a name or city
        _safe_wind_stages = {"collect_budget","recommendation","explain_plan",
                             "ask_escalation","vehicle_history","vehicle_doc_upload",
                             "life_docs","travel_declare","property_history",
                             "optional_health_check","condition_report_upload"}
        if stage in _safe_wind_stages:
            if any(w in msg for w in self._WIND_UP_WORDS):
                return "ask_rating"

        # ── Linear stages 1-7 (unchanged) ─────────────────────────────────
        if stage == "insurance_type":
            if extracted.get("insurance_type"):
                # Clear any stale plan cache from previous session
                self.db.clear_plans_shown(user_id)
                return "collect_name"
            return "insurance_type"

        if stage == "collect_name":
            return "collect_age" if extracted.get("name") else "collect_name"

        if stage == "collect_age":
            return "doc_upload" if extracted.get("age") else "collect_age"

        if stage == "doc_upload":
            if is_skip:
                self.db.upsert_user_profile(user_id, {"gov_id_verified": 0})
                return "collect_gender"
            return "doc_upload"

        if stage == "verify_wait":
            return "verify_wait"  # only /api/upload exits

        if stage == "collect_gender":
            # After ID verification: user may confirm details first before giving gender
            if "yes" in msg and ("correct" in msg or "detail" in msg):
                # Details confirmed — now ask gender
                # Don't advance yet, let next message carry gender
                return "collect_gender"
            if "no" in msg and ("update" in msg or "wrong" in msg or "incorrect" in msg):
                # User wants to update details — go back to name collection
                self.db.upsert_user_profile(user_id, {"name": None, "age": None})
                return "collect_name"
            return "collect_city" if extracted.get("gender") else "collect_gender"

        if stage == "collect_city":
            return "collect_family" if extracted.get("city") else "collect_city"

        if stage == "collect_family":
            if not extracted.get("family_members"):
                return "collect_family"
            # ── Skip collect_medical for types that don't need it ──────────────
            _ins_type = (profile.get("insurance_type") or "").lower()
            if "vehicle" in _ins_type:
                return "vehicle_history"      # → vehicle branch directly
            if "property" in _ins_type:
                return "property_history"     # → property branch directly
            # All others (health, life/term, travel, accident) need collect_medical
            return "collect_medical"

        # ── Step 9: collect_medical → branch by insurance type + conditions ──
        if stage == "collect_medical":
            if not extracted.get("medical_conditions"):
                return "collect_medical"

            cond = extracted.get("medical_conditions", "").lower()
            has_condition = "none" not in cond

            # Health Insurance
            if "health" in ins:
                if has_condition:
                    # Store that a condition exists and doc upload is needed
                    self.db.upsert_user_profile(user_id, {"condition_report_uploaded": 0})
                    return "condition_report_upload"
                else:
                    return "optional_health_check"

            # Vehicle Insurance
            if "vehicle" in ins:
                return "vehicle_history"

            # Life / Term Insurance
            if "life" in ins or "term" in ins:
                return "life_docs"

            # Travel Insurance
            if "travel" in ins:
                return "travel_declare"

            # Property Insurance → property_history
            if "property" in ins:
                return "property_history"

            # Accident Insurance → conditions captured, go to budget
            if "accident" in ins:
                return "collect_budget"

            # Default (unknown type) — go straight to budget
            return "collect_budget"

        # ── Condition report upload (locked — /api/upload exits to collect_budget) ──
        if stage == "condition_report_upload":
            # Stays here until /api/upload sets stage to collect_budget
            # Skip allowed
            if is_skip:
                self.db.upsert_user_profile(user_id,
                    {"condition_report_uploaded": 0, "onboarding_stage": "collect_budget"})
                return "collect_budget"
            return "condition_report_upload"

        if stage == "condition_report_wait":
            return "condition_report_wait"  # only /api/upload exits

        # ── Optional health check (fully skippable) ─────────────────────────
        if stage == "optional_health_check":
            if is_skip or "no" in msg or "none" in msg:
                return "collect_budget"
            # Upload triggers /api/upload → advances to collect_budget
            return "optional_health_check"

        # ── Vehicle history ─────────────────────────────────────────────────
        if stage == "vehicle_history":
            hist = extracted.get("vehicle_history", "")
            if hist:
                if "none" in hist.lower():
                    return "collect_budget"
                # Has previous policy or accident → ask for doc upload
                return "vehicle_doc_upload"
            return "vehicle_history"

        if stage == "vehicle_doc_upload":
            if is_skip:
                return "collect_budget"
            return "vehicle_doc_upload"

        # ── Life docs ───────────────────────────────────────────────────────
        if stage == "life_docs":
            if is_skip or "none" in msg:
                return "collect_budget"
            return "life_docs"

        # ── Travel declare ──────────────────────────────────────────────────
        if stage == "travel_declare":
            if is_skip or "none" in msg:
                return "collect_budget"
            return "travel_declare"

        # ── Property history ────────────────────────────────────────────────
        if stage == "property_history":
            if is_skip or "none" in msg:
                return "collect_budget"
            return "property_history"

        # ── Steps 10-15 (unchanged) ─────────────────────────────────────────
        if stage == "collect_budget":
            return "recommendation" if extracted.get("budget_range") else "collect_budget"

        if stage == "recommendation":
            select_words = ["select","choose","this one","go with","i want","apply",
                            "plan 1","plan 2","plan 3","option 1","first","second","1st","2nd"]
            # User selected a plan → explain it
            if self._detect_plan(message) or any(w in msg for w in select_words):
                return "explain_plan"
            # User wants a human advisor
            if any(w in msg for w in ["advisor","human","agent","call me","speak to"]):
                return "ask_escalation"
            # User wants to END the session
            _exit_words = ["wind up","wind-up","windup","wind","wrap up","wrap-up","wrapup","wrap",
                           "end session","close session","finish","done","bye","goodbye","cya",
                           "no thank","not interested","exit","close","end","quit",
                           "thank you","thanks bye","thats all","that's all","all done",
                           "nothing","nothin","no more","im good","i'm good","good bye",
                           "no plan","not now","maybe later","later","not needed","not interested"]
            if any(w in msg for w in _exit_words):
                return "ask_rating"   # Skip to rating → farewell
            return "recommendation"

        if stage == "explain_plan":
            _exit_words2 = ["wind up","wind-up","windup","wind","wrap up","wrap","done","bye","goodbye",
                             "no thanks","finish","end","quit","exit","nothin","nothing","no more",
                             "thank you","thanks","all done","thats all","i'm good","im good"]
            if any(w in msg for w in _exit_words2):
                return "ask_rating"
            return "ask_escalation"

        if stage == "ask_escalation":
            # Always advance to rating after escalation decision
            return "ask_rating"

        if stage == "ask_rating":
            # Advance to farewell if: numeric rating given OR any acknowledgment
            _ack_words = ["ok","okay","okk","thanks","thank","fine","done","bye","great","good","no","skip","later"]
            if self._extract_rating(message) or any(w in msg for w in _ack_words):
                return "farewell"
            return "ask_rating"

        if stage == "farewell":
            return "farewell"

        return stage

    # ══════════════════════════════════════════════════════
    # STAGE-GATED EXTRACTION
    # ══════════════════════════════════════════════════════
    def _extract(self, message, stage):
        msg = message.lower().strip()
        out = {}

        if stage == "insurance_type":
            ins_map = {
                "health":"Health Insurance", "term":"Term / Life Insurance",
                "life":"Term / Life Insurance", "vehicle":"Vehicle Insurance",
                "car":"Vehicle Insurance", "motor":"Vehicle Insurance",
                "bike":"Vehicle Insurance", "travel":"Travel Insurance",
                "property":"Property Insurance", "home":"Property Insurance",
                "accident":"Accident Insurance",
            }
            for k, v in ins_map.items():
                if k in msg:
                    out["insurance_type"] = v
                    break

        elif stage == "collect_name":
            text = message.strip()
            if text.lower() not in _NOT_A_NAME:
                words = text.split()
                if (1 <= len(words) <= 4
                        and text[0].isalpha()
                        and not any(c.isdigit() for c in text)
                        and text.lower() not in _NOT_A_NAME
                        and not any(kw in text.lower() for kw in
                                    ["insurance","health","vehicle","travel","property","accident","term"])):
                    out["name"] = text.title()

        elif stage == "collect_age":
            nums = re.findall(r'\b(\d{1,3})\b', message)
            for n in nums:
                v = int(n)
                if 1 <= v <= 120:
                    out["age"] = v
                    break

        elif stage == "collect_gender":
            if message.strip() in ["Male", "Female", "Other"]:
                out["gender"] = message.strip()
            elif any(w in msg for w in ["male","man","boy"]):
                out["gender"] = "Male"
            elif any(w in msg for w in ["female","woman","girl"]):
                out["gender"] = "Female"
            elif "other" in msg:
                out["gender"] = "Other"

        elif stage == "collect_city":
            text = message.strip()
            if text and len(text) >= 2 and not text.isdigit() and text.lower() not in _NOT_A_NAME:
                out["city"] = text.title()

        elif stage == "collect_family":
            fam_map = {
                "only me":"Only Me","just me":"Only Me","myself":"Only Me",
                "spouse":"Spouse","wife":"Spouse","husband":"Spouse",
                "child":"Children","children":"Children","kids":"Children",
                "parent":"Parents","mother":"Parents","father":"Parents",
                "family":"Full Family","everyone":"Full Family","full family":"Full Family",
            }
            found = []
            for k, v in fam_map.items():
                if k in msg and v not in found:
                    found.append(v)
            if found:
                out["family_members"] = ", ".join(found)
            elif message.strip() and message.strip().lower() not in _NOT_A_NAME:
                out["family_members"] = message.strip()

        elif stage == "collect_medical":
            # Comprehensive map covering all insurance types' conditions
            med_map = {
                # Health conditions
                "diabetes":"Diabetes","sugar":"Diabetes",
                "blood pressure":"Blood Pressure","hypertension":"Blood Pressure",
                "bp ":"Blood Pressure","high bp":"Blood Pressure",
                "heart":"Heart Disease","cardiac":"Heart Disease",
                "asthma":"Asthma","kidney":"Kidney Disease","renal":"Kidney Disease",
                "cancer":"Cancer","thyroid":"Thyroid",
                # Term/Life conditions
                "smoking":"Smoking / Tobacco","tobacco":"Smoking / Tobacco","smoke":"Smoking / Tobacco",
                "alcohol":"Alcohol Use","drink":"Alcohol Use",
                "cancer history":"Cancer History",
                # Accident conditions
                "injury":"Previous Injury","injured":"Previous Injury",
                "disability":"Permanent Disability","disabled":"Permanent Disability",
                "fracture":"Fracture History","broken":"Fracture History",
                "sport":"Sports Injury",
                "occupational":"Occupational Hazard","hazard":"Occupational Hazard",
                # Travel conditions
                "pregnant":"Pregnancy","pregnancy":"Pregnancy",
                "surgery":"Recent Surgery","operated":"Recent Surgery",
                "respiratory":"Respiratory Issues","breathing":"Respiratory Issues",
                "heart condition":"Heart Condition",
                # Generic
                "other":"Other",
                "none":"None","no ":"None","healthy":"None","nothing":"None","fit":"None","nope":"None",
            }
            found = []
            for k, v in med_map.items():
                if k in msg and v not in found:
                    found.append(v)
            if found:
                out["medical_conditions"] = ", ".join(found)
            elif message.strip() and message.strip().lower() not in _NOT_A_NAME:
                out["medical_conditions"] = message.strip()

        elif stage == "vehicle_history":
            vh_map = {
                "previous policy":"Previous Policy",
                "prev policy":"Previous Policy",
                "accident":"Accident Claim",
                "claim":"Accident Claim",
                "none":"None",
                "no ":"None","nothing":"None",
            }
            found = []
            for k, v in vh_map.items():
                if k in msg and v not in found:
                    found.append(v)
            if found:
                out["vehicle_history"] = ", ".join(found)
            elif message.strip() and message.strip().lower() not in _NOT_A_NAME:
                out["vehicle_history"] = message.strip()

        elif stage == "life_docs":
            ld_map = {
                "medical":"Medical History",
                "income":"Income Proof",
                "salary":"Income Proof",
                "none":"None","no ":"None",
            }
            found = []
            for k, v in ld_map.items():
                if k in msg and v not in found:
                    found.append(v)
            if found:
                out["life_docs"] = ", ".join(found)
            elif message.strip() and message.strip().lower() not in _NOT_A_NAME:
                out["life_docs"] = message.strip()

        elif stage == "travel_declare":
            td_map = {
                "medical":"Medical Condition",
                "trip":"Trip Delay History",
                "delay":"Trip Delay History",
                "none":"None","no ":"None",
            }
            found = []
            for k, v in td_map.items():
                if k in msg and v not in found:
                    found.append(v)
            if found:
                out["travel_declare"] = ", ".join(found)
            elif message.strip() and message.strip().lower() not in _NOT_A_NAME:
                out["travel_declare"] = message.strip()

        elif stage == "property_history":
            ph_map = {
                "fire":"Fire Damage",
                "flood":"Flood Damage",
                "claim":"Previous Claim",
                "damage":"Previous Claim",
                "none":"None","no ":"None",
            }
            found = []
            for k, v in ph_map.items():
                if k in msg and v not in found:
                    found.append(v)
            if found:
                out["property_history"] = ", ".join(found)
            elif message.strip() and message.strip().lower() not in _NOT_A_NAME:
                out["property_history"] = message.strip()

        elif stage == "collect_budget":
            if message.strip():
                out["budget_range"] = message.strip()

        return out

    # ══════════════════════════════════════════════════════
    # OPTIONS — insurance-type-aware for condition branches
    # ══════════════════════════════════════════════════════
    def _options(self, stage, profile=None):
        ins = ((profile or {}).get("insurance_type") or "").lower()

        if stage == "insurance_type":
            return (["Health Insurance","Term / Life Insurance","Vehicle Insurance",
                     "Travel Insurance","Property Insurance","Accident Insurance"], "radio")

        if stage == "collect_gender":
            return (["Male","Female","Other"], "radio")

        if stage == "collect_family":
            return (["Only Me","Spouse","Children","Parents","Full Family"], "multi")

        if stage == "collect_medical":
            # ── Per-insurance-type condition options ────────────────────────
            if "vehicle" in ins:
                return ([], "none")           # vehicle → vehicle_history
            if "property" in ins:
                return ([], "none")           # property → property_history
            if "accident" in ins:
                # Accident: ask about injuries, disabilities, hazards
                return (["Previous Injury","Permanent Disability","Fracture History",
                         "Sports Injury","Occupational Hazard","None"], "multi")
            if "term" in ins or "life" in ins:
                # Term/Life: lifestyle and critical illness history
                return (["Smoking / Tobacco","Alcohol Use","Diabetes",
                         "Heart Disease","High Blood Pressure","Cancer History","None"], "multi")
            if "travel" in ins:
                # Travel: conditions affecting travel coverage
                return (["Diabetes","Heart Condition","Respiratory Issues",
                         "Pregnancy","Recent Surgery","None"], "multi")
            # Health Insurance (default)
            return (["Diabetes","Blood Pressure","Heart Disease",
                     "Asthma","Kidney Disease","Cancer","Thyroid","None"], "multi")

        if stage == "vehicle_history":
            return (["Previous Policy","Accident Claim","None"], "radio")

        if stage == "life_docs":
            return (["Medical History","Income Proof","None"], "radio")

        if stage == "travel_declare":
            return (["Medical Condition","Trip Delay History","None"], "radio")

        if stage == "property_history":
            return (["Fire Damage","Flood Damage","Previous Claim","None"], "radio")

        if stage == "collect_budget":
            return (["Under ₹500","₹500–₹1,000","₹1,000–₹2,000",
                     "₹2,000–₹5,000","Above ₹5,000"], "radio")

        if stage == "recommendation":
            # Build plan radio options from KB recommendations cache
            # The actual recommendation text is built by policy_kb, options are plan names
            plans = (profile or {}).get("_kb_plan_options", [])
            if not plans:
                # Fallback plans — type AND condition specific
                # Normalize: "Smoking / Tobacco, Alcohol Use" → {"smoking","tobacco","alcohol use",...}
                _fb_ins   = ((profile or {}).get("insurance_type") or "").lower()
                _raw_cond = ((profile or {}).get("medical_conditions") or "").lower()
                # Split by comma, strip, flatten slash-separated terms into a set
                _cond_set = set()
                for _part in _raw_cond.split(","):
                    for _sub in _part.split("/"):
                        _cond_set.add(_sub.strip())

                def _has(*keywords):
                    """Check if any keyword appears in any condition token."""
                    return any(kw in token for kw in keywords for token in _cond_set)

                if "health" in _fb_ins:
                    if _has("diabetes"):
                        plans = ["Star Diabetes Safe","Niva Bupa ReAssure 360°","Care Freedom Plan"]
                    elif _has("heart","blood pressure","hypertension","cardiac"):
                        plans = ["Aditya Birla Activ Health Enhanced","Care Heart Plan","Star Cardiac Care"]
                    elif _has("cancer","critical"):
                        plans = ["HDFC Ergo Critical Illness","ICICI Pru Heart & Cancer","Bajaj Allianz CritiCare"]
                    elif _has("kidney","renal"):
                        plans = ["Star Health Medi Classic","Niva Bupa Senior First","Care Advantage Plan"]
                    elif _has("asthma","respiratory","lung"):
                        plans = ["HDFC Ergo Optima Restore","Niva Bupa Health Companion","Star Comprehensive"]
                    elif _has("thyroid"):
                        plans = ["Star Health Comprehensive","Care Classic Plan","HDFC Ergo My Health Suraksha"]
                    else:
                        plans = ["Star Health Family Optima","HDFC Ergo Optima Restore","Niva Bupa Health Companion"]
                elif "life" in _fb_ins or "term" in _fb_ins:
                    if _has("smoking","tobacco"):
                        plans = ["HDFC Click2Protect Plus","ICICI Pru iProtect Smart","Max Life Smart Secure"]
                    elif _has("diabetes","heart","cancer"):
                        plans = ["HDFC Click2Protect Life","Tata AIA Sampoorna Raksha","Bajaj Allianz Smart Protect"]
                    else:
                        plans = ["HDFC Click2Protect Life","LIC Tech Term","Tata AIA Sampoorna Raksha"]
                elif "vehicle" in _fb_ins:
                    _vh = ((profile or {}).get("vehicle_history") or "").lower()
                    if "accident" in _vh or "claim" in _vh:
                        plans = ["HDFC ERGO Motor Optima (Accident Cover)","Bajaj Allianz OD Cover","New India Motor Floater"]
                    elif "previous policy" in _vh or "prev policy" in _vh:
                        plans = ["ICICI Lombard Complete Cover","HDFC ERGO Motor Optima","Bajaj Allianz Comprehensive Motor"]
                    else:
                        plans = ["Bajaj Allianz Comprehensive Motor","ICICI Lombard Complete Cover","HDFC ERGO Motor Optima"]
                elif "travel" in _fb_ins:
                    if _has("diabetes","heart","pregnancy","surgery"):
                        plans = ["Bajaj Allianz Travel Care","HDFC ERGO Travel Protect","Tata AIG Overseas Care"]
                    else:
                        plans = ["Bajaj Allianz Travel Companion","HDFC ERGO Travel Protect","Tata AIG Travel Guard"]
                elif "property" in _fb_ins:
                    plans = ["New India Home Insurance","HDFC ERGO Home Protect","Bajaj Allianz Property Guard"]
                elif "accident" in _fb_ins:
                    if _has("injury","disability","fracture"):
                        plans = ["New India Janata Personal Accident","Bajaj Allianz Personal Guard","Star Accident Care"]
                    else:
                        plans = ["New India Personal Accident","Bajaj Allianz Personal Guard","Tata AIG Accident Cover"]
                else:
                    plans = ["Star Health Family Optima","HDFC Click2Protect Life","Bajaj Allianz Comprehensive Motor"]
            return (plans, "radio")

        if stage == "explain_plan":
            return (["Apply Now 🚀","Compare Other Plans 🔄","Talk to Advisor 👨‍💼"], "radio")

        if stage == "ask_escalation":
            return (["Talk to Human Advisor 👨‍💼","Continue with PolicyBot 🤖"], "radio")

        return [], "none"

    # ══════════════════════════════════════════════════════
    # PROMPT BUILDER
    # ══════════════════════════════════════════════════════
    def _build_prompt(self, message, history, profile, rag_ctx, stage, language):
        name = profile.get("name","")
        ins  = profile.get("insurance_type","insurance")
        cond = profile.get("medical_conditions","")

        safe_p = {k:v for k,v in {
            "name":             profile.get("name",""),
            "insurance_type":   profile.get("insurance_type",""),
            "age":              profile.get("age",""),
            "gender":           profile.get("gender",""),
            "city":             profile.get("city",""),
            "family_members":   profile.get("family_members",""),
            "medical_conditions":profile.get("medical_conditions",""),
            "budget_range":     profile.get("budget_range",""),
            "gov_id_verified":  profile.get("gov_id_verified",0),
            "selected_plan":    profile.get("selected_plan",""),
            "condition_report_uploaded": profile.get("condition_report_uploaded",0),
        }.items() if v not in ("",None,0)}

        recent = "\n".join(
            f"User: {h.get('message','')}\nBot: {h.get('bot_reply','')}"
            for h in history[-5:]
        )

        instructions = {
            # ── Original steps 1-10 ──────────────────────────────────────────
            "insurance_type":(
                "STEP 1: Welcome warmly. Ask what insurance they need. "
                "Say: 'Hi 😊 I'm PolicyBot, your AI insurance advisor! "
                "What type of insurance are you looking for today?' "
                "Insurance type radio buttons will appear automatically."
            ),
            "collect_name":(
                f"STEP 2: User chose '{ins}'. "
                "NOW ask their name. SAY EXACTLY: "
                "'Nice choice 👍 May I know your name?' "
                "Do NOT ask anything else. JUST ask name."
            ),
            "collect_age":(
                f"STEP 3: User's name is '{name}'. "
                f"Ask age. Say: 'Thanks {name or 'there'}! How old are you? 😊'"
            ),
            "doc_upload":(
                "STEP 4: Ask user to upload Government ID using the upload widget on the left. "
                "Accepted: Aadhaar, PAN, Driving License, Passport, Voter ID. "
                "Say they can type 'skip' if they prefer not to verify."
            ),
            "verify_wait":(
                "STEP 5: User uploaded document. "
                "Say ONLY: 'Thanks for uploading 😊 I'm checking your document now, "
                "please wait a moment.' Ask NO questions."
            ),
            "collect_gender":(
                f"STEP 6: ID is verified. "
                f"If user just said 'yes details are correct', say: "
                f"'Perfect! Your details are confirmed ✅ What is your gender{', ' + name if name else ''}? 😊' "
                f"Otherwise say: 'Great! ✅ Your ID is verified! "
                f"What is your gender{', ' + name if name else ''}? 😊' "
                "Gender radio buttons appear automatically."
            ),
            "collect_city":(
                f"STEP 7: Ask city. "
                f"Say: 'Which city do you live in{', ' + name if name else ''}? 🏙️'"
            ),
            "collect_family":(
                "STEP 8: Ask family members to cover. "
                "Say: 'Who would you like to include in your insurance coverage? 👨‍👩‍👧‍👦' "
                "Multi-select buttons appear automatically."
            ),
            "collect_medical":(
                (
                    "STEP 9: Ask about pre-existing injuries or disabilities for Accident Insurance. "
                    f"Say: 'Do you or any family members have any pre-existing injuries, "
                    "disabilities, or occupational hazards? ⚡ (Select all that apply)' "
                    "Buttons appear automatically."
                ) if "accident" in ins.lower() else (
                    "STEP 9: Ask about lifestyle risks and critical illness for Term/Life Insurance. "
                    f"Say: 'Do you or any family members have any health history or lifestyle habits to declare? 📋 "
                    "(This helps us find better term rates — select all that apply)' "
                    "Buttons appear automatically."
                ) if ("term" in ins.lower() or "life" in ins.lower()) else (
                    "STEP 9: Ask about travel health conditions for Travel Insurance. "
                    f"Say: 'Any medical conditions that may affect your travel coverage? ✈️ "
                    "(Select all that apply, or choose None to continue)' "
                    "Buttons appear automatically."
                ) if "travel" in ins.lower() else (
                    f"STEP 9: Ask medical conditions sensitively for {ins}. "
                    "Say: 'Do you or any family members have any medical conditions? 🏥 "
                    "(Select all that apply)' "
                    "Buttons appear automatically."
                )
            ),
            # ── NEW: Condition-based branches ────────────────────────────────
            "condition_report_upload":(
                f"STEP 9a (Health condition found: {cond}): "
                "Ask user to upload their health/medical report using the upload widget. "
                f"Say: 'Since you have {cond}, please upload your latest health report or "
                "prescription so I can recommend the best plan for your condition 🏥 "
                "You can type \"skip\" if you don't have it handy.' "
                "Upload widget is already visible on the left."
            ),
            "condition_report_wait":(
                "STEP 9b: User uploaded health report. "
                "Say ONLY: 'Thanks! 😊 I'm analyzing your health report now, please wait.' "
                "Ask NO questions."
            ),
            "optional_health_check":(
                f"STEP 9c: User has no conditions. "
                "Ask OPTIONALLY if they want to upload a general health check report. "
                "Say: 'Great, no conditions! 😊 Do you have a recent health check report? "
                "Uploading it can help get better plan rates. "
                "You can also skip this step.' "
                "Upload widget is visible on the left."
            ),
            "vehicle_history":(
                f"STEP 9d: Vehicle insurance for {name or 'user'}. "
                f"We have: name='{name}', city='{profile.get('city','')}', family='{profile.get('family_members','')}'. "
                "Now ask about vehicle history. "
                f"Say: 'Great {name or ''}! Do you have any previous vehicle insurance policy or accident claim history? 🚗' "
                "Radio buttons will appear (Previous Policy / Accident Claim / None)."
            ),
            "vehicle_doc_upload":(
                f"STEP 9e: User has vehicle history. "
                "Ask to upload previous policy or accident claim document. "
                "Say: 'Please upload your previous policy or accident claim document using "
                "the upload widget on the left 📄 You can type \"skip\" if unavailable.' "
            ),
            "life_docs":(
                f"STEP 9f: Life/Term insurance. "
                "Ask: 'Do you have any medical history or income documents to share? 📋' "
                "Say these help in getting better term rates. "
                "Upload widget is visible. They can skip."
            ),
            "travel_declare":(
                f"STEP 9g: Travel insurance. "
                "Ask: 'Any health conditions or trip-related issues to declare? ✈️' "
                "Radio buttons appear. They can select None to skip."
            ),
            "property_history":(
                f"STEP 9h: Property insurance for {name or 'user'}. "
                f"We have: name='{name}', city='{profile.get('city','')}'. "
                "Now ask about property history. "
                f"Say: 'Almost there {name or ''}! Any previous damage or insurance claim history for your property? 🏠' "
                "Radio buttons appear (Fire Damage / Flood Damage / Previous Claim / None)."
            ),
            # ── Steps 10-15 (unchanged) ───────────────────────────────────────
            "collect_budget":(
                "STEP 10: Ask monthly budget. "
                "Say: 'Almost there! 🎯 What is your monthly budget for insurance premiums? 💰' "
                "Budget radio buttons appear automatically."
            ),
            "recommendation":(
                (
                    # Plans already shown — do NOT repeat the list
                    "STEP 11: Plans were ALREADY shown to this user. "
                    "DO NOT list plans again. "
                    "User said: '" + message + "'. "
                    "If they said 'none', 'ok thanks', 'no' → ask warmly: "
                    "'Would you like to apply for one of the plans I showed, "
                    "speak to a human advisor, or would you like to change your budget? 😊' "
                    "If they asked a question about a plan → answer it briefly. "
                    "Radio buttons with plan names remain visible."
                ) if profile.get("_plans_already_shown") else (
                    "STEP 11: Recommend exactly 2-3 insurance plans from the Policy Knowledge Base. "
                    "KB recommendations are shown ABOVE this prompt — use them as the PRIMARY source. "
                    "For each plan clearly state: name, company, premium, coverage, waiting period, "
                    "why it specifically suits THIS user's age/budget/medical conditions/city. "
                    "End with: 'Which plan would you like to know more about?' "
                    "Radio buttons with plan names appear automatically. "
                    "RAG CONTEXT:\n" + (rag_ctx or "(Use built-in knowledge for Indian insurance)")
                )
            ),
            "explain_plan":(
                "STEP 12: User selected a specific plan. Explain it in detail: "
                "coverage amount, monthly premium, waiting period, network hospitals, "
                "key benefits, why this plan is perfect for this user specifically. "
                "Keep it friendly and clear (3-5 sentences). "
                "ALWAYS end with exactly: 'Would you like to apply now or compare other plans?' "
                "Buttons will appear automatically."
            ),
            "ask_escalation":(
                "STEP 13: Ask if they want a human advisor. "
                "Radio buttons will appear. Be warm and friendly."
            ),
            "ask_rating":(
                f"STEP 14: Thank {name or 'the user'} warmly. Ask for 1-5 star rating. "
                "Star rating UI will appear."
            ),
            "farewell":(
                f"STEP 15: Warm personal farewell to {name or 'the user'}. "
                "Include 🎉. Mention documents deleted securely. "
                f"End with: '🎉 Thank you {name or ''}! Your insurance journey is complete.'"
            ),
        }.get(stage, "Continue the conversation following the PolicyBot flow.")

        lang_rule = f"CRITICAL: Reply ONLY in {language}. Do NOT use any other language regardless of what the user wrote."
        return f"""CURRENT STEP: {stage}
LANGUAGE: {language}
USER NAME: {name or '(not yet provided)'}
INSURANCE TYPE: {ins}

SESSION PROFILE:
{json.dumps(safe_p, indent=2)}

RECENT CONVERSATION:
{recent or '(start of conversation)'}

USER MESSAGE: "{message}"

YOUR TASK:
{instructions}

{lang_rule}
Reply as PolicyBot (warm, 2-3 sentences, correct step only):"""

    # ── Helpers ────────────────────────────────────────────────────────────
    def _confidence(self, profile, stage):
        if stage not in ("recommendation","explain_plan","ask_escalation","ask_rating","farewell"):
            return ""
        return ("High ✅ — Identity verified. Faster approval guaranteed."
                if profile.get("gov_id_verified")
                else "Low ⚠️ — Verify your ID for better rates & faster approval.")

    def _rag_query(self, profile):
        return " ".join(filter(None,[
            profile.get("insurance_type",""),
            profile.get("medical_conditions",""),
            profile.get("family_members",""),
            profile.get("budget_range",""),
            profile.get("city",""),
        ]))

    def _extract_rating(self, message):
        nums = re.findall(r'\b([1-5])\b', message)
        return int(nums[0]) if nums else None

    def _detect_plan(self, message):
        msg = message.lower()
        for k in ["star health","hdfc ergo","niva bupa","care","aditya birla","bajaj allianz",
                  "lic","icici lombard","plan 1","plan 2","plan 3","option 1","option 2"]:
            if k in msg:
                return message.strip()
        return None

    def _module(self, stage):
        m = {
            "insurance_type":"welcome","collect_name":"onboarding","collect_age":"onboarding",
            "doc_upload":"verification","verify_wait":"verification",
            "collect_gender":"profiling","collect_city":"profiling",
            "collect_family":"profiling","collect_medical":"profiling",
            "condition_report_upload":"condition_check","condition_report_wait":"condition_check",
            "optional_health_check":"condition_check",
            "vehicle_history":"condition_check","vehicle_doc_upload":"condition_check",
            "life_docs":"condition_check","travel_declare":"condition_check",
            "property_history":"condition_check",
            "collect_budget":"profiling","recommendation":"recommendation",
            "explain_plan":"recommendation","ask_escalation":"escalation",
            "ask_rating":"feedback","farewell":"feedback",
        }
        return m.get(stage, "general")