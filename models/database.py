"""Database v4 — Full PolicyBot schema + condition check columns"""
import sqlite3, datetime, os
from contextlib import contextmanager

DB_PATH = "policybot.db"

class Database:
    def __init__(self): self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
        try: yield conn; conn.commit()
        except Exception as e: conn.rollback(); raise e
        finally: conn.close()

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY, name TEXT, age INTEGER,
                gender TEXT, city TEXT, occupation TEXT, annual_income TEXT,
                family_members TEXT, insurance_type TEXT, budget_range TEXT,
                existing_insurance TEXT, medical_conditions TEXT, selected_plan TEXT,
                gov_id_verified INTEGER DEFAULT 0, doc_type_found TEXT,
                medical_proof_uploaded INTEGER DEFAULT 0, vehicle_doc_uploaded INTEGER DEFAULT 0,
                condition_selected TEXT,
                condition_report_uploaded INTEGER DEFAULT 0,
                condition_report_result TEXT,
                vehicle_history TEXT,
                life_docs TEXT,
                travel_declare TEXT,
                property_history TEXT,
                theme_preference TEXT DEFAULT 'neon', language TEXT DEFAULT 'English',
                onboarding_stage TEXT DEFAULT 'insurance_type',
                plans_shown INTEGER DEFAULT 0,
                plans_shown_names TEXT,
                created_at TEXT, updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, session_id TEXT,
                message TEXT, bot_reply TEXT, module TEXT, language TEXT, timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, plan_name TEXT,
                premium TEXT, coverage TEXT, waiting_period TEXT, reason TEXT,
                accepted INTEGER DEFAULT 0, timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS leads (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, plan_name TEXT,
                interest_level TEXT, lead_status TEXT, phone TEXT, best_call_time TEXT, timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                score INTEGER, comment TEXT, timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, phone TEXT,
                best_time TEXT, plan_name TEXT, status TEXT DEFAULT 'pending', timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT, file_path TEXT,
                doc_type TEXT, user_id TEXT, active INTEGER DEFAULT 1,
                deleted INTEGER DEFAULT 0, uploaded_at TEXT
            );
            CREATE TABLE IF NOT EXISTS selected_options (
                id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, stage TEXT,
                question TEXT, selected_option TEXT, timestamp TEXT
            );
            CREATE TABLE IF NOT EXISTS policy_documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                file_path   TEXT NOT NULL,
                file_size   INTEGER DEFAULT 0,
                doc_hash    TEXT UNIQUE,
                status      TEXT DEFAULT 'processing',
                version     INTEGER DEFAULT 1,
                uploaded_at TEXT,
                updated_at  TEXT,
                uploaded_by TEXT DEFAULT 'admin'
            );
            CREATE TABLE IF NOT EXISTS policy_plans (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id              INTEGER,
                company_name        TEXT,
                plan_name           TEXT,
                insurance_type      TEXT,
                coverage_amount     TEXT,
                premium_range       TEXT,
                waiting_period      TEXT,
                conditions_covered  TEXT,
                exclusions          TEXT,
                claim_process       TEXT,
                network_hospitals   TEXT,
                eligibility_age     TEXT,
                special_benefits    TEXT,
                raw_summary         TEXT,
                is_master           INTEGER DEFAULT 0,
                active              INTEGER DEFAULT 1,
                recommend_count     INTEGER DEFAULT 0,
                view_count          INTEGER DEFAULT 0,
                created_at          TEXT,
                updated_at          TEXT
            );
            CREATE TABLE IF NOT EXISTS policy_versions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id      INTEGER,
                version     INTEGER,
                filename    TEXT,
                file_path   TEXT,
                changed_at  TEXT,
                changed_by  TEXT DEFAULT 'admin',
                change_note TEXT
            );
            CREATE TABLE IF NOT EXISTS kb_analytics (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT,
                plan_id     INTEGER,
                plan_name   TEXT,
                user_id     TEXT,
                detail      TEXT,
                created_at  TEXT
            );
            """)
        # ── Migrate existing DB: safely add any missing columns ──────────────
        # SQLite does not support ADD COLUMN IF NOT EXISTS, so we catch the error
        self._migrate()

    def _migrate(self):
        """Add any columns that exist in the schema but are missing from an older DB file.
        SQLite has no ADD COLUMN IF NOT EXISTS, so we attempt each ALTER and silently
        ignore the error if the column already exists."""
        new_columns = [
            # (column_name, definition)
            ("plans_shown",          "INTEGER DEFAULT 0"),
            ("plans_shown_names",    "TEXT"),
            ("condition_selected",        "TEXT"),
            ("condition_report_uploaded", "INTEGER DEFAULT 0"),
            ("condition_report_result",   "TEXT"),
            ("vehicle_history",           "TEXT"),
            ("life_docs",                 "TEXT"),
            ("travel_declare",            "TEXT"),
            ("property_history",          "TEXT"),
            ("medical_proof_uploaded",    "INTEGER DEFAULT 0"),
            ("vehicle_doc_uploaded",      "INTEGER DEFAULT 0"),
            ("doc_type_found",            "TEXT"),
            ("occupation",                "TEXT"),
            ("annual_income",             "TEXT"),
        ]
        with self._conn() as conn:
            for col, defn in new_columns:
                try:
                    conn.execute(f"ALTER TABLE users ADD COLUMN {col} {defn}")
                except Exception:
                    pass  # column already exists — safe to ignore

    def _now(self): return datetime.datetime.utcnow().isoformat()

    def get_user_profile(self, uid):
        with self._conn() as c:
            r = c.execute("SELECT * FROM users WHERE user_id=?", (uid,)).fetchone()
            return dict(r) if r else {}

    def upsert_user_profile(self, uid, fields):
        if not fields: return
        now = self._now()
        # Drop None/"" values
        fields = {k: v for k, v in fields.items() if v is not None and v != ""}
        if not fields: return
        # Only write columns that actually exist in the DB (guards against old schema)
        with self._conn() as c:
            existing_cols = {r[1] for r in
                             c.execute("PRAGMA table_info(users)").fetchall()}
        fields = {k: v for k, v in fields.items() if k in existing_cols}
        if not fields: return

        existing = self.get_user_profile(uid)
        if not existing:
            fields.update({"user_id": uid, "created_at": now, "updated_at": now})
            cols = ", ".join(fields); ph = ", ".join(["?"]*len(fields))
            with self._conn() as c:
                c.execute(f"INSERT OR IGNORE INTO users ({cols}) VALUES ({ph})",
                          list(fields.values()))
        else:
            fields["updated_at"] = now
            sc = ", ".join(f"{k}=?" for k in fields)
            with self._conn() as c:
                c.execute(f"UPDATE users SET {sc} WHERE user_id=?",
                          list(fields.values()) + [uid])

    def reset_session_profile(self, uid):
        """Clear ALL data for a fresh session — wipes profile AND chat history.
        Builds the UPDATE dynamically so it never fails on missing columns."""
        now = self._now()
        # Fields we WANT to reset — only included if the column actually exists
        wanted = {
            "name": None, "age": None, "gender": None, "city": None,
            "family_members": None, "insurance_type": None, "budget_range": None,
            "medical_conditions": None, "selected_plan": None,
            "gov_id_verified": 0, "doc_type_found": None,
            "condition_selected": None, "condition_report_uploaded": 0,
            "condition_report_result": None,
            "vehicle_history": None, "life_docs": None,
            "travel_declare": None, "property_history": None,
            "onboarding_stage": "insurance_type",
            "plans_shown": 0,
            "plans_shown_names": None,
        }
        with self._conn() as c:
            # Get actual columns in the DB right now
            existing_cols = {r[1] for r in
                             c.execute("PRAGMA table_info(users)").fetchall()}
            # Only reset fields that exist — safe against old/unmigrated DB
            safe = {k: v for k, v in wanted.items() if k in existing_cols}
            if safe:
                safe["updated_at"] = now
                parts  = ", ".join(f"{k}=?" for k in safe)
                values = list(safe.values()) + [uid]
                c.execute(f"UPDATE users SET {parts} WHERE user_id=?", values)
            # Always clear history and selected options (these tables always exist)
            c.execute("DELETE FROM chats WHERE user_id=?", (uid,))
            c.execute("DELETE FROM selected_options WHERE user_id=?", (uid,))

    def update_verification(self, uid, field, value):
        now = self._now()
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO users (user_id,created_at,updated_at) VALUES (?,?,?)", (uid,now,now))
            c.execute(f"UPDATE users SET {field}=?, updated_at=? WHERE user_id=?", (value,now,uid))

    def search_users(self, q="", limit=50, offset=0):
        with self._conn() as c:
            if q:
                like = f"%{q}%"
                rows = c.execute("SELECT * FROM users WHERE name LIKE ? OR city LIKE ? OR insurance_type LIKE ? ORDER BY created_at DESC LIMIT ? OFFSET ?", (like,like,like,limit,offset)).fetchall()
            else: rows = c.execute("SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit,offset)).fetchall()
            return [dict(r) for r in rows]

    def get_all_users_raw(self):
        with self._conn() as c: return c.execute("SELECT * FROM users").fetchall()

    def count_users(self, q=""):
        with self._conn() as c:
            if q:
                like = f"%{q}%"
                return c.execute("SELECT COUNT(*) FROM users WHERE name LIKE ? OR city LIKE ?", (like,like)).fetchone()[0]
            return c.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def store_chat(self, user_id, message, bot_reply, module="", session_id="", language="English"):
        with self._conn() as c:
            c.execute("INSERT INTO chats (user_id,session_id,message,bot_reply,module,language,timestamp) VALUES (?,?,?,?,?,?,?)",
                      (user_id,session_id,message,bot_reply,module,language,self._now()))

    def get_chat_history(self, uid, limit=20):
        with self._conn() as c:
            rows = c.execute("SELECT * FROM chats WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (uid,limit)).fetchall()
            return [dict(r) for r in reversed(rows)]

    def search_chats(self, user_id=None, q="", limit=50):
        with self._conn() as c:
            if user_id: rows = c.execute("SELECT * FROM chats WHERE user_id=? ORDER BY timestamp DESC LIMIT ?", (user_id,limit)).fetchall()
            elif q:
                like = f"%{q}%"
                rows = c.execute("SELECT * FROM chats WHERE message LIKE ? OR bot_reply LIKE ? ORDER BY timestamp DESC LIMIT ?", (like,like,limit)).fetchall()
            else: rows = c.execute("SELECT * FROM chats ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [dict(r) for r in rows]

    def store_recommendation(self, uid, plan):
        with self._conn() as c:
            c.execute("INSERT INTO recommendations (user_id,plan_name,premium,coverage,waiting_period,reason,timestamp) VALUES (?,?,?,?,?,?,?)",
                      (uid,plan.get("name",""),plan.get("premium",""),plan.get("coverage",""),plan.get("waiting_period",""),plan.get("reason",""),self._now()))

    def store_lead(self, uid, plan, interest, status):
        with self._conn() as c:
            c.execute("INSERT INTO leads (user_id,plan_name,interest_level,lead_status,timestamp) VALUES (?,?,?,?,?)",
                      (uid,plan,interest,status,self._now()))

    def get_leads(self):
        with self._conn() as c: return [dict(r) for r in c.execute("SELECT * FROM leads ORDER BY timestamp DESC LIMIT 100").fetchall()]

    def store_rating(self, uid, score, comment=""):
        with self._conn() as c:
            c.execute("INSERT INTO ratings (user_id,score,comment,timestamp) VALUES (?,?,?,?)", (uid,score,comment,self._now()))

    def get_ratings(self):
        with self._conn() as c: return [dict(r) for r in c.execute("SELECT r.*,u.name FROM ratings r LEFT JOIN users u ON r.user_id=u.user_id ORDER BY r.timestamp DESC LIMIT 100").fetchall()]

    def store_escalation(self, uid, phone, best_time, plan_name):
        with self._conn() as c:
            c.execute("INSERT INTO escalations (user_id,phone,best_time,plan_name,timestamp) VALUES (?,?,?,?,?)", (uid,phone,best_time,plan_name,self._now()))

    def store_document(self, filename, file_path, doc_type, user_id=""):
        with self._conn() as c:
            c.execute("INSERT INTO documents (filename,file_path,doc_type,user_id,uploaded_at) VALUES (?,?,?,?,?)",
                      (filename,file_path,doc_type,user_id,self._now()))

    def get_documents(self):
        with self._conn() as c: return [dict(r) for r in c.execute("SELECT * FROM documents WHERE deleted=0 ORDER BY uploaded_at DESC").fetchall()]

    def get_user_documents(self, uid):
        with self._conn() as c: return [dict(r) for r in c.execute("SELECT * FROM documents WHERE user_id=? AND deleted=0", (uid,)).fetchall()]

    def delete_user_documents(self, uid):
        with self._conn() as c: c.execute("UPDATE documents SET deleted=1, file_path='' WHERE user_id=?", (uid,))

    def toggle_document(self, doc_id):
        with self._conn() as c: c.execute("UPDATE documents SET active=1-active WHERE id=?", (doc_id,))

    def delete_document(self, doc_id):
        with self._conn() as c: c.execute("DELETE FROM documents WHERE id=?", (doc_id,))

    def store_option_selection(self, uid, stage, question, option):
        with self._conn() as c:
            c.execute("INSERT INTO selected_options (user_id,stage,question,selected_option,timestamp) VALUES (?,?,?,?,?)",
                      (uid,stage,question,option,self._now()))

    def get_analytics(self):
        with self._conn() as c:
            tu = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            vf = c.execute("SELECT COUNT(*) FROM users WHERE gov_id_verified=1").fetchone()[0]
            tc = c.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            tl = c.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
            db = c.execute("SELECT COUNT(*) FROM users WHERE medical_conditions LIKE '%Diabetes%'").fetchone()[0]
            ar = c.execute("SELECT AVG(score) FROM ratings").fetchone()[0] or 0
            tr = c.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
            es = c.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
            it = [dict(r) for r in c.execute("SELECT insurance_type,COUNT(*) c FROM users WHERE insurance_type IS NOT NULL GROUP BY insurance_type ORDER BY c DESC LIMIT 6").fetchall()]
            pp = [dict(r) for r in c.execute("SELECT plan_name,COUNT(*) c FROM recommendations GROUP BY plan_name ORDER BY c DESC LIMIT 5").fetchall()]
            ls = [dict(r) for r in c.execute("SELECT language,COUNT(*) c FROM chats WHERE language IS NOT NULL GROUP BY language ORDER BY c DESC").fetchall()]
            rd = [dict(r) for r in c.execute("SELECT score,COUNT(*) c FROM ratings GROUP BY score ORDER BY score").fetchall()]
            return {"total_users":tu,"verified_users":vf,"verified_pct":round(vf/max(tu,1)*100,1),
                    "total_chats":tc,"total_leads":tl,"diabetes_users":db,"avg_rating":round(ar,1),
                    "total_ratings":tr,"total_escalations":es,"conversion_rate":round(tl/max(tu,1)*100,1),
                    "popular_insurance_types":it,"popular_plans":pp,"language_stats":ls,"rating_distribution":rd}

    # ═══════════════════════════════════════════════════════════════════════
    # POLICY KNOWLEDGE BASE METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def _file_hash(self, file_bytes: bytes) -> str:
        import hashlib
        return hashlib.sha256(file_bytes).hexdigest()[:32]

    def kb_doc_exists(self, file_bytes: bytes) -> bool:
        h = self._file_hash(file_bytes)
        with self._conn() as c:
            r = c.execute("SELECT id FROM policy_documents WHERE doc_hash=?", (h,)).fetchone()
            return r is not None

    def kb_store_document(self, filename, file_path, file_bytes, uploaded_by="admin"):
        now = self._now()
        h = self._file_hash(file_bytes)
        with self._conn() as c:
            c.execute("""INSERT INTO policy_documents
                (filename, file_path, file_size, doc_hash, status, version, uploaded_at, updated_at, uploaded_by)
                VALUES (?,?,?,?,'processing',1,?,?,?)""",
                (filename, file_path, len(file_bytes), h, now, now, uploaded_by))
            return c.execute("SELECT last_insert_rowid()").fetchone()[0]

    def kb_update_doc_status(self, doc_id, status):
        with self._conn() as c:
            c.execute("UPDATE policy_documents SET status=?, updated_at=? WHERE id=?",
                      (status, self._now(), doc_id))

    def kb_store_plans(self, doc_id, plans: list, is_master=0):
        now = self._now()
        with self._conn() as c:
            c.execute("DELETE FROM policy_plans WHERE doc_id=?", (doc_id,))
            for p in plans:
                c.execute("""INSERT INTO policy_plans
                    (doc_id,company_name,plan_name,insurance_type,coverage_amount,
                     premium_range,waiting_period,conditions_covered,exclusions,
                     claim_process,network_hospitals,eligibility_age,special_benefits,
                     raw_summary,is_master,active,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)""",
                    (doc_id,
                     p.get("company_name",""), p.get("plan_name",""),
                     p.get("insurance_type",""), p.get("coverage_amount",""),
                     p.get("premium_range",""), p.get("waiting_period",""),
                     p.get("conditions_covered",""), p.get("exclusions",""),
                     p.get("claim_process",""), p.get("network_hospitals",""),
                     p.get("eligibility_age",""), p.get("special_benefits",""),
                     p.get("raw_summary",""), is_master, now, now))

    def kb_get_all_docs(self):
        with self._conn() as c:
            return [dict(r) for r in c.execute("""
                SELECT d.*,
                  (SELECT COUNT(*) FROM policy_plans p WHERE p.doc_id=d.id AND p.active=1) AS plan_count
                FROM policy_documents d
                ORDER BY d.uploaded_at DESC""").fetchall()]

    def kb_get_doc(self, doc_id):
        with self._conn() as c:
            r = c.execute("SELECT * FROM policy_documents WHERE id=?", (doc_id,)).fetchone()
            return dict(r) if r else None

    def kb_get_plans(self, doc_id=None, active_only=True):
        with self._conn() as c:
            base = "SELECT p.*, d.filename FROM policy_plans p LEFT JOIN policy_documents d ON p.doc_id=d.id"
            conditions = []
            params = []
            if active_only:
                conditions.append("p.active=1")
            if doc_id is not None:
                conditions.append("p.doc_id=?")
                params.append(doc_id)
            q = base + (" WHERE " + " AND ".join(conditions) if conditions else "") + " ORDER BY p.is_master DESC, p.company_name"
            return [dict(r) for r in c.execute(q, params).fetchall()]

    def kb_get_all_plans_for_recommendation(self):
        """Return all active plans for RAG recommendation engine."""
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM policy_plans WHERE active=1 ORDER BY is_master DESC, recommend_count DESC"
            ).fetchall()]

    def kb_delete_doc(self, doc_id):
        with self._conn() as c:
            c.execute("DELETE FROM policy_plans WHERE doc_id=?", (doc_id,))
            c.execute("DELETE FROM policy_documents WHERE id=?", (doc_id,))

    def kb_toggle_plan(self, plan_id):
        with self._conn() as c:
            c.execute("UPDATE policy_plans SET active=1-active, updated_at=? WHERE id=?",
                      (self._now(), plan_id))

    def kb_delete_plan(self, plan_id):
        with self._conn() as c:
            c.execute("DELETE FROM policy_plans WHERE id=?", (plan_id,))

    def kb_save_version(self, doc_id, filename, file_path, version, note=""):
        with self._conn() as c:
            c.execute("""INSERT INTO policy_versions (doc_id,version,filename,file_path,changed_at,change_note)
                         VALUES (?,?,?,?,?,?)""",
                      (doc_id, version, filename, file_path, self._now(), note))
        with self._conn() as c:
            c.execute("UPDATE policy_documents SET version=?,file_path=?,filename=?,updated_at=? WHERE id=?",
                      (version, file_path, filename, self._now(), doc_id))

    def kb_get_versions(self, doc_id):
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM policy_versions WHERE doc_id=? ORDER BY version DESC", (doc_id,)
            ).fetchall()]

    def kb_log_event(self, event_type, plan_id=None, plan_name="", user_id="", detail=""):
        with self._conn() as c:
            c.execute("INSERT INTO kb_analytics (event_type,plan_id,plan_name,user_id,detail,created_at) VALUES (?,?,?,?,?,?)",
                      (event_type, plan_id, plan_name, user_id, detail, self._now()))

    def kb_increment_recommend(self, plan_id):
        with self._conn() as c:
            c.execute("UPDATE policy_plans SET recommend_count=recommend_count+1,updated_at=? WHERE id=?",
                      (self._now(), plan_id))

    def kb_get_analytics(self):
        with self._conn() as c:
            total_docs  = c.execute("SELECT COUNT(*) FROM policy_documents").fetchone()[0]
            active_plans = c.execute("SELECT COUNT(*) FROM policy_plans WHERE active=1").fetchone()[0]
            top_rec = [dict(r) for r in c.execute(
                "SELECT plan_name,SUM(recommend_count) as rc FROM policy_plans GROUP BY plan_name ORDER BY rc DESC LIMIT 5"
            ).fetchall()]
            missing = [dict(r) for r in c.execute(
                """SELECT plan_name,company_name FROM policy_plans
                   WHERE (coverage_amount IS NULL OR coverage_amount='')
                      OR (premium_range IS NULL OR premium_range='')
                      OR (waiting_period IS NULL OR waiting_period='')
                   LIMIT 10"""
            ).fetchall()]
            failed_searches = [dict(r) for r in c.execute(
                "SELECT detail, COUNT(*) c FROM kb_analytics WHERE event_type='failed_search' GROUP BY detail ORDER BY c DESC LIMIT 10"
            ).fetchall()]
            top_viewed = [dict(r) for r in c.execute(
                "SELECT plan_name, view_count FROM policy_plans WHERE view_count>0 ORDER BY view_count DESC LIMIT 5"
            ).fetchall()]
            return {
                "total_docs": total_docs, "active_plans": active_plans,
                "top_recommended": top_rec, "missing_info": missing,
                "failed_searches": failed_searches, "top_viewed": top_viewed,
            }

    def kb_seed_master(self):
        """Seed the default master policy data if no master exists yet."""
        with self._conn() as c:
            exists = c.execute("SELECT id FROM policy_plans WHERE is_master=1 LIMIT 1").fetchone()
            if exists:
                return False
        # Insert master doc placeholder
        now = self._now()
        with self._conn() as c:
            c.execute("""INSERT INTO policy_documents
                (filename,file_path,file_size,doc_hash,status,version,uploaded_at,updated_at,uploaded_by)
                VALUES ('master_policy.docx','built-in',0,'MASTER_BUILTIN','active',1,?,?,'system')""",
                (now, now))
            doc_id = c.execute("SELECT last_insert_rowid()").fetchone()[0]

        from models.rag_engine import DEFAULT as RAG_DEFAULT
        master_plans = [
            {"company_name":"Star Health","plan_name":"Star Family Health Optima","insurance_type":"Health Insurance",
             "coverage_amount":"₹3L–₹25L","premium_range":"₹800–₹2,500/month","waiting_period":"30 days general, 2 years pre-existing",
             "conditions_covered":"Diabetes, Hypertension, Maternity, Day care","exclusions":"Cosmetic, dental, self-injury",
             "claim_process":"Cashless at network hospital or reimbursement within 30 days","network_hospitals":"14,000+",
             "eligibility_age":"18–65 years","special_benefits":"No-claim bonus 10%, maternity add-on, Ayush","raw_summary":"Family floater plan, best for 2–5 members"},
            {"company_name":"Star Health","plan_name":"Star Diabetes Safe","insurance_type":"Health Insurance",
             "coverage_amount":"₹3L–₹10L","premium_range":"₹1,200–₹3,000/month","waiting_period":"ZERO days for diabetes complications",
             "conditions_covered":"Diabetes Type 1 & 2, dialysis, eye complications","exclusions":"Self-inflicted, intoxication",
             "claim_process":"Cashless","network_hospitals":"14,000+","eligibility_age":"18–65 years",
             "special_benefits":"Day-1 diabetes cover, laser eye","raw_summary":"Best for existing diabetes patients"},
            {"company_name":"HDFC ERGO","plan_name":"HDFC Ergo Optima Restore","insurance_type":"Health Insurance",
             "coverage_amount":"₹3L–₹50L","premium_range":"₹700–₹3,000/month","waiting_period":"30 days, 4 years pre-existing",
             "conditions_covered":"All standard illnesses","exclusions":"OPD (optional), cosmetic",
             "claim_process":"Cashless or reimbursement","network_hospitals":"13,000+","eligibility_age":"18–65 years",
             "special_benefits":"Restore benefit, no room rent sub-limits, CSR 91%","raw_summary":"Best for restore benefit seekers"},
            {"company_name":"Niva Bupa","plan_name":"Niva Bupa Health Companion","insurance_type":"Health Insurance",
             "coverage_amount":"₹3L–₹1Cr","premium_range":"₹600–₹4,000/month","waiting_period":"30 days, 3 years pre-existing",
             "conditions_covered":"All standard","exclusions":"Cosmetic, war","claim_process":"Cashless or reimbursement",
             "network_hospitals":"10,000+","eligibility_age":"18–65 years","special_benefits":"No room rent limit, global cover, wellness rewards","raw_summary":"Premium health plan"},
            {"company_name":"Care Health","plan_name":"Care Health Insurance","insurance_type":"Health Insurance",
             "coverage_amount":"₹4L–₹6Cr","premium_range":"₹500–₹3,500/month","waiting_period":"4 years pre-existing",
             "conditions_covered":"All standard","exclusions":"Cosmetic, dental",
             "claim_process":"Cashless","network_hospitals":"21,000+","eligibility_age":"18–65 years",
             "special_benefits":"Unlimited recharge, no co-pay, CSR 95%","raw_summary":"High coverage care plan"},
            {"company_name":"Aditya Birla","plan_name":"Aditya Birla Activ Health Enhanced","insurance_type":"Health Insurance",
             "coverage_amount":"₹5L–₹2Cr","premium_range":"₹900–₹4,000/month","waiting_period":"2 years, diabetes plans available",
             "conditions_covered":"Diabetes, BP, chronic disease management","exclusions":"Cosmetic",
             "claim_process":"Cashless","network_hospitals":"10,000+","eligibility_age":"18–65 years",
             "special_benefits":"HealthReturns up to 100%, OPD cover, CSR 93%","raw_summary":"Best for diabetes/BP patients"},
            {"company_name":"Star Health","plan_name":"Star Senior Citizens Red Carpet","insurance_type":"Health Insurance",
             "coverage_amount":"₹1L–₹25L","premium_range":"₹1,500–₹5,000/month","waiting_period":"1 year pre-existing",
             "conditions_covered":"Senior citizen conditions","exclusions":"War, self-injury",
             "claim_process":"Cashless","network_hospitals":"14,000+","eligibility_age":"60–75 years",
             "special_benefits":"No medical test up to 75, 20% co-pay","raw_summary":"Best for senior citizens"},
            {"company_name":"HDFC Life","plan_name":"HDFC Click2Protect Life","insurance_type":"Term Life Insurance",
             "coverage_amount":"₹50L–₹10Cr","premium_range":"₹800–₹5,000/month","waiting_period":"90 days",
             "conditions_covered":"Death, terminal illness","exclusions":"Suicide within 1 year",
             "claim_process":"Nominee submits death certificate + policy + ID","network_hospitals":"N/A","eligibility_age":"18–65 years",
             "special_benefits":"CSR 98.6%, income replacement, critical illness rider","raw_summary":"Best term life plan"},
            {"company_name":"LIC","plan_name":"LIC Jeevan Anand","insurance_type":"Life Insurance",
             "coverage_amount":"₹5L–₹1Cr","premium_range":"₹2,000–₹8,000/month","waiting_period":"90 days",
             "conditions_covered":"Death, maturity","exclusions":"Suicide within 1 year",
             "claim_process":"Nominee submits documents to LIC branch","network_hospitals":"N/A","eligibility_age":"18–50 years",
             "special_benefits":"Death + maturity benefit, bonus, loan facility, government backed","raw_summary":"Traditional savings + protection"},
            {"company_name":"Bajaj Allianz","plan_name":"Bajaj Allianz Comprehensive Motor","insurance_type":"Vehicle Insurance",
             "coverage_amount":"As per IDV","premium_range":"₹3,000–₹15,000/year","waiting_period":"None",
             "conditions_covered":"Own damage, third party, theft, natural disasters","exclusions":"Drunk driving, racing",
             "claim_process":"Intimate within 48 hrs, surveyor visits, cashless repair","network_hospitals":"6,000+ garages","eligibility_age":"All vehicle owners",
             "special_benefits":"Roadside assistance, zero-dep add-on","raw_summary":"Comprehensive motor cover"},
            {"company_name":"Bajaj Allianz","plan_name":"Bajaj Allianz Travel Companion","insurance_type":"Travel Insurance",
             "coverage_amount":"Medical: ₹25L","premium_range":"₹500–₹2,000/trip","waiting_period":"None",
             "conditions_covered":"Medical emergency, trip cancel, baggage loss","exclusions":"Pre-existing conditions, war",
             "claim_process":"Submit bills + boarding pass to TPA","network_hospitals":"Worldwide TPA","eligibility_age":"1–70 years",
             "special_benefits":"Emergency evacuation, passport loss cover","raw_summary":"Best travel insurance"},
            {"company_name":"New India Assurance","plan_name":"New India Personal Accident","insurance_type":"Accident Insurance",
             "coverage_amount":"₹1L–₹25L","premium_range":"₹100–₹500/month","waiting_period":"None",
             "conditions_covered":"Accidental death, disability, medical expenses","exclusions":"Self-inflicted, suicide",
             "claim_process":"FIR + medical reports + claim form","network_hospitals":"All hospitals","eligibility_age":"5–70 years",
             "special_benefits":"Disability income, weekly benefit","raw_summary":"Basic PA cover"},
        ]
        self.kb_store_plans(doc_id, master_plans, is_master=1)
        self.kb_update_doc_status(doc_id, 'active')
        return True

    # ── Recommendation memory helpers ──────────────────────────────────────────
    def mark_plans_shown(self, uid: str, plan_names: list):
        """Mark that recommendations were shown to this user this session."""
        import json
        with self._conn() as c:
            c.execute("UPDATE users SET plans_shown=1, plans_shown_names=?, updated_at=? WHERE user_id=?",
                      (json.dumps(plan_names), self._now(), uid))

    def get_plans_shown(self, uid: str) -> dict:
        """Return {'shown': bool, 'plan_names': list}"""
        import json
        with self._conn() as c:
            row = c.execute("SELECT plans_shown, plans_shown_names FROM users WHERE user_id=?", (uid,)).fetchone()
            if not row:
                return {"shown": False, "plan_names": []}
            shown = bool(row["plans_shown"])
            names = []
            if row["plans_shown_names"]:
                try:
                    names = json.loads(row["plans_shown_names"])
                except Exception:
                    pass
            return {"shown": shown, "plan_names": names}

    def clear_plans_shown(self, uid: str):
        """Reset plans_shown — called when profile changes (budget, etc.)."""
        with self._conn() as c:
            c.execute("UPDATE users SET plans_shown=0, plans_shown_names=NULL WHERE user_id=?", (uid,))