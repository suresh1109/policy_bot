"""Lead Manager & Fraud Checker v3"""
import logging

log = logging.getLogger("LeadManager")

INTEREST_KEYWORDS = [
    "apply", "i like", "looks good", "interested", "want this",
    "buy", "purchase", "get this plan", "sign up", "enroll",
    "this is good", "perfect", "great plan", "apply now",
    "tell me more", "how to apply", "i want", "choose this"
]


class LeadManager:
    def __init__(self, db):
        self.db = db

    def detect(self, user_id: str, message: str):
        """Detect interest signals and store lead."""
        msg = message.lower()
        if any(k in msg for k in INTEREST_KEYWORDS):
            self.db.store_lead(user_id, "auto-detected", "high", "interested")
            log.info(f"Lead detected for {user_id}")
            return True
        return False

    def mark(self, user_id: str, plan_name: str,
             interest_level: str = "medium", status: str = "interested"):
        """Manually mark a lead."""
        self.db.store_lead(user_id, plan_name or "unknown", interest_level, status)

    # ── v2 backward-compatible aliases ────────────────────────────────────
    def detect_and_store(self, user_id: str, message: str, plans=None):
        """v2 alias for detect()"""
        return self.detect(user_id, message)

    def mark_lead(self, user_id: str, plan_name: str,
                  interest_level: str = "medium", status: str = "interested"):
        """v2 alias for mark()"""
        return self.mark(user_id, plan_name, interest_level, status)


class FraudChecker:
    def check(self, data: dict) -> dict:
        flags, risk = [], "LOW"

        # Age validation
        age = data.get("age")
        if age:
            try:
                a = int(age)
                if a < 0 or a > 120:
                    flags.append("Implausible age value")
                    risk = "HIGH"
                elif a < 14:
                    flags.append("Age below minimum eligible (14)")
                    risk = "MEDIUM"
            except Exception:
                pass

        # Medical condition without proof
        med = (data.get("medical_conditions") or "").lower()
        if med and med not in ["none", "no", "", "healthy", "nothing"]:
            if not data.get("medical_proof_uploaded"):
                flags.append("Medical condition declared without supporting proof")
                risk = self._max_risk(risk, "MEDIUM")

        # ID not verified
        if not data.get("gov_id_verified"):
            flags.append("Government ID not verified")
            risk = self._max_risk(risk, "LOW")

        return {
            "risk_level": risk,
            "flags": flags,
            "gov_id_verified": bool(data.get("gov_id_verified")),
            "user_id": data.get("user_id", ""),
            "recommendation": "Proceed normally" if risk == "LOW" else
                              "Verify documents before policy issuance" if risk == "MEDIUM" else
                              "Manual review required"
        }

    def _max_risk(self, current: str, new: str) -> str:
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        return new if order.get(new, 0) > order.get(current, 0) else current
