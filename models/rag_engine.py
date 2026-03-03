"""RAGEngine v3"""
import os, json, pickle
import numpy as np
try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    PDF_OK = False

VECTOR_DB = "vector_db/embeddings.pkl"
CHUNKS_DB = "vector_db/chunks.json"

DEFAULT = """
=== STAR FAMILY HEALTH OPTIMA ===
Plan: Star Family Health Optima | Coverage: ₹3L–₹25L | Premium: ₹800–₹2,500/month
Waiting: 30 days general, 2 years pre-existing | CSR: 82%
Benefits: Cashless 14,000+ hospitals, maternity, no-claim bonus 10%
Hospitals Coimbatore: PSG, KMCH, Ganga Hospital, Kovai Medical
Best For: Families 2–5, budget ₹1,000–₹3,000/month

=== HDFC ERGO OPTIMA RESTORE ===
Plan: HDFC Ergo Optima Restore | Coverage: ₹3L–₹50L | Premium: ₹700–₹3,000/month
Waiting: 30 days, 4 years pre-existing | CSR: 91%
Benefits: Restore benefit (refills sum insured), no room rent sub-limits, 13,000+ hospitals
Hospitals Coimbatore: PSG, KMCH, Sri Ramakrishna
Best For: Individuals wanting restoration benefit

=== NIVA BUPA HEALTH COMPANION ===
Plan: Niva Bupa Health Companion | Coverage: ₹3L–₹1Cr | Premium: ₹600–₹4,000/month
Waiting: 30 days, 3 years pre-existing | CSR: 89%
Benefits: No room rent limit, global cover, wellness rewards
Hospitals Coimbatore: KMCH, Ganga, G Kuppuswamy Naidu

=== CARE HEALTH INSURANCE ===
Plan: Care Health Insurance | Coverage: ₹4L–₹6Cr | Premium: ₹500–₹3,500/month
Waiting: 4 years pre-existing | CSR: 95% | Benefits: Unlimited recharge, no co-pay
Network: 21,000+ India-wide

=== ADITYA BIRLA ACTIV HEALTH ===
Plan: Aditya Birla Activ Health Enhanced | Coverage: ₹5L–₹2Cr | Premium: ₹900–₹4,000/month
Waiting: 2 years (diabetes plans available) | CSR: 93%
Benefits: Chronic disease management, HealthReturns up to 100%, OPD cover
Best For: Diabetes/BP patients

=== STAR DIABETES SAFE ===
Plan: Star Diabetes Safe | Coverage: ₹3L–₹10L | Premium: ₹1,200–₹3,000/month
Waiting: ZERO days for diabetes complications | CSR: 82%
Benefits: Day-1 diabetes cover, dialysis, laser eye treatment
Best For: Diabetes patients needing immediate coverage

=== PERSONAL ACCIDENT ===
Plan: New India Personal Accident | Coverage: ₹1L–₹25L | Premium: ₹100–₹500/month
Waiting: None | Benefits: Accidental death, disability income, medical expenses
Best For: Students, bike riders, young professionals

=== SENIOR CITIZEN RED CARPET ===
Plan: Star Senior Citizens Red Carpet | Coverage: ₹1L–₹25L | Premium: ₹1,500–₹5,000/month
Waiting: 1 year pre-existing | No medical test up to age 75
Best For: Parents 60+ years

=== CRITICAL ILLNESS RIDER ===
Plan: HDFC Ergo Critical Illness | Coverage: ₹5L–₹1Cr lump sum | Premium: ₹200–₹1,000/month add-on
Waiting: 90 days survival | Covers: 36 conditions including cancer, heart attack, kidney failure

=== VEHICLE — BAJAJ ALLIANZ ===
Plan: Bajaj Allianz Comprehensive Motor | Premium: ₹3,000–₹15,000/year
Benefits: Cashless 6,000+ garages, roadside assist, zero-dep add-on
Waiting: None

=== VEHICLE — ICICI LOMBARD ===
Plan: ICICI Lombard Motor | Premium: ₹2,500–₹12,000/year
Benefits: Instant settlement, digital inspection, 12,800+ garages

=== LIFE — LIC JEEVAN ANAND ===
Plan: LIC Jeevan Anand | Coverage: ₹5L–₹1Cr | Premium: ₹2,000–₹8,000/month
Term: 15–35 years | Benefits: Death + maturity, bonus, loan facility
Best For: Long-term savings + protection

=== TRAVEL — BAJAJ ALLIANZ ===
Plan: Bajaj Allianz Travel Companion | Medical Cover: ₹25L | Premium: ₹500–₹2,000/trip
Benefits: Emergency evacuation, trip cancellation, passport loss, baggage cover

=== TERM LIFE — HDFC CLICK2PROTECT ===
Plan: HDFC Click2Protect Life | Coverage: ₹50L–₹10Cr | Premium: ₹800–₹5,000/month
Term: 10–40 years | CSR: 98.6% | Benefits: Pure term life, income replacement, critical illness rider
Best For: Breadwinners with dependents

=== POLICY GLOSSARY ===
Waiting Period: Time before pre-existing conditions are covered.
Co-payment: % of claim you pay. 20% co-pay means you pay ₹20K on ₹1L bill.
Room Rent Limit: Max daily hospital room rent covered. Affects total bill.
Network Hospital: Cashless settlement — no upfront payment needed.
CSR (Claim Settlement Ratio): % of claims successfully settled.
Sum Insured: Max payout per year. Resets annually.
No Claim Bonus: Extra coverage added each year without claims.
Restore Benefit: Sum insured refills automatically if exhausted.
"""


class RAGEngine:
    def __init__(self):
        os.makedirs("vector_db", exist_ok=True)
        self.chunks = self._load_chunks()
        self.embeddings = self._load_embeddings()

    def _load_chunks(self):
        if os.path.exists(CHUNKS_DB):
            with open(CHUNKS_DB) as f:
                return json.load(f)
        chunks = self._chunk(DEFAULT, "default_master")
        self._save_chunks(chunks)
        return chunks

    def _load_embeddings(self):
        if os.path.exists(VECTOR_DB):
            with open(VECTOR_DB, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_chunks(self, chunks):
        with open(CHUNKS_DB, "w") as f:
            json.dump(chunks, f)

    def _save_emb(self):
        with open(VECTOR_DB, "wb") as f:
            pickle.dump(self.embeddings, f)

    def _chunk(self, text, source, size=350):
        words = text.split()
        chunks = []
        for i in range(0, len(words), size - 30):
            c = " ".join(words[i:i+size])
            chunks.append({"id": f"{source}_{i}", "text": c, "source": source})
        return chunks

    def add_document(self, path, filename):
        ext = os.path.splitext(filename)[1].lower()
        text = ""
        if ext == ".pdf" and PDF_OK:
            try:
                with open(path, "rb") as f:
                    r = PyPDF2.PdfReader(f)
                    text = " ".join(p.extract_text() or "" for p in r.pages)
            except Exception as e:
                print(f"PDF error: {e}")
        elif ext in [".txt", ".md"]:
            with open(path, errors="ignore") as f:
                text = f.read()
        if text:
            src = os.path.splitext(filename)[0]
            self.chunks.extend(self._chunk(text, src))
            self._save_chunks(self.chunks)
            return True
        return False

    def _cosine(self, a, b):
        a, b = np.array(a), np.array(b)
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / n) if n else 0.0

    def _keyword_search(self, query, k=5):
        qw = set(query.lower().split())
        scored = sorted(
            [(len(qw & set(c["text"].lower().split())), c) for c in self.chunks],
            reverse=True, key=lambda x: x[0])
        return [c for s, c in scored[:k] if s > 0]

    def search(self, query, k=5, gemini=None):
        if gemini and self.embeddings:
            try:
                qe = gemini.embed_query(query)
                if qe:
                    scored = [(self._cosine(qe, e), cid)
                              for cid, e in self.embeddings.items()]
                    scored.sort(reverse=True)
                    result = []
                    for sim, cid in scored[:k]:
                        if sim > 0.2:
                            c = next((x for x in self.chunks if x["id"]==cid), None)
                            if c:
                                result.append(c)
                    if result:
                        return result
            except Exception as e:
                print(f"Embed search error: {e}")
        return self._keyword_search(query, k)

    def get_context(self, query, gemini=None):
        chunks = self.search(query, k=5, gemini=gemini)
        return "\n\n---\n\n".join(c["text"] for c in chunks) if chunks else ""
