"""GeminiManager v4 — Multi-key failover + model fallback + per-key cooldown"""
import os, time, logging, threading
import google.generativeai as genai

log = logging.getLogger("Gemini")

# Model priority order — if quota hit on primary, try next model
# gemini-2.0-flash-lite is free tier friendly and has separate quota pool
MODELS_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

# How long to cool down a key after quota exhaustion (seconds)
QUOTA_COOLDOWN_SECS = 65   # just over 1 minute (free tier resets each minute)
RATE_COOLDOWN_SECS  = 8    # short wait for per-minute rate limit


class GeminiManager:
    MODEL = MODELS_PRIORITY[0]  # default model

    def __init__(self):
        raw = [os.getenv(f"GEMINI_API_KEY_{i}", "") for i in range(1, 6)]  # support up to 5 keys
        self.keys  = [k for k in raw if k]
        self.cur   = 0
        self._lock = threading.Lock()

        # Per-key state: {key_index: {"requests": int, "errors": int, "cooldown_until": float}}
        self._key_state = {
            i: {"requests": 0, "errors": 0, "cooldown_until": 0.0}
            for i in range(len(self.keys))
        }

        # Per-key per-model state: tracks which models work for which key
        self._key_model_ok = {
            i: {m: True for m in MODELS_PRIORITY}
            for i in range(len(self.keys))
        }

        self._configure()
        log.info(f"[Gemini] Initialized with {len(self.keys)} key(s)")

    def _configure(self):
        if self.keys:
            genai.configure(api_key=self.keys[self.cur])

    def _set_cooldown(self, key_idx: int, seconds: float):
        self._key_state[key_idx]["cooldown_until"] = time.time() + seconds
        self._key_state[key_idx]["errors"] += 1
        log.warning(f"[Gemini] Key {key_idx+1} cooling down for {seconds:.0f}s")

    def _key_available(self, key_idx: int) -> bool:
        return time.time() >= self._key_state[key_idx]["cooldown_until"]

    def _next_available_key(self) -> int | None:
        """Return the next available key index, or None if all are cooling down."""
        for offset in range(len(self.keys)):
            idx = (self.cur + offset) % len(self.keys)
            if self._key_available(idx):
                return idx
        return None

    def _select_key(self, preferred_idx: int | None = None) -> bool:
        """Switch to the best available key. Returns False if all cooling down."""
        with self._lock:
            idx = self._next_available_key()
            if idx is None:
                return False
            if idx != self.cur:
                self.cur = idx
                genai.configure(api_key=self.keys[self.cur])
                log.info(f"[Gemini] Switched to key {self.cur + 1}")
            return True

    def get_key_usage(self):
        now = time.time()
        result = []
        for i in range(len(self.keys)):
            state = self._key_state.get(i, {})
            cd    = state.get("cooldown_until", 0)
            result.append({
                "key_index":  i + 1,
                "label":      f"Key {i + 1}",
                "requests":   state.get("requests", 0),
                "errors":     state.get("errors", 0),
                "masked":     "****" + self.keys[i][-4:] if i < len(self.keys) else "N/A",
                "status":     "cooling" if cd > now else "ready",
                "cooldown_remaining": max(0, round(cd - now)),
            })
        return result

    def _is_quota_error(self, err_str: str) -> bool:
        return any(x in err_str for x in [
            "quota", "429", "resource_exhausted", "resource exhausted",
            "daily limit", "exceeded", "limits"
        ])

    def _is_rate_error(self, err_str: str) -> bool:
        return any(x in err_str for x in [
            "rate", "timeout", "unavailable", "overloaded",
            "503", "500", "too many"
        ])

    def _is_invalid_error(self, err_str: str) -> bool:
        return any(x in err_str for x in [
            "invalid", "api key", "permission", "unauthorized", "401", "403",
            "not found", "blocked"
        ])

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024) -> str:
        if not self.keys:
            return ("I'm sorry, the AI service is not configured. "
                    "Please add a GEMINI_API_KEY to your .env file.")

        # Try each key + model combination with smart backoff
        tried_combinations = 0
        max_tries = len(self.keys) * len(MODELS_PRIORITY)

        for attempt in range(max_tries):
            # Pick best available key
            if not self._select_key():
                # All keys cooling — wait for shortest cooldown
                min_wait = min(
                    max(0, self._key_state[i]["cooldown_until"] - time.time())
                    for i in range(len(self.keys))
                )
                wait = min(min_wait + 0.5, 10)
                log.warning(f"[Gemini] All keys cooling, waiting {wait:.1f}s")
                time.sleep(wait)
                if not self._select_key():
                    break

            # Find a working model for this key
            model_name = None
            for m in MODELS_PRIORITY:
                if self._key_model_ok.get(self.cur, {}).get(m, True):
                    model_name = m
                    break
            if not model_name:
                # All models failed for this key — mark it cooling and try next key
                self._set_cooldown(self.cur, QUOTA_COOLDOWN_SECS)
                self.cur = (self.cur + 1) % len(self.keys)
                continue

            try:
                genai.configure(api_key=self.keys[self.cur])
                model = genai.GenerativeModel(
                    model_name,
                    system_instruction=system_prompt or None
                )
                resp = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.7
                    )
                )
                self._key_state[self.cur]["requests"] += 1
                if model_name != self.MODEL:
                    log.info(f"[Gemini] Used fallback model {model_name} on key {self.cur+1}")
                return resp.text

            except Exception as e:
                err = str(e).lower()
                tried_combinations += 1

                if self._is_quota_error(err):
                    # This key's quota exhausted — long cooldown, try other models first
                    log.warning(f"[Gemini] Key {self.cur+1} quota on {model_name}: {str(e)[:80]}")
                    self._key_model_ok[self.cur][model_name] = False
                    # If all models exhausted for this key → full cooldown
                    if not any(self._key_model_ok[self.cur].values()):
                        self._set_cooldown(self.cur, QUOTA_COOLDOWN_SECS)
                        # Reset model availability for next use
                        self._key_model_ok[self.cur] = {m: True for m in MODELS_PRIORITY}
                        self.cur = (self.cur + 1) % len(self.keys)

                elif self._is_rate_error(err):
                    log.warning(f"[Gemini] Key {self.cur+1} rate limit, brief pause")
                    self._set_cooldown(self.cur, RATE_COOLDOWN_SECS)
                    self.cur = (self.cur + 1) % len(self.keys)
                    time.sleep(1)

                elif self._is_invalid_error(err):
                    log.error(f"[Gemini] Key {self.cur+1} invalid/blocked: {str(e)[:80]}")
                    # Mark permanently cooling (24h) — bad key
                    self._set_cooldown(self.cur, 86400)
                    self.cur = (self.cur + 1) % len(self.keys)

                else:
                    log.error(f"[Gemini] Key {self.cur+1} unknown error: {e}")
                    return "I encountered an issue generating a response. Please try again."

        log.error("[Gemini] All keys and models exhausted")
        return (
            "I'm having trouble connecting right now. "
            "Please check your API keys in the .env file or try again in a minute. 🙏"
        )

    def generate_with_image(self, prompt: str, image_b64: str,
                             mime_type: str = "image/png", max_tokens: int = 4096) -> str:
        """Multimodal — send image + prompt to Gemini Vision."""
        if not self.keys:
            return ""
        for attempt in range(len(self.keys)):
            if not self._select_key():
                break
            try:
                genai.configure(api_key=self.keys[self.cur])
                model = genai.GenerativeModel(self.MODEL)
                resp = model.generate_content(
                    [{"inline_data": {"mime_type": mime_type, "data": image_b64}}, prompt],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_tokens, temperature=0.2)
                )
                self._key_state[self.cur]["requests"] += 1
                return resp.text or ""
            except Exception as e:
                err = str(e).lower()
                if self._is_quota_error(err) or self._is_rate_error(err):
                    self._set_cooldown(self.cur, RATE_COOLDOWN_SECS)
                    self.cur = (self.cur + 1) % len(self.keys)
                else:
                    log.error(f"[Gemini] Vision error: {e}")
                    return ""
        return ""

    def generate_with_pdf(self, prompt: str, pdf_b64: str, max_tokens: int = 4096) -> str:
        """Send PDF bytes directly to Gemini for content extraction."""
        if not self.keys:
            return ""
        for attempt in range(len(self.keys)):
            if not self._select_key():
                break
            try:
                genai.configure(api_key=self.keys[self.cur])
                model = genai.GenerativeModel(self.MODEL)
                resp = model.generate_content(
                    [{"inline_data": {"mime_type": "application/pdf", "data": pdf_b64}}, prompt],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=max_tokens, temperature=0.2)
                )
                self._key_state[self.cur]["requests"] += 1
                return resp.text or ""
            except Exception as e:
                err = str(e).lower()
                if self._is_quota_error(err) or self._is_rate_error(err):
                    self._set_cooldown(self.cur, RATE_COOLDOWN_SECS)
                    self.cur = (self.cur + 1) % len(self.keys)
                else:
                    log.error(f"[Gemini] PDF Vision error: {e}")
                    return ""
        return ""

    def embed(self, text: str) -> list:
        if not self.keys: return []
        for _ in range(len(self.keys)):
            if not self._select_key(): break
            try:
                r = genai.embed_content(
                    model="models/embedding-001",
                    content=text, task_type="retrieval_document")
                return r["embedding"]
            except Exception:
                self._set_cooldown(self.cur, RATE_COOLDOWN_SECS)
                self.cur = (self.cur + 1) % len(self.keys)
        return []

    def embed_query(self, text: str) -> list:
        if not self.keys: return []
        for _ in range(len(self.keys)):
            if not self._select_key(): break
            try:
                r = genai.embed_content(
                    model="models/embedding-001",
                    content=text, task_type="retrieval_query")
                return r["embedding"]
            except Exception:
                self._set_cooldown(self.cur, RATE_COOLDOWN_SECS)
                self.cur = (self.cur + 1) % len(self.keys)
        return []

    def health_check(self) -> dict:
        """Quick health check — tries a minimal prompt on each available key."""
        results = {}
        now = time.time()
        for i, key in enumerate(self.keys):
            if self._key_state[i]["cooldown_until"] > now:
                remaining = round(self._key_state[i]["cooldown_until"] - now)
                results[f"key_{i+1}"] = {"status": "cooling", "cooldown_remaining": remaining}
                continue
            try:
                genai.configure(api_key=key)
                m = genai.GenerativeModel("gemini-2.0-flash-lite")
                resp = m.generate_content("Reply with OK",
                    generation_config=genai.GenerationConfig(max_output_tokens=5))
                results[f"key_{i+1}"] = {"status": "ok", "response": resp.text[:20]}
            except Exception as e:
                err = str(e).lower()
                results[f"key_{i+1}"] = {
                    "status": "error",
                    "type": "quota" if self._is_quota_error(err) else
                            "rate"  if self._is_rate_error(err) else
                            "invalid" if self._is_invalid_error(err) else "unknown",
                    "detail": str(e)[:120]
                }
        # Restore correct key config
        if self.keys:
            genai.configure(api_key=self.keys[self.cur])
        return results