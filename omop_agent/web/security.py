"""Request-level protections for the web app: CSRF tokens, login throttling, response headers."""

from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from ..fsutil import atomic_write_text

# Pages and assets are served only from this app; nothing is loaded from elsewhere.
APP_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)
# The inspection report is a self-contained file with inline CSS and no scripts.
REPORT_CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
BASE_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def load_secret_key(data_dir: Path, env_value: str | None = None) -> str:
    """The session-signing key: from the environment, else ``<data_dir>/secret_key`` (created once)."""
    if env_value:
        if len(env_value) < 32:
            raise ValueError("OMOP_AGENT_WEB_SECRET must be at least 32 characters")
        return env_value
    path = Path(data_dir) / "secret_key"
    if path.exists():
        return path.read_text().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(48)
    atomic_write_text(path, key + "\n", mode=0o600)
    return key


def csrf_token(session: dict) -> str:
    token = session.get("csrf")
    if not token:
        token = session["csrf"] = secrets.token_urlsafe(32)
    return token


def csrf_ok(session: dict, submitted: str | None) -> bool:
    expected = session.get("csrf")
    return bool(expected and submitted and hmac.compare_digest(str(expected), str(submitted)))


class LoginThrottle:
    """Locks sign-in out for a while after repeated failures.

    Keys are ``kind:value``; each kind has its own limit within the window:

    * ``pair`` (user name + client address): 5. Guessing one account from one place.
    * ``user``: 20, from anywhere. Slows guessing one account from many places, without letting
      a few typos from someone else's machine lock its owner out.
    * ``ip``: 30, for any names. Slows one machine trying many accounts.
    """

    LIMITS = {"pair": 5, "user": 20, "ip": 30}

    def __init__(self, window_s: float = 900, lockout_s: float = 900, limits: dict[str, int] | None = None):
        self.window_s, self.lockout_s = window_s, lockout_s
        self.limits = {**self.LIMITS, **(limits or {})}
        self._fails: dict[str, deque[float]] = defaultdict(deque)
        self._locked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def keys(user: str, ip: str) -> tuple[str, str, str]:
        return (f"pair:{user}|{ip}", f"user:{user}", f"ip:{ip}")

    def blocked(self, *keys: str) -> bool:
        now = time.monotonic()
        with self._lock:
            return any(self._locked_until.get(k, 0) > now for k in keys)

    def failure(self, *keys: str) -> None:
        now = time.monotonic()
        with self._lock:
            for k in keys:
                q = self._fails[k]
                q.append(now)
                while q and q[0] < now - self.window_s:
                    q.popleft()
                if len(q) >= self.limits.get(k.split(":", 1)[0], 5):
                    self._locked_until[k] = now + self.lockout_s
                    q.clear()
            if len(self._fails) > 10_000:  # forget stale entries rather than grow without bound
                for k in [k for k, q in self._fails.items() if not q or q[-1] < now - self.window_s]:
                    del self._fails[k]

    def success(self, *keys: str) -> None:
        with self._lock:
            for k in keys:
                if k.startswith("pair:"):  # a success clears only this user's own record
                    self._fails.pop(k, None)
                    self._locked_until.pop(k, None)


def safe_next(target: str | None, default: str = "/") -> str:
    """Only same-site paths are allowed as a post-login redirect."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return default
