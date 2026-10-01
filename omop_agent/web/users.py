"""Local accounts for the review web app.

Accounts live in ``<data_dir>/users.json`` (readable only by the app's own user) and are
managed with ``omop-agent users ...`` on the server, so whoever can run commands there
administers the app. Passwords are stored as scrypt hashes, never in clear. Roles, each
including the ones before it:

* ``viewer``: reads projects, ETL specs, mappings, run logs and reports;
* ``reviewer``: also edits mappings and approves the review gates;
* ``operator``: also starts and stops pipeline runs.

Every change to an account (password, role, disabling it) bumps its ``epoch``, which signs
out the sessions it already has.

This module uses only the standard library so ``omop-agent users`` works without the web
extras installed.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..fsutil import atomic_write_text, file_lock
from ..workspace import utcnow

ROLES = ("viewer", "reviewer", "operator")
MIN_PASSWORD_LENGTH = 10
USERNAME = re.compile(r"^[a-z0-9][a-z0-9._-]{1,39}$")
_SCRYPT = {"n": 2**14, "r": 8, "p": 1}


class UserError(ValueError):
    """An account change that cannot be made (unknown user, weak password, ...)."""


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **_SCRYPT)
    b64 = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return f"scrypt${_SCRYPT['n']}${_SCRYPT['r']}${_SCRYPT['p']}${b64(salt)}${b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        got = hashlib.scrypt(
            password.encode(), salt=base64.b64decode(salt), dklen=len(expected), n=int(n), r=int(r), p=int(p)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)


_DUMMY_HASH = hash_password("not a real password: used to time unknown users")


@dataclass
class User:
    name: str
    role: str
    password: str = ""
    full_name: str = ""
    active: bool = True
    epoch: int = 0
    created_at: str = ""

    def can(self, role: str) -> bool:
        """True when this account is active and has ``role`` or a higher one."""
        return self.active and ROLES.index(self.role) >= ROLES.index(role)

    @property
    def display(self) -> str:
        return self.full_name or self.name


def check_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise UserError(f"passwords need at least {MIN_PASSWORD_LENGTH} characters")


class UserStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.path = self.data_dir / "users.json"

    # ---- storage -----------------------------------------------------------------------
    def _load(self) -> dict[str, User]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text() or "{}")
        return {name: User(name=name, **{k: v for k, v in u.items() if k != "name"}) for name, u in raw.items()}

    def _save(self, users: dict[str, User]) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {name: {k: v for k, v in asdict(u).items() if k != "name"} for name, u in sorted(users.items())}
        atomic_write_text(self.path, json.dumps(data, indent=2) + "\n", mode=0o600)

    def _change(self, name: str, fn) -> User:
        with file_lock(self.data_dir / ".users.lock"):
            users = self._load()
            user = users.get(name)
            if user is None:
                raise UserError(f"no user named '{name}'")
            fn(user)
            user.epoch += 1
            self._save(users)
            return user

    # ---- queries -----------------------------------------------------------------------
    def get(self, name: str) -> User | None:
        return self._load().get(name)

    def list(self) -> list[User]:
        return sorted(self._load().values(), key=lambda u: u.name)

    def authenticate(self, name: str, password: str) -> User | None:
        """The active user with this name and password, else None (in about the same time)."""
        user = self.get(name.strip().lower())
        ok = verify_password(password, user.password if user else _DUMMY_HASH)
        return user if (user is not None and ok and user.active) else None

    # ---- changes -----------------------------------------------------------------------
    def add(self, name: str, password: str, role: str, full_name: str = "") -> User:
        name = name.strip().lower()
        if not USERNAME.match(name):
            raise UserError("user names are 2-40 characters: lower-case letters, digits, '.', '_' or '-'")
        if role not in ROLES:
            raise UserError(f"role must be one of: {', '.join(ROLES)}")
        check_password(password)
        with file_lock(self.data_dir / ".users.lock"):
            users = self._load()
            if name in users:
                raise UserError(f"user '{name}' already exists")
            user = User(
                name=name, role=role, password=hash_password(password), full_name=full_name.strip(),
                created_at=utcnow().isoformat(timespec="seconds"),
            )
            users[name] = user
            self._save(users)
            return user

    def set_password(self, name: str, password: str) -> User:
        check_password(password)
        return self._change(name, lambda u: setattr(u, "password", hash_password(password)))

    def set_role(self, name: str, role: str) -> User:
        if role not in ROLES:
            raise UserError(f"role must be one of: {', '.join(ROLES)}")
        return self._change(name, lambda u: setattr(u, "role", role))

    def set_active(self, name: str, active: bool) -> User:
        return self._change(name, lambda u: setattr(u, "active", active))

    def sign_out_everywhere(self, name: str) -> User:
        return self._change(name, lambda u: None)
