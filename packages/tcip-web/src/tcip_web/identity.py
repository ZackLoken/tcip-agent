"""Current-user identity for provenance stamping (created_by / accepted_by).

The GUI is the source of truth: the annotator/reviewer sets their name in the app and it rides
along on each save/review request (``user`` field). When a request omits it (non-GUI callers,
older clients), we fall back to an env override then the OS login. ``user_id`` applies the
platform convention that humans are ``user:<name>`` while tool producers stay bare (``sam``,
``baseline``) — so a stamped GT object always says whether a person or a model authored it.
"""

from __future__ import annotations

import os


def current_user() -> str:
    """Backend fallback identity (bare name): ``TCIP_USER`` / ``TCIP_REVIEW_USER`` env, else OS login."""
    for key in ("TCIP_USER", "TCIP_REVIEW_USER"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    try:
        import getpass

        return getpass.getuser() or "gui"
    except Exception:
        return "gui"


def resolve_user(explicit: str | None) -> str:
    """The GUI-supplied ``user`` (bare name) when present, else the backend fallback."""
    val = (explicit or "").strip()
    return val if val else current_user()


def user_id(name: str | None) -> str:
    """A human ``created_by`` / ``accepted_by`` value: ``user:<name>`` (idempotent; never bare)."""
    val = (name or "").strip() or "gui"
    return val if val.startswith("user:") else f"user:{val}"
