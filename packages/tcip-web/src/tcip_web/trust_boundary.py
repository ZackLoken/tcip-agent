"""The network trust boundary: the backend serves connections that arrived through this machine,
and answers only to loopback names.

Locality is a property of the accepted connection, never of a configured bind host: the ASGI
``scope["server"]`` is the local address the connection arrived on, and a connection through
anything but a loopback address or a UNIX socket is refused. One canonical authority parser serves
the arrival, the Host header and the Origin header.
"""

from __future__ import annotations

import ipaddress
import logging
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from typing import Any
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Mapping[str, Any]]]
Send = Callable[[Mapping[str, Any]], Awaitable[None]]

Authority = tuple[str, int | None]
"""A canonical host and a port; ``None`` stands for the arrival's own port."""

_DEFAULT_PORTS = {"http": 80, "https": 443}
_LOOPBACK_NAMES = frozenset({"localhost"})

STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
"""HTTP methods the trust boundary treats as mutating: a request using one of these must carry
an allowed Origin, the same requirement every WebSocket scope carries regardless of method."""

_ORIGIN_REFUSAL_WS = "origin not allowed"
_ORIGIN_REFUSAL_HTTP = (
    f"{_ORIGIN_REFUSAL_WS}: a state-changing request from another origin is refused."
)

EXPOSURE_REFUSAL = (
    "this connection arrived through a network address: the backend serves this machine only, "
    "since it hands an unauthenticated client filesystem reads and writes and an interactive "
    "agent terminal, which is keyboard access to a coding agent."
)


def canonical_host(host: str) -> str:
    """The canonical spelling of a host: lower-cased, no trailing dot, IPv6 unbracketed, an
    IPv4-mapped IPv6 address unwrapped to its IPv4 form. Raises ``ValueError`` for anything that
    is not a host: empty, whitespace or control characters, userinfo, a backslash."""
    h = host.strip()
    if not h or any(c.isspace() or ord(c) < 32 for c in h) or "@" in h or "\\" in h:
        raise ValueError(f"not a host: {host!r}")
    h = h.lower().rstrip(".")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return h
    mapped = getattr(ip, "ipv4_mapped", None)
    return str(mapped or ip)


def parse_authority(value: str, default_port: int | None) -> Authority:
    """``host[:port]`` into a canonical authority; a missing port takes ``default_port``.

    Raises ``ValueError`` for a malformed value: a scheme, a path, a query, a fragment, userinfo,
    a non-numeric or out-of-range port, or mismatched IPv6 brackets.
    """
    v = value.strip()
    if not v or "/" in v or "?" in v or "#" in v:
        raise ValueError(f"not an authority: {value!r}")
    if v.startswith("["):
        end = v.find("]")
        if end < 0:
            raise ValueError(f"not an authority: {value!r}")
        host, rest = v[: end + 1], v[end + 1:]
        if rest and not rest.startswith(":"):
            raise ValueError(f"not an authority: {value!r}")
        port_text = rest[1:] if rest else ""
    else:
        host, sep, port_text = v.rpartition(":")
        if not sep:
            host, port_text = v, ""
        elif ":" in host:
            raise ValueError(f"not an authority: {value!r}")
    port: int | None
    if port_text:
        if not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
            raise ValueError(f"not an authority: {value!r}")
        port = int(port_text)
    else:
        port = default_port
    return canonical_host(host), port


def is_loopback_host(host: str) -> bool:
    """True if ``host`` names only the local machine (127.0.0.0/8, ::1, localhost).

    ``0.0.0.0`` / ``::`` mean "all interfaces" and are therefore not loopback: binding
    them exposes the server to the network.
    """
    try:
        h = canonical_host(host)
    except ValueError:
        return False
    if h in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def arrival(scope: Mapping[str, Any]) -> tuple[str, int | None] | None:
    """The local address a connection was accepted on, or None when the scope carries none."""
    server = scope.get("server")
    if not server or not server[0]:
        return None
    return str(server[0]), server[1]


def local_arrival(scope: Mapping[str, Any]) -> bool:
    """True when the connection arrived through this machine: a loopback address or name, or a
    UNIX socket path."""
    at = arrival(scope)
    if at is None:
        return False
    host = at[0]
    return is_loopback_host(host) or "/" in host or "\\" in host


def _http_scheme(scheme: str) -> str:
    """``scheme`` with a WebSocket scheme named by its HTTP counterpart."""
    return {"ws": "http", "wss": "https"}.get(scheme, scheme)


def _request_scheme(scope: Mapping[str, Any]) -> str:
    return _http_scheme(str(scope.get("scheme") or "http"))


def _header_values(scope: Mapping[str, Any], name: bytes) -> list[str]:
    return [v.decode("latin-1") for k, v in scope.get("headers") or () if k.lower() == name]


def request_authority(scope: Mapping[str, Any]) -> Authority | None:
    """The Host header as a canonical authority, or None when it is absent, duplicated or
    malformed."""
    values = _header_values(scope, b"host")
    if len(values) != 1:
        return None
    try:
        return parse_authority(values[0], _DEFAULT_PORTS.get(_request_scheme(scope)))
    except ValueError:
        return None


def host_allowed(scope: Mapping[str, Any]) -> bool:
    """Whether the request's Host is a loopback name at the port the connection arrived on."""
    at = arrival(scope)
    authority = request_authority(scope)
    if at is None or authority is None:
        return False
    return is_loopback_host(authority[0]) and (at[1] is None or authority[1] == at[1])


def _parse_origin(origin: str) -> tuple[str, Authority] | None:
    parts = urlsplit(origin.strip())
    if parts.scheme not in ("http", "https", "ws", "wss") or not parts.netloc:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment or "@" in parts.netloc:
        return None
    scheme = _http_scheme(parts.scheme)
    try:
        return scheme, parse_authority(parts.netloc, _DEFAULT_PORTS[scheme])
    except ValueError:
        return None


def origin_allowed(origin: str | None, scope: Mapping[str, Any]) -> bool:
    """Whether an Origin is one this backend serves.

    Only an absent Origin (``None``) is a non-browser client and is allowed. A present Origin,
    empty included, is refused unless it parses as a bare authority naming a loopback host, at any
    port.
    """
    if origin is None:
        return True
    parsed = _parse_origin(origin)
    return parsed is not None and is_loopback_host(parsed[1][0])


class TrustBoundaryMiddleware:
    """Refuse connections the backend must not serve, before any route runs.

    Applies to ``http`` and ``websocket`` scopes only; a ``lifespan`` scope carries no arrival. An
    arrival that is not :func:`local_arrival` is refused with the exposure message; a Host the
    backend does not answer to is refused as an invalid host. After the Host check, every
    WebSocket scope and every ``http`` scope whose method is in :data:`STATE_CHANGING_METHODS` must
    also carry an Origin :func:`origin_allowed` admits; a duplicated Origin header is refused the
    same way a duplicated Host is. A refused arrival is logged once per client and arrival address
    pair.
    """

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app
        self._logged: set[str] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not local_arrival(scope):
            self._log_refusal(scope)
            await _refuse(scope, send, 403, EXPOSURE_REFUSAL)
            return
        if not host_allowed(scope):
            await _refuse(scope, send, 400, "invalid host header")
            return
        if scope["type"] == "websocket" or scope.get("method") in STATE_CHANGING_METHODS:
            if not self._origin_ok(scope):
                detail = _ORIGIN_REFUSAL_WS if scope["type"] == "websocket" else _ORIGIN_REFUSAL_HTTP
                await _refuse(scope, send, 403, detail)
                return
        await self.app(scope, receive, send)

    def _origin_ok(self, scope: Mapping[str, Any]) -> bool:
        """Whether this scope's Origin header, if any, is one :func:`origin_allowed` admits; a
        duplicated Origin is refused.
        """
        values = _header_values(scope, b"origin")
        if len(values) > 1:
            return False
        return origin_allowed(values[0] if values else None, scope)

    def _log_refusal(self, scope: Mapping[str, Any]) -> None:
        client = scope.get("client")
        at = arrival(scope)
        key = f"{client[0] if client else 'unknown client'} via {at[0] if at else 'no address'}"
        if key not in self._logged:
            self._logged.add(key)
            logger.warning("refused %s: %s", key, EXPOSURE_REFUSAL)


async def _refuse(scope: Mapping[str, Any], send: Send, status: int, detail: str) -> None:
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008, "reason": detail[:120]})
        return
    body = detail.encode("utf-8")
    await send({
        "type": "http.response.start",
        "status": status,
        "headers": [(b"content-type", b"text/plain; charset=utf-8"),
                    (b"content-length", str(len(body)).encode("ascii"))],
    })
    await send({"type": "http.response.body", "body": body})
