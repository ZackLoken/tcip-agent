"""The network trust boundary: which connections the backend serves and which names it answers to.

Exposure is a property of the accepted connection, never of a configured bind host. The ASGI
``scope["server"]`` is the local address the connection arrived on (uvicorn fills it from the
accepted socket's own ``getsockname``), so a connection through a loopback address is local, one
through a routable address is exposed, and one the backend cannot classify is refused. An exposed
arrival is served only when the operator has opted in with ``TCIP_WEB_ALLOW_INSECURE=1``, because
an exposed GUI hands an unauthenticated network client filesystem reads and writes and the
interactive agent terminal.

One canonical authority parser serves the arrival, the Host header, the Origin header and the
operator's advertised list (``TCIP_WEB_ADVERTISED_HOSTS``, comma-separated ``host[:port]``
entries for a name clients reach this machine by that it does not know itself, such as a DNS
alias or a same-machine reverse proxy). Advertising a name declares that network clients reach
this backend, so the list is consulted only under the opt-in. There is no wildcard anywhere.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from functools import cache
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
_ADVERTISED_ENV = "TCIP_WEB_ADVERTISED_HOSTS"
_OPT_IN_ENV = "TCIP_WEB_ALLOW_INSECURE"

STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
"""HTTP methods the trust boundary treats as mutating: a request using one of these must carry
an allowed Origin, the same requirement every WebSocket scope carries regardless of method."""

_ORIGIN_REFUSAL_WS = "origin not allowed"
_ORIGIN_REFUSAL_HTTP = (
    f"{_ORIGIN_REFUSAL_WS}: a state-changing request from another origin is refused."
)

EXPOSURE_REFUSAL = (
    "this connection arrived through a network address and the backend is not opted into network "
    "exposure: an exposed GUI hands an unauthenticated network client filesystem reads and writes "
    "and an interactive agent terminal, which is keyboard access to Claude Code. Set "
    f"{_OPT_IN_ENV}=1 only on a trusted network."
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


def _is_routable_ip(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(canonical_host(host))
    except ValueError:
        return False
    return not ip.is_loopback and not ip.is_unspecified


def insecure_opt_in() -> bool:
    """Whether the operator has opted into serving network clients with no authentication."""
    return os.environ.get(_OPT_IN_ENV) == "1"


def advertised_authorities() -> list[Authority]:
    """The operator's advertised authorities, validated; ``ValueError`` names a bad entry.

    An entry with no port stands for the arrival's own port. A wildcard is refused by name.
    """
    raw = os.environ.get(_ADVERTISED_ENV, "")
    out: list[Authority] = []
    for entry in (e for e in raw.split(",") if e.strip()):
        if "*" in entry:
            raise ValueError(f"{_ADVERTISED_ENV} entry {entry!r} is a wildcard; advertise names")
        try:
            out.append(parse_authority(entry, None))
        except ValueError as exc:
            raise ValueError(f"{_ADVERTISED_ENV} entry {entry!r} is not a host[:port]") from exc
    return out


@cache
def _own_names() -> frozenset[str]:
    """The machine's own names, resolved once per process; addresses come from the arrival."""
    names = set()
    for getter in (socket.gethostname, socket.getfqdn):
        try:
            names.add(canonical_host(getter()))
        except (OSError, ValueError):
            continue
    return frozenset(names)


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


def exposed_arrival(scope: Mapping[str, Any]) -> bool:
    """True when the connection arrived through a routable address."""
    at = arrival(scope)
    return at is not None and _is_routable_ip(at[0])


def _request_scheme(scope: Mapping[str, Any]) -> str:
    scheme = str(scope.get("scheme") or "http")
    return {"ws": "http", "wss": "https"}.get(scheme, scheme)


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


def _advertised_match(authority: Authority, arrival_port: int | None) -> bool:
    if not insecure_opt_in():
        return False
    for host, port in advertised_authorities():
        if host == authority[0] and (port if port is not None else arrival_port) == authority[1]:
            return True
    return False


def host_allowed(scope: Mapping[str, Any]) -> bool:
    """Whether the request's Host names this backend as reached through its arrival.

    The authority must equal the arrival's address and port, or be a loopback name at the
    arrival port on a local arrival, the machine's own name at the arrival port, or an advertised
    authority under the opt-in.
    """
    at = arrival(scope)
    authority = request_authority(scope)
    if at is None or authority is None:
        return False
    host, port = authority
    arrival_host, arrival_port = at
    try:
        arrival_canonical = canonical_host(arrival_host)
    except ValueError:
        return False
    port_matches = arrival_port is None or port == arrival_port
    if host == arrival_canonical and port_matches:
        return True
    if local_arrival(scope) and is_loopback_host(host) and port_matches:
        return True
    if host in _own_names() and port_matches:
        return True
    return _advertised_match(authority, arrival_port)


def _parse_origin(origin: str) -> tuple[str, Authority] | None:
    parts = urlsplit(origin.strip())
    if parts.scheme not in ("http", "https", "ws", "wss") or not parts.netloc:
        return None
    if parts.path not in ("", "/") or parts.query or parts.fragment or "@" in parts.netloc:
        return None
    scheme = {"ws": "http", "wss": "https"}.get(parts.scheme, parts.scheme)
    try:
        return scheme, parse_authority(parts.netloc, _DEFAULT_PORTS[scheme])
    except ValueError:
        return None


def origin_allowed(origin: str | None, scope: Mapping[str, Any]) -> bool:
    """Whether an Origin is one this backend serves for the connection it arrived on.

    Applied by :class:`TrustBoundaryMiddleware` to every WebSocket scope and to an ``http``
    scope whose method is in :data:`STATE_CHANGING_METHODS`. Only an absent Origin (``None``)
    is a non-browser client (the MCP tools send none) and is allowed: this check is a
    browser-side mitigation against a cross-site page reading GUI state or driving a mutation,
    not authentication. A present Origin, empty included, is checked like any other and refused
    if it does not parse as a bare authority; this is a behaviour change on the socket path,
    whose handlers today read an empty header (``if not origin``) as missing. A present Origin
    must be exactly the request's own origin (the validated Host at the request scheme), or a
    loopback host at any port on a local arrival (the Vite dev server proxies from its own
    port), or an advertised authority under the opt-in.
    """
    if origin is None:
        return True
    parsed = _parse_origin(origin)
    if parsed is None:
        return False
    scheme, authority = parsed
    at = arrival(scope)
    request = request_authority(scope)
    if at is None or request is None:
        return False
    if scheme == _request_scheme(scope) and authority == request:
        return True
    if local_arrival(scope) and is_loopback_host(authority[0]):
        return True
    return _advertised_match(authority, at[1])


class TrustBoundaryMiddleware:
    """Refuse connections the backend must not serve, before any route runs.

    Applies to ``http`` and ``websocket`` scopes only; a ``lifespan`` scope carries no arrival.
    An arrival the backend cannot classify, and an exposed arrival without the opt-in, are refused
    with the exposure message; a Host the backend does not answer to is refused as an invalid
    host. After the Host check, every WebSocket scope and every ``http`` scope whose method is
    in :data:`STATE_CHANGING_METHODS` must also carry an Origin :func:`origin_allowed` admits; a
    duplicated Origin header is refused the same way a duplicated Host is. On a loopback
    arrival every loopback origin at every port is admitted, so another local server's page
    passes this check too; only the JSON-body guard's unanswered preflight
    (``routes/_body_common.py``) stops its browser from mutating. A refused exposed arrival is
    logged once per client and arrival address pair so the operator sees it.
    """

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self.app = app
        advertised_authorities()
        self._logged: set[str] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not local_arrival(scope) and not (exposed_arrival(scope) and insecure_opt_in()):
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
        """Whether this scope's Origin header, if any, is one :func:`origin_allowed` admits.

        A duplicated Origin header is refused outright, the same way a duplicated Host is
        (:func:`request_authority`), rather than resolved to either of its values.
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


def log_exposure_opt_in() -> None:
    """Name, at startup, what an opted-in exposed bind hands out; silent otherwise."""
    if insecure_opt_in():
        logger.warning(
            "%s=1: network clients are served with no authentication; they get filesystem reads "
            "and writes and the interactive agent terminal", _OPT_IN_ENV)
