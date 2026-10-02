"""The network trust boundary's own policy functions: the authority parser, the arrival, the Host
and the Origin rule against a constructed ASGI scope. The boundary as the app applies it is
covered in test_trust_boundary_routes.py.
"""

from __future__ import annotations

import pytest

from tcip_web.trust_boundary import (
    canonical_host,
    host_allowed,
    is_loopback_host,
    local_arrival,
    origin_allowed,
    parse_authority,
)


def _scope(server: tuple[str, int | None] | None, host: str | None, scheme: str = "http") -> dict:
    headers = [(b"host", host.encode())] if host is not None else []
    return {"type": "websocket" if scheme in ("ws", "wss") else "http", "scheme": scheme,
            "server": list(server) if server else None, "headers": headers}


def test_the_authority_parser_canonicalises_every_spelling() -> None:
    assert canonical_host("ORCHARD-PC.") == "orchard-pc"
    assert canonical_host("[::ffff:127.0.0.1]") == "127.0.0.1"
    assert parse_authority("[::1]:8765", 80) == ("::1", 8765)
    assert parse_authority("192.168.1.23", 80) == ("192.168.1.23", 80)
    for bad in ("a:b:c", "host:0", "host:abc", "[::1", "host/x", "host?x", "user@host"):
        with pytest.raises(ValueError):
            parse_authority(bad, 80)


def test_an_arrival_is_local_or_not() -> None:
    assert local_arrival(_scope(("127.0.0.1", 8765), None))
    assert local_arrival(_scope(("::ffff:127.0.0.1", 8765), None))
    assert local_arrival(_scope(("localhost", 80), None))
    assert local_arrival(_scope(("/tmp/tcip.sock", None), None))
    assert not local_arrival(_scope(("192.168.1.23", 8765), None))
    for unclassifiable in (("testserver", 80), ("", None), None):
        assert not local_arrival(_scope(unclassifiable, None))
    assert is_loopback_host("[::ffff:127.0.0.1]") and not is_loopback_host("0.0.0.0")


def test_the_host_must_be_a_loopback_name_at_the_arrival_port() -> None:
    assert host_allowed(_scope(("127.0.0.1", 8765), "127.0.0.1:8765"))
    assert host_allowed(_scope(("127.0.0.1", 8765), "localhost:8765"))
    assert not host_allowed(_scope(("127.0.0.1", 8765), "localhost:9999"))
    assert not host_allowed(_scope(("127.0.0.1", 8765), "orchard-pc:8765"))
    assert not host_allowed(_scope(("127.0.0.1", 8765), None))


def test_a_loopback_origin_at_any_port_is_served_on_a_local_arrival() -> None:
    scope = _scope(("127.0.0.1", 8765), "127.0.0.1:8765", scheme="ws")
    assert origin_allowed("http://localhost:5173", scope)
    assert origin_allowed("ws://127.0.0.1:8765", scope)
    assert origin_allowed(None, scope)
    assert not origin_allowed("http://evil.example.com", scope)
    assert not origin_allowed("null", scope)
    assert not origin_allowed("http://attacker.invalid@127.0.0.1:8765/x", scope)
    assert not origin_allowed("http://192.168.1.23:8765", scope)
