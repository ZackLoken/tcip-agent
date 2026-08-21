"""The network trust boundary's own policy functions: the authority parser, the advertised list,
and the Origin rule against a constructed ASGI scope. The boundary as the app applies it is
covered in test_trust_boundary_routes.py.
"""

from __future__ import annotations

import pytest

from tcip_web.trust_boundary import (
    advertised_authorities,
    canonical_host,
    exposed_arrival,
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


def test_an_arrival_is_local_exposed_or_neither() -> None:
    assert local_arrival(_scope(("127.0.0.1", 8765), None))
    assert local_arrival(_scope(("::ffff:127.0.0.1", 8765), None))
    assert local_arrival(_scope(("localhost", 80), None))
    assert local_arrival(_scope(("/tmp/tcip.sock", None), None))
    assert exposed_arrival(_scope(("192.168.1.23", 8765), None))
    assert not local_arrival(_scope(("192.168.1.23", 8765), None))
    for unclassifiable in (("testserver", 80), ("", None), None):
        assert not local_arrival(_scope(unclassifiable, None))
        assert not exposed_arrival(_scope(unclassifiable, None))
    assert is_loopback_host("[::ffff:127.0.0.1]") and not is_loopback_host("0.0.0.0")


def test_a_wildcard_or_malformed_advertised_entry_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for bad in ("*", "http://gui.example", "gui.example/path", "user@gui.example", "gui:99999"):
        monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", bad)
        with pytest.raises(ValueError, match="TCIP_WEB_ADVERTISED_HOSTS"):
            advertised_authorities()
    monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", "Gui.Example.:443,[::1]:8765")
    assert advertised_authorities() == [("gui.example", 443), ("::1", 8765)]


def test_a_loopback_origin_at_any_port_is_served_on_a_local_arrival() -> None:
    scope = _scope(("127.0.0.1", 8765), "127.0.0.1:8765", scheme="ws")
    assert origin_allowed("http://localhost:5173", scope)
    assert origin_allowed("ws://127.0.0.1:8765", scope)
    assert origin_allowed(None, scope)
    assert not origin_allowed("http://evil.example.com", scope)
    assert not origin_allowed("null", scope)
    assert not origin_allowed("http://attacker.invalid@127.0.0.1:8765/x", scope)


def test_the_origin_policy_reads_the_arrival_not_an_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TCIP_WEB_ALLOW_INSECURE", "1")
    exposed = _scope(("192.168.1.23", 8765), "192.168.1.23:8765", scheme="ws")
    assert origin_allowed("http://192.168.1.23:8765", exposed)
    assert not origin_allowed("http://localhost:5173", exposed)
    assert not origin_allowed("http://192.168.1.23:3000", exposed)
    monkeypatch.setenv("TCIP_WEB_ADVERTISED_HOSTS", "gui.example:443")
    assert origin_allowed("https://gui.example", exposed)
    monkeypatch.delenv("TCIP_WEB_ALLOW_INSECURE", raising=False)
    assert not origin_allowed("https://gui.example", exposed)
