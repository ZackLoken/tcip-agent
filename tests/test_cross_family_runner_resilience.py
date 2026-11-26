"""The cross-family runner keeps each family's answer when one harness returns nothing.

The runner poses one question to several agent harnesses concurrently and writes each one's
transcript. A harness that produces no stdout does not raise inside the worker.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import TYPE_CHECKING, NoReturn

import pytest

from tcip_web import terminal
from tests._scratch_repo_fixtures import commit_all, git, scratch_repo, write
from tests._scratch_server_fixtures import recording_launch
from tests._web_fixtures import backend_session
from tools import cross_family_ask

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_an_antigravity_launch_passes_no_effort_flag_and_records_the_ids_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An Antigravity launch names the model and no effort flag; the record holds the suffix.

    The run proceeds the same with no effort named and with the id's own effort named.
    """
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )

    for effort in (None, "low"):
        question_id = f"qid-{effort}"
        meta = cross_family_ask.run_one(
            question_id,
            "as-shipped",
            "question",
            tmp_path,
            tmp_path / "out",
            5,
            cross_family_ask.resolve_launch(
                "antigravity", "gemini-4-argon-low", effort
            ),
        )

        run_dir = tmp_path / "out" / question_id / "as-shipped" / "antigravity"
        argv = (run_dir / "argv.txt").read_text(encoding="utf-8").splitlines()
        assert "--effort" not in argv
        assert argv[argv.index("--model") + 1] == "gemini-4-argon-low"
        assert meta["effort_requested"] == "low"


def test_a_run_naming_no_model_launches_antigravity_on_the_parity_argon_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run naming neither model nor effort launches Antigravity on the parity Argon model."""
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )

    meta = cross_family_ask.run_one(
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        cross_family_ask.resolve_launch("antigravity", None, None),
    )

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "antigravity"
    argv = (run_dir / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert argv[argv.index("--model") + 1] == "gemini-4-argon-high"
    assert meta["effort_requested"] == "high"


def test_a_codex_run_naming_no_effort_passes_and_records_the_parity_effort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Codex run naming no model or effort runs on its config's model at the parity effort.

    Under the no-tools condition the harness ignores its own config, so the model and effort the
    run records are the ones its argv carries.
    """
    config = tmp_path / "config.toml"
    config.write_text('model = "gpt-from-config"' + chr(10), encoding="utf-8")
    monkeypatch.setattr(cross_family_ask, "CODEX_CONFIG", config)
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )

    meta = cross_family_ask.run_one(
        "qid",
        "no-tools",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        cross_family_ask.resolve_launch("codex", None, None),
    )

    run_dir = tmp_path / "out" / "qid" / "no-tools" / "codex"
    argv = (run_dir / "argv.txt").read_text(encoding="utf-8").splitlines()
    assert "--ignore-user-config" in argv
    assert argv[argv.index("--model") + 1] == "gpt-from-config"
    assert 'model_reasoning_effort="medium"' in argv
    assert meta["model_source"] == str(config)
    assert meta["effort_requested"] == "medium"


def test_an_effort_claude_takes_and_a_codex_effort_pass_through() -> None:
    assert (
        cross_family_ask.effective_effort("claude", "opus", "high") == "high"
    )
    assert (
        cross_family_ask.effective_effort("codex", "gpt-5.6-sol", "high")
        == "high"
    )


def _stub_run(
    stdout: str | bytes | None, stderr: str = ""
) -> Callable[..., subprocess.CompletedProcess[str | bytes | None]]:
    """Return a subprocess.run replacement returning the given streams verbatim."""

    def _run(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str | bytes | None]:
        return subprocess.CompletedProcess(
            args=["stub"], returncode=0, stdout=stdout, stderr=stderr
        )

    return _run


@pytest.mark.parametrize("stdout", [None, b"bytes not str"])
def test_a_harness_returning_no_usable_stdout_still_records_its_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes | None,
) -> None:
    """A harness returning None or bytes stdout gets its stdout and meta files written."""
    monkeypatch.setattr(cross_family_ask.subprocess, "run", _stub_run(stdout))
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    meta = cross_family_ask.run_one(
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        cross_family_ask.resolve_launch("codex", "stub-model", None),
    )

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "codex"
    assert (run_dir / "stdout.txt").is_file()
    assert (run_dir / "meta.json").is_file()
    assert meta["family"] == "codex"


def test_one_familys_failure_leaves_the_others_results_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising harness is recorded as a runner error while the surviving family's answer is kept.

    The run exits non-zero, the surviving family's response holds its answer, and the summary
    rows carry the survivor's clean exit and the raising family's error text.
    """

    def _explode(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("harness blew up")

    real_run_one = cross_family_ask.run_one

    def _dispatch(
        *args: Path | int | str | cross_family_ask.Launch | None,
        **kwargs: list[Path] | str | None,
    ) -> dict:
        launch = args[6]
        assert isinstance(launch, cross_family_ask.Launch)
        if launch.family == "codex":
            _explode()
        return real_run_one(*args, **kwargs)

    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    for fam in ("codex", "antigravity"):
        monkeypatch.setitem(
            cross_family_ask.BUILDERS,
            fam,
            lambda *a, **k: (["stub", "argv"], None),
        )
    monkeypatch.setattr(cross_family_ask, "run_one", _dispatch)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex,antigravity",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    # Non-zero, because a family that failed is a failed invocation and must not read as clean.
    assert cross_family_ask.main() != 0

    surviving = (
        tmp_path / "out" / "qid" / "as-shipped" / "antigravity" / "response.md"
    )
    assert surviving.is_file(), (
        "the family that answered lost its transcript to the one that failed"
    )
    assert surviving.read_text(encoding="utf-8").strip() == "an answer"

    summary = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    rows = {
        r["family"]: r
        for r in (summary if isinstance(summary, list) else summary["runs"])
    }
    assert rows["antigravity"]["exit_code"] == 0
    assert rows["codex"]["response_source"] == "runner_error"
    assert "harness blew up" in rows["codex"]["runner_error"]


def test_a_prompt_file_with_a_byte_order_mark_reaches_the_harness_without_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt file's byte order mark is absent from the question ``run_one`` receives."""
    seen: list[str] = []

    real_run_one = cross_family_ask.run_one

    def _capture(
        qid: str,
        condition: str,
        prompt: str,
        *args: Path | int | str | cross_family_ask.Launch | None,
        **kwargs: list[Path] | str | None,
    ) -> dict:
        seen.append(prompt)
        return real_run_one(qid, condition, prompt, *args, **kwargs)

    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    monkeypatch.setattr(cross_family_ask, "run_one", _capture)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8-sig")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() == 0
    assert len(seen) == 1
    assert seen[0] == "question"


def test_a_codex_agent_message_stream_yields_the_last_message() -> None:
    """A codex agent-message stream yields its last agent message as the answer.

    The stream carries its answers inside item.completed/agent_message events.
    """
    lines = [
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_0",
                    "type": "agent_message",
                    "text": "first pass",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_1",
                    "type": "command_execution",
                    "command": "ls",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        ),
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_2",
                    "type": "agent_message",
                    "text": "second and final pass",
                },
            }
        ),
        json.dumps({"type": "turn.completed"}),
    ]
    text, source = cross_family_ask.extract_response(
        "codex", "\n".join(lines), None
    )
    assert text == "second and final pass"
    assert source == "stream_agent_message"


def test_an_oversized_raw_stream_is_a_failed_extraction_not_an_answer() -> (
    None
):
    """An oversized raw stream reads as a failed extraction, not an answer.

    A stream past the size bound yields no text and the ``extraction_failed`` source.
    """
    huge = "not json, just noise " * 6000
    assert len(huge.strip()) > cross_family_ask.MAX_RAW_STREAM_CHARS
    text, source = cross_family_ask.extract_response("codex", huge, None)
    assert (text, source) == ("", "extraction_failed")


def test_an_oversized_raw_stream_through_main_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose oversized stream could not be extracted fails through main.

    Its meta records the ``extraction_failed`` response source.
    """
    huge = "not json, just noise " * 6000
    monkeypatch.setattr(cross_family_ask.subprocess, "run", _stub_run(huge))
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() != 0

    meta = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert meta["response_source"] == "extraction_failed"


def test_a_payload_with_only_empty_recognized_keys_reads_as_an_empty_response() -> (
    None
):
    """A payload whose only recognized keys are empty reads as an empty response.

    An antigravity SUCCESS payload whose ``response`` is blank yields no text and the
    ``empty_response`` source.
    """
    stdout = json.dumps(
        {
            "conversation_id": "cadb33f6-6b32-413f-9ddd-0b2c168c573b",
            "status": "SUCCESS",
            "response": "",
            "duration_seconds": 12.8666267,
            "num_turns": 1,
            "usage": {
                "input_tokens": 19088,
                "output_tokens": 891,
                "thinking_tokens": 661,
                "cache_read_tokens": 24842,
                "total_tokens": 19979,
            },
        }
    )
    text, source = cross_family_ask.extract_response(
        "antigravity", stdout, None
    )
    assert (text, source) == ("", "empty_response")


def test_an_empty_response_through_main_fails_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run whose recognized answer is empty fails through main."""
    stdout = json.dumps({"status": "SUCCESS", "response": "  "})
    monkeypatch.setattr(cross_family_ask.subprocess, "run", _stub_run(stdout))
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "antigravity",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "antigravity",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() != 0


def test_completely_empty_stdout_does_not_read_as_a_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completely empty stdout does not read as a clean run.

    A harness whose stdout is the empty string fails the run, and its meta records a fault under
    the ``raw_stdout`` response source.
    """
    monkeypatch.setattr(cross_family_ask.subprocess, "run", _stub_run(""))
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "antigravity",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "antigravity",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() != 0

    meta = json.loads(
        (
            tmp_path
            / "out"
            / "qid"
            / "as-shipped"
            / "antigravity"
            / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert meta["response_source"] == "raw_stdout"
    assert meta["fault"]


def test_a_whitespace_only_agent_message_stream_is_a_failed_run_not_a_clean_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex stream whose only agent message is whitespace fails the run.

    The run's meta records zero response characters and a fault.
    """
    lines = [
        json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "id": "item_0",
                    "type": "agent_message",
                    "text": "   \n  ",
                },
            }
        ),
        json.dumps({"type": "turn.completed"}),
    ]
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("\n".join(lines))
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() != 0

    meta = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert meta["response_source"] == "stream_agent_message"
    assert meta["response_chars"] == 0
    assert meta["fault"]


def test_a_runs_own_fault_verdict_is_recorded_in_its_meta_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run's own fault verdict is recorded in its meta.json.

    A clean run's ``fault`` verdict, empty, is in the meta it returns and in its meta.json on
    disk.
    """
    monkeypatch.setattr(
        cross_family_ask.subprocess,
        "run",
        _stub_run("a real answer from the harness"),
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    meta = cross_family_ask.run_one(
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        cross_family_ask.resolve_launch("codex", "stub-model", None),
    )
    assert meta["fault"] == ""

    on_disk = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert on_disk["fault"] == ""


def test_a_small_plain_text_stdout_still_returns_as_a_raw_stdout_answer_through_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short plain-text stdout returns as a full-length ``raw_stdout`` answer through main."""
    monkeypatch.setattr(
        cross_family_ask.subprocess,
        "run",
        _stub_run("a short plain-text answer"),
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() == 0

    meta = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert meta["response_source"] == "raw_stdout"
    assert meta["response_chars"] == len("a short plain-text answer")


def test_a_dict_payload_with_no_recognized_keys_still_returns_whole_payload() -> (
    None
):
    """A dict with unrecognized keys yields the ``whole_payload`` source.

    Its text names ``unexpected_field``.
    """
    stdout = json.dumps(
        {"unexpected_field": "some value", "another_field": 42}
    )
    text, source = cross_family_ask.extract_response(
        "antigravity", stdout, None
    )
    assert source == "whole_payload"
    assert "unexpected_field" in text


def test_a_non_string_answer_value_still_dumps_the_whole_payload() -> None:
    """A non-string value under a recognized key yields the ``whole_payload`` source.

    A content-block list and a null under a recognized key each yield that source.
    """
    stdout = json.dumps(
        {"content": [{"type": "text", "text": "the real answer"}]}
    )
    text, source = cross_family_ask.extract_response(
        "antigravity", stdout, None
    )
    assert source == "whole_payload"
    assert "the real answer" in text

    text, source = cross_family_ask.extract_response(
        "antigravity", json.dumps({"result": None}), None
    )
    assert source == "whole_payload"


def test_a_character_outside_the_console_codepage_survives_the_trip_to_a_stdin_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A box-drawing character reaches a stdin child as utf-8 and its stdout decodes intact."""
    echo_child = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write(sys.stdin.read())",
    ]
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS, "codex", lambda *a, **k: (echo_child, None)
    )

    prompt = "a box-drawing rule: ───"
    meta = cross_family_ask.run_one(
        "qid",
        "as-shipped",
        prompt,
        tmp_path,
        tmp_path / "out",
        30,
        cross_family_ask.resolve_launch("codex", "stub-model", None),
    )

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "codex"
    body = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    assert "─" in body
    assert (run_dir / "stdout.txt").read_text(encoding="utf-8") == body
    assert meta["exit_code"] == 0


# ── the codex model is named on every run ───────────────────────────────────


def test_the_codex_model_is_read_from_the_config_and_its_source_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.toml"
    config.write_text('model = "gpt-from-config"' + chr(10), encoding="utf-8")
    monkeypatch.setattr(cross_family_ask, "CODEX_CONFIG", config)

    assert cross_family_ask.resolve_codex_model(None) == (
        "gpt-from-config",
        str(config),
    )
    assert cross_family_ask.resolve_codex_model("gpt-from-flag") == (
        "gpt-from-flag",
        "flag",
    )


def test_a_run_with_no_usable_answer_says_so_in_words_at_the_end_of_its_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run with no usable answer prints a RUN FAILED verdict naming the family, in words.

    The verdict appears on stdout and stderr beside the non-zero exit, and the output's last line
    says nothing is reviewable.
    """
    monkeypatch.setattr(
        cross_family_ask.subprocess,
        "run",
        _stub_run(json.dumps({"status": "SUCCESS", "response": "  "})),
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "antigravity",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "antigravity",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() != 0
    captured = capsys.readouterr()
    assert "RUN FAILED" in captured.out
    assert "antigravity" in captured.out.split("RUN FAILED")[1]
    assert "no answer text" in captured.out
    # Last, so a tail of the output cannot cut the verdict off in favor of a "wrote ..." line.
    assert (
        captured.out.rstrip()
        .splitlines()[-1]
        .strip()
        .startswith("Nothing here is reviewable")
    )
    assert "RUN FAILED" in captured.err


# ── the tcip MCP block is read from .mcp.json, not restated ─────────────────


def test_load_tcip_mcp_config_refuses_when_mcp_json_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cross_family_ask, "REPO_ROOT", tmp_path)
    with pytest.raises(
        SystemExit, match="cannot read the tcip MCP server block"
    ):
        cross_family_ask.load_tcip_mcp_config()


def test_load_tcip_mcp_config_refuses_when_the_tcip_block_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"claude-context": {}}}), encoding="utf-8"
    )
    monkeypatch.setattr(cross_family_ask, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no mcpServers.tcip block"):
        cross_family_ask.load_tcip_mcp_config()


def test_a_run_whose_families_all_answered_says_so_rather_than_staying_silent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A run whose families all answered prints a RUN OK verdict and no RUN FAILED."""
    monkeypatch.setattr(
        cross_family_ask.subprocess,
        "run",
        _stub_run("a real answer from the harness"),
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "antigravity",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "antigravity",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() == 0
    captured = capsys.readouterr()
    assert "RUN OK: 1 of 1 families answered." in captured.out
    assert "RUN FAILED" not in captured.out
    assert "RUN FAILED" not in captured.err


def test_an_attached_image_reaches_codex_as_a_flag_and_the_other_families_in_the_prompt(
    tmp_path: Path,
) -> None:
    """An attached image reaches codex as a flag and antigravity and Claude Code in the prompt.

    Codex takes the image through ``-i``, the flag followed by the next option and the trailing
    stdin marker kept; antigravity takes an @path in the prompt; Claude Code takes a named path
    in the prompt. A run with no image carries no image flag and no preface.
    """
    image = tmp_path / "render.png"
    image.write_bytes(b"png")
    condition = cross_family_ask.CONDITIONS["no-tools"]
    prompt_file = tmp_path / "p.txt"

    argv, _ = cross_family_ask.build_codex(
        prompt_file,
        tmp_path,
        tmp_path,
        condition,
        "gpt-5",
        "high",
        5,
        images=(image,),
    )
    at = argv.index("-i")
    assert argv[at + 1] == str(image)
    assert argv[at + 2].startswith("--")
    assert argv[-1] == "-"
    bare, _ = cross_family_ask.build_codex(
        prompt_file, tmp_path, tmp_path, condition, "gpt-5", "high", 5
    )
    assert "-i" not in bare

    assert f"@{image}" in cross_family_ask.image_preface(
        "antigravity", (image,)
    )
    assert str(image) in cross_family_ask.image_preface("claude", (image,))
    assert str(image) in cross_family_ask.image_preface("codex", (image,))
    assert cross_family_ask.image_preface("claude", ()) == ""


def test_an_attached_image_prefaces_the_prompt_and_is_recorded_in_the_runs_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attached image prefaces the prompt and is recorded in the run's meta.json.

    The preface sits ahead of the question, the builder receives the resolved paths, and
    meta.json names the images.
    """
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    seen = {}

    def builder(
        prompt_file: Path,
        run_dir: Path,
        cwd: Path,
        condition: dict,
        model: str | None,
        effort: str | None,
        timeout: int,
        images: tuple[Path, ...] = (),
    ) -> tuple[list[str], Path | None]:
        seen["images"] = images
        return ["stub", "argv"], None

    monkeypatch.setitem(cross_family_ask.BUILDERS, "codex", builder)
    image = tmp_path / "render.png"
    image.write_bytes(b"png")

    meta = cross_family_ask.run_one(
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        cross_family_ask.resolve_launch("codex", "stub-model", None),
        images=[image],
    )

    assert meta["images"] == [str(image.resolve())]
    assert seen["images"] == (image.resolve(),)
    prompt = (
        tmp_path / "out" / "qid" / "as-shipped" / "codex" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert prompt.startswith("The image(s) attached to this prompt are ")
    assert prompt.endswith("question")


# ── per-family prompt override and the single-family re-run marker ─────────


def test_a_family_prompt_override_replaces_the_shared_prompt_for_that_family_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family prompt override replaces the shared prompt for that family alone.

    --family-prompt swaps the prompt text for one family and records which file it came from; a
    family with no override gets the shared --prompt-file.
    """
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    for family in ("codex", "antigravity"):
        monkeypatch.setitem(
            cross_family_ask.BUILDERS,
            family,
            lambda *a, **k: (["stub", "argv"], None),
        )

    (tmp_path / "shared.txt").write_text("shared question", encoding="utf-8")
    override = tmp_path / "codex_only.txt"
    override.write_text("a codex-only question", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "shared.txt"),
            "--families",
            "codex,antigravity",
            "--family-prompt",
            f"codex={override}",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() == 0

    codex_meta = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert codex_meta["prompt_source"] == str(override)
    codex_prompt = (
        tmp_path / "out" / "qid" / "as-shipped" / "codex" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert codex_prompt == "a codex-only question"

    agy_meta = json.loads(
        (
            tmp_path
            / "out"
            / "qid"
            / "as-shipped"
            / "antigravity"
            / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert agy_meta["prompt_source"] == str(tmp_path / "shared.txt")
    agy_prompt = (
        tmp_path / "out" / "qid" / "as-shipped" / "antigravity" / "prompt.txt"
    ).read_text(encoding="utf-8")
    assert agy_prompt == "shared question"


@pytest.mark.parametrize(
    "bad_flag", ["codex-no-separator", "nobody=missing.txt"]
)
def test_a_malformed_family_prompt_flag_is_refused_before_any_family_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    bad_flag: str,
) -> None:
    """A flag with no '=' and a flag naming an unknown family refuse, and no family launches."""
    launched = []
    monkeypatch.setattr(
        cross_family_ask, "run_one", lambda *a, **k: launched.append(a)
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--family-prompt",
            bad_flag,
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as refused:
        cross_family_ask.main()
    assert refused.value.code != 0
    err = capsys.readouterr().err
    assert "must be FAMILY=PATH" in err or "unknown family" in err
    assert launched == []


def test_an_effort_disagreeing_with_the_antigravity_id_is_refused_before_any_family_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An --effort the Antigravity id disagrees with refuses at the flag, and no family launches.

    The id's own effort named on the flag launches every family named, each with the effort its
    record will state.
    """
    launched: list[tuple] = []

    def stub_run_one(*a: object, **k: object) -> dict:
        launched.append(a)
        return cross_family_ask.failed_run_meta("stub", RuntimeError("stub"))

    monkeypatch.setattr(cross_family_ask, "run_one", stub_run_one)
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    config = tmp_path / "config.toml"
    config.write_text('model = "gpt-from-config"' + chr(10), encoding="utf-8")
    monkeypatch.setattr(cross_family_ask, "CODEX_CONFIG", config)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    (tmp_path / "out" / "qid" / "as-shipped").mkdir(parents=True)

    def argv(effort: str) -> list[str]:
        return [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex,antigravity",
            "--effort",
            effort,
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--serial",
        ]

    monkeypatch.setattr(sys, "argv", argv("medium"))
    with pytest.raises(SystemExit, match="pass gemini-4-argon-medium"):
        cross_family_ask.main()
    assert launched == []

    monkeypatch.setattr(sys, "argv", argv("high"))
    cross_family_ask.main()
    assert [(call[6].family, call[6].effort) for call in launched] == [
        ("codex", "high"),
        ("antigravity", "high"),
    ]


def test_a_family_prompt_flag_naming_a_missing_file_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    launched = []
    monkeypatch.setattr(
        cross_family_ask, "run_one", lambda *a, **k: launched.append(a)
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--family-prompt",
            f"codex={tmp_path / 'absent.txt'}",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as refused:
        cross_family_ask.main()
    assert refused.value.code != 0
    assert "file not found" in capsys.readouterr().err
    assert launched == []


@pytest.mark.parametrize(
    ("argv", "refusal"),
    [
        (["--families", "codex"], "codex model is unnamed"),
        (
            ["--families", "claude", "--effort", "turbo"],
            "claude effort 'turbo'",
        ),
    ],
)
def test_a_launch_a_family_refuses_is_refused_at_the_flag_before_any_family_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    refusal: str,
) -> None:
    """An unnamed Codex model and a Claude effort the harness would ignore refuse at the flag."""
    launched: list[tuple] = []

    def stub_run_one(*a: object, **k: object) -> dict:
        launched.append(a)
        return cross_family_ask.failed_run_meta("stub", RuntimeError("stub"))

    monkeypatch.setattr(cross_family_ask, "run_one", stub_run_one)
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setattr(
        cross_family_ask, "CODEX_CONFIG", tmp_path / "absent" / "config.toml"
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    (tmp_path / "out" / "qid" / "as-shipped").mkdir(parents=True)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            *argv,
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit, match=refusal):
        cross_family_ask.main()
    assert launched == []


def _set_up_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[dict]:
    """Stand the harnesses in for ``main`` and return the list each launch is recorded in.

    Codex and Antigravity build a fixed argv and answer at once; the question is in ``q.txt``.
    """
    seen: list[dict] = []
    _stand_in_harness(monkeypatch, recording_launch(seen))
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    for family in ("codex", "antigravity"):
        monkeypatch.setitem(
            cross_family_ask.BUILDERS,
            family,
            lambda *a, **k: (["stub", "argv"], None),
        )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    return seen


def _name_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    families: str,
    *flags: str,
) -> None:
    """Set the command line of a ``main`` run of ``families`` under question id ``qid``."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            families,
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
            *flags,
        ],
    )


def _snapshot(root: Path) -> dict[str, bytes | None]:
    """Return every path under ``root`` with its bytes, a folder's being ``None``."""
    return {
        entry: (root / entry).read_bytes()
        if (root / entry).is_file()
        else None
        for entry in _entries(root)
    }


def _assert_refused_as_a_rerun(
    raised: pytest.ExceptionInfo[SystemExit], folder: Path
) -> None:
    """Assert the refusal names ``folder`` and a new question id."""
    assert str(folder) in str(raised.value)
    assert "new question id" in str(raised.value)


def test_a_second_run_under_the_same_question_and_family_is_refused_and_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family's second run under one question id and condition is refused whole.

    Nothing launches, and every file under the out folder, the summary among them, keeps its name
    and its bytes, the other family's record included.
    """
    seen = _set_up_runs(tmp_path, monkeypatch)
    _name_run(tmp_path, monkeypatch, "codex,antigravity")
    assert cross_family_ask.main() == 0
    seen.clear()
    out = tmp_path / "out"
    first = _snapshot(out)
    codex_meta = json.loads(
        (out / "qid" / "as-shipped" / "codex" / "meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert codex_meta["rerun"] is False

    _name_run(tmp_path, monkeypatch, "codex")
    with pytest.raises(SystemExit) as refused:
        cross_family_ask.main()

    _assert_refused_as_a_rerun(refused, out / "qid" / "as-shipped" / "codex")
    assert seen == []
    assert _snapshot(out) == first


@pytest.mark.parametrize("flags", [(), ("--serial",)], ids=["pool", "serial"])
def test_a_run_naming_a_family_with_a_folder_and_one_without_is_refused_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: tuple[str, ...]
) -> None:
    """A refused family refuses the family named before it too: no folder, no launch."""
    seen = _set_up_runs(tmp_path, monkeypatch)
    _name_run(tmp_path, monkeypatch, "codex")
    assert cross_family_ask.main() == 0
    seen.clear()
    out = tmp_path / "out"
    first = _snapshot(out)

    _name_run(tmp_path, monkeypatch, "antigravity,codex", *flags)
    with pytest.raises(SystemExit) as refused:
        cross_family_ask.main()

    _assert_refused_as_a_rerun(refused, out / "qid" / "as-shipped" / "codex")
    assert not (out / "qid" / "as-shipped" / "antigravity").exists()
    assert seen == []
    assert _snapshot(out) == first


def test_a_run_over_the_files_of_a_launch_that_raised_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder a raised launch left, files and no record, is refused like a whole record."""
    _set_up_runs(tmp_path, monkeypatch)

    def raising_launch(*_args: object, **_kwargs: object) -> NoReturn:
        raise RuntimeError("a launch failure the runner does not catch")

    _stand_in_harness(monkeypatch, raising_launch)
    _name_run(tmp_path, monkeypatch, "codex")
    out = tmp_path / "out"
    cross_family_ask.main()
    folder = out / "qid" / "as-shipped" / "codex"
    assert (folder / "prompt.txt").is_file()
    assert not (folder / "meta.json").exists()

    seen: list[dict] = []
    _stand_in_harness(monkeypatch, recording_launch(seen))
    first = _snapshot(out)
    with pytest.raises(SystemExit) as refused:
        cross_family_ask.main()

    _assert_refused_as_a_rerun(refused, folder)
    assert seen == []
    assert _snapshot(out) == first


@pytest.mark.parametrize("after_a_run", [True, False], ids=["run", "empty"])
def test_run_one_into_a_family_folder_that_exists_is_refused_and_leaves_it_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after_a_run: bool
) -> None:
    """A direct call into an existing family folder, a run's or an empty one, launches nothing.

    The empty folder is what a run leaves when a step after the making of its folder raises.
    """
    seen: list[dict] = []
    _stand_in_harness(monkeypatch, recording_launch(seen))
    out = tmp_path / "out"
    folder = out / "qid" / "as-shipped" / "codex"
    if after_a_run:
        _run_codex_in(tmp_path, out)
        seen.clear()
    else:

        def raising_preface(*_args: object, **_kwargs: object) -> NoReturn:
            raise RuntimeError("a failure after the run folder was made")

        with monkeypatch.context() as raising:
            raising.setattr(cross_family_ask, "image_preface", raising_preface)
            with pytest.raises(RuntimeError, match="after the run folder"):
                _run_codex_in(tmp_path, out)
        assert folder.is_dir()
        assert not list(folder.iterdir())
    first = _snapshot(out)

    with pytest.raises(SystemExit) as refused:
        _run_codex_in(tmp_path, out)

    _assert_refused_as_a_rerun(refused, folder)
    assert seen == []
    assert _snapshot(out) == first


def test_a_family_named_twice_is_refused_before_anything_is_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A family named twice on one command line is refused, and no folder or launch results."""
    seen = _set_up_runs(tmp_path, monkeypatch)
    _name_run(tmp_path, monkeypatch, "codex,antigravity,codex")

    with pytest.raises(SystemExit, match="named twice"):
        cross_family_ask.main()

    assert seen == []
    assert not (tmp_path / "out").exists()


def test_a_later_run_naming_a_family_with_no_folder_joins_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later run of another family under the same question id succeeds beside the first.

    The summary holds both rows, and the earlier family's row and record are what they were.
    """
    _set_up_runs(tmp_path, monkeypatch)
    _name_run(tmp_path, monkeypatch, "codex")
    assert cross_family_ask.main() == 0
    out = tmp_path / "out"
    summary_path = out / "qid" / "as-shipped" / "summary.json"
    (codex_row,) = json.loads(summary_path.read_text(encoding="utf-8"))
    codex_folder = _snapshot(out / "qid" / "as-shipped" / "codex")

    _name_run(tmp_path, monkeypatch, "antigravity")
    assert cross_family_ask.main() == 0

    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {row["family"]: row for row in rows}["codex"] == codex_row
    assert {row["family"] for row in rows} == {"codex", "antigravity"}
    assert _snapshot(out / "qid" / "as-shipped" / "codex") == codex_folder


def test_a_malformed_prior_summary_loses_only_itself_not_the_current_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed prior summary loses itself, not the current run's rows.

    Over a summary.json that is not valid JSON, the run exits zero and the summary holds the
    current run's own rows.
    """
    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS,
        "codex",
        lambda *a, **k: (["stub", "argv"], None),
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    summary_path = tmp_path / "out" / "qid" / "as-shipped" / "summary.json"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("{not valid json", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--model",
            "stub-model",
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {row["family"] for row in summary} == {"codex"}


def test_a_missing_image_is_refused_before_any_family_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing image is refused, and no family launches."""
    launched = []
    monkeypatch.setattr(
        cross_family_ask, "run_one", lambda *a, **k: launched.append(a)
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--image",
            str(tmp_path / "absent.png"),
            "--cwd",
            str(tmp_path),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit) as refused:
        cross_family_ask.main()
    assert refused.value.code != 0
    assert "image not found" in capsys.readouterr().err
    assert launched == []


def test_the_antigravity_launch_grants_commands_beside_the_sandbox(
    tmp_path: Path,
) -> None:
    """The Antigravity argv carries the sandbox and a settings value allowing every command.

    Without the grant a headless run runs or denies a command by the harness's own agent mode.
    """
    prompt_file = tmp_path / "p.txt"
    prompt_file.write_text("question", encoding="utf-8")

    argv, _ = cross_family_ask.build_agy(
        prompt_file,
        tmp_path,
        tmp_path,
        cross_family_ask.CONDITIONS["no-tools"],
        "stub-model",
        None,
        5,
    )

    assert "--settings" in argv
    assert "--sandbox" in argv
    assert json.loads(argv[argv.index("--settings") + 1]) == {
        "permissions": {"allow": ["command(*)"]}
    }


# ── the browser-only condition: a cold reader with a browser alone ──────────


@pytest.mark.parametrize(
    "condition", ["as-shipped", "guidance-equalized", "no-tools"]
)
def test_each_earlier_condition_builds_the_claude_launch_it_built(
    tmp_path: Path, condition: str
) -> None:
    argv, _ = cross_family_ask.build_claude(
        tmp_path / "p.txt",
        tmp_path,
        tmp_path,
        cross_family_ask.CONDITIONS[condition],
        "opus",
        "high",
        5,
    )

    assert argv == [
        cross_family_ask.CLAUDE,
        "--print",
        "--output-format",
        "json",
        "--permission-mode",
        "plan",
        "--no-session-persistence",
        "--add-dir",
        str(tmp_path),
        "--effort",
        "high",
        "--mcp-config",
        str(tmp_path / "mcp.json"),
        "--strict-mcp-config",
        "--model",
        "opus",
    ]
    written = json.loads((tmp_path / "mcp.json").read_text(encoding="utf-8"))
    if condition == "no-tools":
        assert written == {"mcpServers": {}}
    else:
        assert written == cross_family_ask.load_tcip_mcp_config()


def test_the_browser_only_launch_carries_one_browser_server_and_no_repository(
    tmp_path: Path,
) -> None:
    """The browser-only argv: one headless, in-memory browser server and no built-in tool.

    Its MCP configuration is strict, it runs in restricted mode, it allows that server's tools
    but the one that runs code outside the page, and it carries neither the added directory, plan
    mode nor a setting-source list.
    """
    argv, _ = cross_family_ask.build_claude(
        tmp_path / "p.txt",
        tmp_path,
        tmp_path,
        cross_family_ask.CONDITIONS["browser-only"],
        "opus",
        "high",
        5,
    )

    config = json.loads((tmp_path / "mcp.json").read_text(encoding="utf-8"))
    ((name, server),) = config["mcpServers"].items()
    assert {"--headless", "--isolated"} <= set(server["args"])
    assert any(arg.startswith("@playwright/mcp@") for arg in server["args"])
    assert argv[argv.index("--mcp-config") + 1] == str(tmp_path / "mcp.json")
    assert "--strict-mcp-config" in argv
    assert "--restricted" in argv
    assert "--setting-sources" not in argv
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--allowedTools") + 1] == f"mcp__{name}"
    assert (
        argv[argv.index("--disallowedTools") + 1]
        == f"mcp__{name}__browser_run_code_unsafe"
    )
    assert "--add-dir" not in argv
    assert "--permission-mode" not in argv


def test_the_browser_only_reader_launches_from_an_empty_folder_with_no_claude_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader's launch drops what the in-app terminal withholds; the run records the condition.

    Two variables a Claude Code session sets and ``PYTHONPATH`` are absent from the launch's
    environment; ``PATH`` is kept.
    """
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    empty = tmp_path / "empty"
    empty.mkdir()

    meta = cross_family_ask.run_one(
        "qid",
        "browser-only",
        "a goal",
        empty,
        tmp_path / "out",
        5,
        cross_family_ask.resolve_launch("claude", None, None),
    )

    (launched,) = seen
    assert launched["cwd"] == empty
    assert launched["env"] is not None
    assert not {"CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "PYTHONPATH"} & set(
        launched["env"]
    )
    assert "PATH" in {k.upper() for k in launched["env"]}
    assert meta["condition"] == "browser-only"
    assert meta["mcp"] == "browser"
    assert meta["fault"] == ""


def test_the_browser_only_reader_and_the_in_app_terminals_child_inherit_one_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner's reader launch and the in-app terminal's ``spawn_env`` agree.

    Each route's own answer is compared, under one environment holding a variable of each
    withheld kind: the environment the runner hands the reader's launch, and the terminal child's
    less the two variables the terminal sets for its own child.
    """
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    for name in (
        "CLAUDECODE",
        "FORCE_PROMPT_CACHING_5M",
        "DISABLE_PROMPT_CACHING_OPUS",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(name, "1")
    empty = tmp_path / "empty"
    empty.mkdir()
    (tmp_path / "workspace").mkdir()

    with backend_session(tmp_path / "workspace"):
        cross_family_ask.run_one(
            "qid",
            "browser-only",
            "a goal",
            empty,
            tmp_path / "out",
            5,
            cross_family_ask.resolve_launch("claude", None, None),
        )
        child = terminal.spawn_env("t1")

    (launched,) = seen
    own = {"PWD", terminal.TERMINAL_SESSION_ENV}
    assert {k: v for k, v in launched["env"].items() if k not in own} == {
        k: v for k, v in child.items() if k not in own
    }


@pytest.mark.parametrize(
    ("family", "model"), [("codex", "stub-model"), ("antigravity", None)]
)
def test_browser_only_is_refused_for_any_family_but_claude_before_a_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    model: str | None,
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(SystemExit, match="claude alone"):
        cross_family_ask.run_one(
            "qid",
            "browser-only",
            "a goal",
            empty,
            tmp_path / "out",
            5,
            cross_family_ask.resolve_launch(family, model, None),
        )
    assert seen == []
    assert not (tmp_path / "out").exists()


def test_browser_only_is_refused_from_a_working_directory_holding_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )
    (tmp_path / "CLAUDE.md").write_text("an instruction", encoding="utf-8")

    with pytest.raises(SystemExit, match="not an empty directory"):
        cross_family_ask.run_one(
            "qid",
            "browser-only",
            "a goal",
            tmp_path,
            tmp_path / "out",
            5,
            cross_family_ask.resolve_launch("claude", None, None),
        )
    assert seen == []


def test_the_command_refuses_browser_only_for_codex_before_claude_launches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[dict] = []
    monkeypatch.setattr(
        cross_family_ask, "launch_harness", recording_launch(seen)
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    (tmp_path / "q.txt").write_text("a goal", encoding="utf-8")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--condition",
            "browser-only",
            "--families",
            "claude,codex",
            "--model",
            "stub-model",
            "--cwd",
            str(empty),
            "--out",
            str(tmp_path / "out"),
        ],
    )

    with pytest.raises(SystemExit, match="claude alone"):
        cross_family_ask.main()
    assert seen == []


def _stand_in_harness(
    monkeypatch: pytest.MonkeyPatch,
    launch: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    """Replace the harness launch and its version probe, leaving git and the builders real."""
    monkeypatch.setattr(cross_family_ask, "launch_harness", launch)
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )


def _run_codex_in(cwd: Path, out: Path) -> dict:
    """Run one codex family on one question from ``cwd``, its run folder under ``out``."""
    return cross_family_ask.run_one(
        "qid",
        "as-shipped",
        "question",
        cwd,
        out,
        5,
        cross_family_ask.resolve_launch("codex", "stub-model", None),
    )


def _meta_on_disk(out: Path) -> dict:
    """Return the ``meta.json`` the codex run of :func:`_run_codex_in` wrote under ``out``."""
    path = out / "qid" / "as-shipped" / "codex" / "meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _commit_while_running(repo: Path, made: list[str]) -> Callable[[], None]:
    """Return a stand-in's action that commits a new file in ``repo`` and appends the new HEAD."""

    def commit() -> None:
        write(repo / "moved.txt", "moved")
        made.append(commit_all(repo, "moved while the harness ran"))

    return commit


def test_a_checkout_that_moved_during_the_run_is_named_in_the_fault_by_both_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in commits while it runs: the fault names both hashes and the record holds each."""
    repo = scratch_repo(tmp_path)
    first = commit_all(repo, "first")
    made: list[str] = []
    _stand_in_harness(
        monkeypatch,
        recording_launch([], during=_commit_while_running(repo, made)),
    )

    meta = _run_codex_in(repo, tmp_path / "out")

    (second,) = made
    assert first in meta["fault"]
    assert second in meta["fault"]
    assert meta["commit_at_start"] == first
    assert meta["commit_at_end"] == second
    assert meta == _meta_on_disk(tmp_path / "out")


def test_a_checkout_left_alone_records_one_commit_at_both_ends_and_no_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in that leaves the repository alone: both keys hold HEAD and nothing is faulted."""
    repo = scratch_repo(tmp_path)
    head = commit_all(repo, "first")
    _stand_in_harness(monkeypatch, recording_launch([]))

    meta = _run_codex_in(repo, tmp_path / "out")

    assert "commit_at_start" in meta
    assert meta["commit_at_start"] == head
    assert meta["commit_at_end"] == head
    assert meta["commit_read_error"] == ""
    assert meta["fault"] == ""
    assert meta == _meta_on_disk(tmp_path / "out")


def test_the_command_exits_nonzero_for_a_moved_checkout_and_the_summary_row_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Through ``main`` the moved checkout is exit 1, and summary row and record carry one fault."""
    repo = scratch_repo(tmp_path)
    first = commit_all(repo, "first")
    made: list[str] = []
    _stand_in_harness(
        monkeypatch,
        recording_launch([], during=_commit_while_running(repo, made)),
    )
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cross_family_ask.py",
            "--question-id",
            "qid",
            "--prompt-file",
            str(tmp_path / "q.txt"),
            "--families",
            "codex",
            "--model",
            "stub-model",
            "--cwd",
            str(repo),
            "--out",
            str(out),
            "--timeout",
            "5",
        ],
    )

    assert cross_family_ask.main() == 1

    (second,) = made
    summary = json.loads(
        (out / "qid" / "as-shipped" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    (row,) = summary
    fault = _meta_on_disk(out)["fault"]
    assert row["fault"] == fault
    assert first in fault
    assert second in fault


def test_a_failed_harness_and_a_moved_checkout_are_both_in_the_fault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in that commits and exits 1: the fault names the exit and both hashes."""
    repo = scratch_repo(tmp_path)
    first = commit_all(repo, "first")
    made: list[str] = []
    _stand_in_harness(
        monkeypatch,
        recording_launch([], code=1, during=_commit_while_running(repo, made)),
    )

    meta = _run_codex_in(repo, tmp_path / "out")

    (second,) = made
    assert first in meta["fault"]
    assert second in meta["fault"]
    assert "the harness exited 1" in meta["fault"]


def test_a_checkout_whose_commit_cannot_be_read_after_the_run_keeps_the_record_and_faults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stand-in that damages HEAD: the record is written, holds git's message and is faulted."""
    repo = scratch_repo(tmp_path)
    head = commit_all(repo, "first")
    _stand_in_harness(
        monkeypatch,
        recording_launch(
            [],
            during=lambda: write(
                repo / ".git" / "HEAD", "ref: refs/heads/missing\n"
            ),
        ),
    )

    meta = _run_codex_in(repo, tmp_path / "out")

    assert meta["commit_read_error"] != ""
    assert meta["commit_read_error"] in meta["fault"]
    assert meta["commit_at_start"] == head
    assert meta["commit_at_end"] is None
    assert meta == _meta_on_disk(tmp_path / "out")


def test_a_checkout_whose_commit_cannot_be_read_refuses_the_run_before_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A repository with no commit refuses the run with git's message and writes nothing.

    No harness is started.
    """
    repo = scratch_repo(tmp_path)
    git_says = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    ).stderr.strip()
    assert git_says
    seen: list[dict] = []
    _stand_in_harness(monkeypatch, recording_launch(seen))

    with pytest.raises(RuntimeError, match=re.escape(git_says)) as raised:
        _run_codex_in(repo, tmp_path / "out")

    assert type(raised.value) is cross_family_ask.CommitReadError
    assert seen == []
    assert not (tmp_path / "out").exists()


def test_a_directory_in_no_checkout_records_no_commit_and_runs_no_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """From a directory with no checkout the commit keys hold None, with no fault and no command."""
    plain = tmp_path / "plain"
    plain.mkdir()
    commands: list[object] = []

    def record_command(*args: object, **_kwargs: object) -> NoReturn:
        commands.append(args)
        raise AssertionError(
            "no command is run for a directory in no checkout"
        )

    _stand_in_harness(monkeypatch, recording_launch([]))
    monkeypatch.setattr(cross_family_ask.subprocess, "run", record_command)

    meta = _run_codex_in(plain, tmp_path / "out")

    assert "commit_at_start" in meta
    assert meta["commit_at_start"] is None
    assert meta["commit_at_end"] is None
    assert meta["fault"] == ""
    assert commands == []
    assert meta == _meta_on_disk(tmp_path / "out")


def _prepare_default_out_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
    out: Path | None = None,
) -> list[dict]:
    """Set up one codex run of the ``no-tools`` condition for ``main`` and return its launches.

    ``REPO_ROOT`` is ``repo_root``; ``--cwd`` and the process's current folder are two other
    folders under ``tmp_path``, so a default found from either of them is not the one under test.
    Git's variables that choose a repository are removed and its search ceiling is ``tmp_path``.
    The harness is a stand-in that only answers, and the returned list records each launch.
    """
    for name in git(tmp_path, "rev-parse", "--local-env-vars").split():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    harness_cwd = tmp_path / "harness-cwd"
    harness_cwd.mkdir()
    process_cwd = tmp_path / "process-cwd"
    process_cwd.mkdir()
    monkeypatch.chdir(process_cwd)
    seen: list[dict] = []
    _stand_in_harness(monkeypatch, recording_launch(seen))
    monkeypatch.setattr(
        cross_family_ask.shutil, "which", lambda *a, **k: "/stub/harness"
    )
    monkeypatch.setattr(cross_family_ask, "REPO_ROOT", repo_root)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    argv = [
        "cross_family_ask.py",
        "--question-id",
        "qid",
        "--prompt-file",
        str(tmp_path / "q.txt"),
        "--condition",
        "no-tools",
        "--families",
        "codex",
        "--model",
        "stub-model",
        "--timeout",
        "5",
        "--cwd",
        str(harness_cwd),
    ]
    if out is not None:
        argv += ["--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    return seen


def _entries(root: Path) -> list[str]:
    """Return every path under ``root``, relative to it and sorted."""
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def _git_says_in(folder: Path) -> str:
    """Return git's own message for ``git rev-parse --show-toplevel`` run in ``folder``."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=str(folder),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    return proc.stderr.strip()


def _assert_refused_before_any_run(
    raised: pytest.ExceptionInfo[SystemExit],
    repo_root: Path,
    seen: list[dict],
    tmp_path: Path,
) -> None:
    """Assert the exit names ``repo_root`` and ``--out``, and no launch or run folder came first."""
    message = str(raised.value)
    assert str(repo_root) in message
    assert "--out" in message
    assert seen == []
    assert not list(tmp_path.rglob("cross-family"))


def test_a_run_given_no_out_from_a_linked_worktree_writes_in_the_main_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ``--out``, a linked worktree's run lands in the main working tree's records."""
    main_tree = scratch_repo(tmp_path)
    commit_all(main_tree, "first")
    worktree = tmp_path / "linked"
    git(main_tree, "worktree", "add", "-q", str(worktree), "-b", "other")
    _prepare_default_out_run(tmp_path, monkeypatch, worktree)

    assert cross_family_ask.main() == 0

    condition = (
        main_tree / ".claude" / "records" / "cross-family" / "qid" / "no-tools"
    )
    assert (condition / "codex" / "meta.json").is_file()
    assert (condition / "summary.json").is_file()
    assert not (worktree / ".claude" / "records").exists()


def test_a_run_given_no_out_from_a_main_working_tree_with_its_git_directory_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The records lie in the working tree, never inside or beside its separate git directory."""
    work = tmp_path / "work"
    git_dir = tmp_path / "git-dirs" / "work.git"
    git_dir.parent.mkdir()
    git(tmp_path, "init", "-q", "--separate-git-dir", str(git_dir), str(work))
    git(work, "config", "user.email", "test@example.com")
    git(work, "config", "user.name", "Test")
    write(work / "notes.txt", "notes\n")
    commit_all(work, "first")
    _prepare_default_out_run(tmp_path, monkeypatch, work)

    assert cross_family_ask.main() == 0

    condition = (
        work / ".claude" / "records" / "cross-family" / "qid" / "no-tools"
    )
    assert (condition / "codex" / "meta.json").is_file()
    assert (condition / "summary.json").is_file()
    assert [path.name for path in git_dir.parent.iterdir()] == ["work.git"]
    assert not list(git_dir.rglob(".claude"))


def test_a_run_given_no_out_from_a_worktree_of_a_bare_repository_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No working tree holds a bare repository, so none is picked and nothing is written."""
    source = scratch_repo(tmp_path)
    commit_all(source, "first")
    repos = tmp_path / "repos"
    repos.mkdir()
    bare = repos / "bare.git"
    git(tmp_path, "clone", "-q", "--bare", str(source), str(bare))
    worktree = tmp_path / "bare-linked"
    git(bare, "worktree", "add", "-q", "--detach", str(worktree))
    seen = _prepare_default_out_run(tmp_path, monkeypatch, worktree)
    bare_before, worktree_before = _entries(bare), _entries(worktree)

    with pytest.raises(SystemExit) as raised:
        cross_family_ask.main()

    _assert_refused_before_any_run(raised, worktree, seen, tmp_path)
    assert _git_says_in(repos) in str(raised.value)
    assert _entries(bare) == bare_before
    assert _entries(worktree) == worktree_before


@pytest.mark.parametrize("placement", ["b.git", "i/b.git"])
def test_a_run_given_no_out_from_a_worktree_of_a_bare_repository_inside_another_working_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, placement: str
) -> None:
    """The working tree around a bare repository is not its own; the exit is the tool's text."""
    outer = scratch_repo(tmp_path)
    commit_all(outer, "first")
    bare = outer / placement
    bare.parent.mkdir(exist_ok=True)
    git(tmp_path, "clone", "-q", "--bare", str(outer), str(bare))
    worktree = tmp_path / "bare-linked"
    git(bare, "worktree", "add", "-q", "--detach", str(worktree))
    seen = _prepare_default_out_run(tmp_path, monkeypatch, worktree)
    outer_before, worktree_before = _entries(outer), _entries(worktree)

    with pytest.raises(SystemExit) as raised:
        cross_family_ask.main()

    _assert_refused_before_any_run(raised, worktree, seen, tmp_path)
    message = str(raised.value)
    assert "does not lie in a working tree of its own" in message
    assert "fatal:" not in message
    assert _entries(outer) == outer_before
    assert _entries(worktree) == worktree_before


def test_a_run_given_no_out_from_a_folder_in_no_repository_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A folder in no repository names no main working tree; the exit carries git's message."""
    no_repository = tmp_path / "no-repository"
    no_repository.mkdir()
    seen = _prepare_default_out_run(tmp_path, monkeypatch, no_repository)

    with pytest.raises(SystemExit) as raised:
        cross_family_ask.main()

    _assert_refused_before_any_run(raised, no_repository, seen, tmp_path)
    assert _git_says_in(no_repository) in str(raised.value)


def test_a_run_given_no_out_where_git_cannot_be_run_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no git on the search path the exit carries the operating system's message."""
    repo = scratch_repo(tmp_path)
    commit_all(repo, "first")
    seen = _prepare_default_out_run(tmp_path, monkeypatch, repo)
    no_git = tmp_path / "no-git-here"
    no_git.mkdir()
    monkeypatch.setenv("PATH", str(no_git))
    with pytest.raises(OSError) as probe:
        subprocess.run(["git", "--version"], capture_output=True, check=False)

    with pytest.raises(SystemExit) as raised:
        cross_family_ask.main()

    _assert_refused_before_any_run(raised, repo, seen, tmp_path)
    assert str(probe.value) in str(raised.value)


def test_a_run_given_out_from_a_folder_in_no_repository_writes_under_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A given ``--out`` never reaches the check for a main working tree."""
    no_repository = tmp_path / "no-repository"
    no_repository.mkdir()
    given = tmp_path / "given"
    _prepare_default_out_run(tmp_path, monkeypatch, no_repository, out=given)

    assert cross_family_ask.main() == 0

    assert (given / "qid" / "no-tools" / "codex" / "meta.json").is_file()
    assert (given / "qid" / "no-tools" / "summary.json").is_file()
