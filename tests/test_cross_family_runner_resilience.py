"""The cross-family runner keeps each family's answer when one harness returns nothing.

The runner poses one question to several agent harnesses concurrently and writes each one's
transcript. A harness that produces no stdout does not raise inside the worker.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING, NoReturn

import pytest

from tools import cross_family_ask

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_a_model_id_that_states_its_effort_is_the_effort_whatever_the_flag_said() -> (
    None
):
    """A model id that states its effort takes it, with the --effort flag medium or unset."""
    assert (
        cross_family_ask.effective_effort(
            "antigravity", "gemini-3.1-pro-high", "medium"
        )
        == "high"
    )
    assert (
        cross_family_ask.effective_effort(
            "antigravity", "gemini-3.1-pro-high", None
        )
        == "high"
    )


def test_an_effort_claude_would_ignore_is_refused_by_name() -> None:
    with pytest.raises(SystemExit, match="claude effort 'turbo'"):
        cross_family_ask.effective_effort("claude", "opus", "turbo")
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
        "codex",
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        "stub-model",
        None,
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
        family: str,
        *args: Path | int | str | None,
        **kwargs: list[Path] | str | None,
    ) -> dict:
        if family == "codex":
            _explode()
        return real_run_one(family, *args, **kwargs)

    monkeypatch.setattr(
        cross_family_ask.subprocess, "run", _stub_run("an answer")
    )
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    # Keep the test hermetic: without this, main()'s harness-on-PATH check exits before the
    # collection loop whenever the real codex/agy CLIs are absent (the CI environment).
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
        family: str,
        qid: str,
        condition: str,
        prompt: str,
        *args: Path | int | str | None,
        **kwargs: list[Path] | str | None,
    ) -> dict:
        seen.append(prompt)
        return real_run_one(family, qid, condition, prompt, *args, **kwargs)

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
        "codex",
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        "stub-model",
        None,
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
        "import sys; sys.stdin.reconfigure(encoding='utf-8'); "
        "sys.stdout.reconfigure(encoding='utf-8'); "
        "sys.stdout.write(sys.stdin.read())",
    ]
    monkeypatch.setattr(
        cross_family_ask, "harness_version", lambda *a, **k: "stub-version"
    )
    monkeypatch.setitem(
        cross_family_ask.BUILDERS, "codex", lambda *a, **k: (echo_child, None)
    )

    prompt = "a box-drawing rule: ───"
    meta = cross_family_ask.run_one(
        "codex",
        "qid",
        "as-shipped",
        prompt,
        tmp_path,
        tmp_path / "out",
        30,
        "stub-model",
        None,
    )

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "codex"
    body = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    assert "─" in body
    assert (run_dir / "stdout.txt").read_text(encoding="utf-8") == body
    assert meta["exit_code"] == 0


# ── the codex model is named on every run ───────────────────────────────────


def test_an_unnamed_codex_model_with_no_config_is_refused_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unnamed codex model with no config is refused by name."""
    monkeypatch.setattr(
        cross_family_ask, "CODEX_CONFIG", tmp_path / "absent" / "config.toml"
    )

    with pytest.raises(SystemExit, match="codex model is unnamed"):
        cross_family_ask.resolve_codex_model(None)


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
        "codex",
        "qid",
        "as-shipped",
        "question",
        tmp_path,
        tmp_path / "out",
        5,
        "stub-model",
        None,
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
    assert launched == []


def test_a_family_prompt_flag_naming_a_missing_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert launched == []


def test_a_single_family_rerun_marks_itself_and_leaves_the_others_summary_row_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-family rerun marks itself a rerun and keeps both families' summary rows.

    Re-running codex under a question id that holds both families' results leaves antigravity's
    meta.json unchanged when read as UTF-8 text and both families in the summary, and the rerun's
    meta says it is a rerun where the first run's did not.
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
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    first_argv = [
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
    ]
    monkeypatch.setattr(sys, "argv", first_argv)
    assert cross_family_ask.main() == 0

    antigravity_meta_path = (
        tmp_path / "out" / "qid" / "as-shipped" / "antigravity" / "meta.json"
    )
    antigravity_meta_before = antigravity_meta_path.read_text(encoding="utf-8")
    codex_meta_before = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert codex_meta_before["rerun"] is False

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

    assert (
        antigravity_meta_path.read_text(encoding="utf-8")
        == antigravity_meta_before
    )

    codex_meta_after = json.loads(
        (
            tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json"
        ).read_text(encoding="utf-8")
    )
    assert codex_meta_after["rerun"] is True

    summary = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    families_in_summary = {row["family"] for row in summary}
    assert families_in_summary == {"codex", "antigravity"}


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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
    assert launched == []
