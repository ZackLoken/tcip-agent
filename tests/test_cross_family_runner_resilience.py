"""One harness returning nothing must not destroy the other families' answers.

The cross-family runner poses one question to several agent harnesses concurrently and writes each
one's transcript. A harness that produces no stdout used to raise inside the worker, and the
exception escaped the result-collection loop, so the sibling families' completed runs were lost with
it. That is the expensive failure: the runs that succeeded are the artifact, and a harness that
returns nothing is a normal outcome rather than a programming error.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "tools" / "cross_family_ask.py"


def _load():
    """Import the runner by path: it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("cross_family_ask", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cross_family_ask"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def runner():
    return _load()


def test_a_model_id_that_states_its_effort_is_the_effort_whatever_the_flag_said(runner):
    """One invocation naming every family can carry one --effort: a family whose model id
    carries the effort records the id's own, never refusing and never passing the flag on."""
    assert runner.effective_effort("antigravity", "gemini-3.1-pro-high", "medium") == "high"
    assert runner.effective_effort("antigravity", "gemini-3.1-pro-high", None) == "high"


def test_an_effort_claude_would_ignore_is_refused_by_name(runner):
    with pytest.raises(SystemExit, match="claude effort 'turbo'"):
        runner.effective_effort("claude", "opus", "turbo")
    assert runner.effective_effort("claude", "opus", "high") == "high"
    assert runner.effective_effort("codex", "gpt-5.6-sol", "high") == "high"


def _stub_run(stdout, stderr=""):
    """A subprocess.run replacement returning the given streams verbatim."""
    def _run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=["stub"], returncode=0,
                                           stdout=stdout, stderr=stderr)
    return _run


@pytest.mark.parametrize("stdout", [None, b"bytes not str"])
def test_a_harness_returning_no_usable_stdout_still_records_its_run(runner, tmp_path, monkeypatch,
                                                                   stdout):
    """A None or bytes stdout is written as text rather than raising TypeError."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run(stdout))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setitem(runner.BUILDERS, "codex",
                        lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    meta = runner.run_one("codex", "qid", "as-shipped", "question",
                          tmp_path, tmp_path / "out", 5, "stub-model", None)

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "codex"
    assert (run_dir / "stdout.txt").is_file()
    assert (run_dir / "meta.json").is_file()
    assert meta["family"] == "codex"


def test_one_familys_failure_leaves_the_others_results_intact(runner, tmp_path, monkeypatch):
    """The collection loop records a failed run and keeps every sibling's result."""
    def _explode(*_args, **_kwargs):
        raise RuntimeError("harness blew up")

    real_run_one = runner.run_one

    def _dispatch(family, *args, **kwargs):
        if family == "codex":
            _explode()
        return real_run_one(family, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", _stub_run("an answer"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    # Keep the test hermetic: without this, main()'s harness-on-PATH check exits before the
    # collection loop whenever the real codex/agy CLIs are absent (the CI environment).
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    for fam in ("codex", "antigravity"):
        monkeypatch.setitem(runner.BUILDERS, fam, lambda *a, **k: (["stub", "argv"], None))
    monkeypatch.setattr(runner, "run_one", _dispatch)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"),
        "--families", "codex,antigravity",
        "--model", "stub-model",
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    # Non-zero, because a family that failed is a failed invocation and must not read as clean.
    assert runner.main() != 0

    surviving = tmp_path / "out" / "qid" / "as-shipped" / "antigravity" / "response.md"
    assert surviving.is_file(), "the family that answered lost its transcript to the one that failed"
    assert surviving.read_text(encoding="utf-8").strip() == "an answer"

    summary = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "summary.json").read_text(encoding="utf-8"))
    rows = {r["family"]: r for r in (summary if isinstance(summary, list) else summary["runs"])}
    assert rows["antigravity"]["exit_code"] == 0
    assert rows["codex"]["response_source"] == "runner_error"
    assert "harness blew up" in rows["codex"]["runner_error"]


def test_a_prompt_file_with_a_byte_order_mark_reaches_the_harness_without_it(
    runner, tmp_path, monkeypatch
) -> None:
    """An editor-saved BOM is an encoding artifact, not part of the question text."""
    seen: list[str] = []

    real_run_one = runner.run_one

    def _capture(family, qid, condition, prompt, *args, **kwargs):
        seen.append(prompt)
        return real_run_one(family, qid, condition, prompt, *args, **kwargs)

    monkeypatch.setattr(runner.subprocess, "run", _stub_run("an answer"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "codex", lambda *a, **k: (["stub", "argv"], None))
    monkeypatch.setattr(runner, "run_one", _capture)
    (tmp_path / "q.txt").write_text("question", encoding="utf-8-sig")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"),
        "--families", "codex",
        "--model", "stub-model",
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() == 0
    assert len(seen) == 1
    assert seen[0] == "question"


def test_a_codex_agent_message_stream_yields_the_last_message(runner):
    """Real codex JSONL carries the answer inside item.completed/agent_message events, not
    at the top level, so the stream fallback must know that shape before it falls further."""
    lines = [
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_0", "type": "agent_message", "text": "first pass"}}),
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_1", "type": "command_execution",
                             "command": "ls", "exit_code": 0, "status": "completed"}}),
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_2", "type": "agent_message",
                             "text": "second and final pass"}}),
        json.dumps({"type": "turn.completed"}),
    ]
    text, source = runner.extract_response("codex", "\n".join(lines), None)
    assert text == "second and final pass"
    assert source == "stream_agent_message"


def test_an_oversized_raw_stream_is_a_failed_extraction_not_an_answer(runner):
    """A stream past the size bound is dumped nowhere near a real answer; it reads as failed
    extraction rather than as a 100k-character reply."""
    huge = "not json, just noise " * 6000
    assert len(huge.strip()) > runner.MAX_RAW_STREAM_CHARS
    text, source = runner.extract_response("codex", huge, None)
    assert (text, source) == ("", "extraction_failed")


def test_an_oversized_raw_stream_through_main_fails_the_run(runner, tmp_path, monkeypatch):
    """The exit gate must not read a run whose stream could not be extracted as clean."""
    huge = "not json, just noise " * 6000
    monkeypatch.setattr(runner.subprocess, "run", _stub_run(huge))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "codex", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"),
        "--families", "codex",
        "--model", "stub-model",
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() != 0

    meta = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json").read_text(encoding="utf-8"))
    assert meta["response_source"] == "extraction_failed"


def test_a_payload_with_only_empty_recognized_keys_reads_as_an_empty_response(runner):
    """The antigravity SUCCESS-with-empty-answer shape: a recognized key is present but blank,
    so the payload is a genuinely empty answer, not an unknown shape to dump whole."""
    stdout = json.dumps({
        "conversation_id": "cadb33f6-6b32-413f-9ddd-0b2c168c573b",
        "status": "SUCCESS",
        "response": "",
        "duration_seconds": 12.8666267,
        "num_turns": 1,
        "usage": {
            "input_tokens": 19088, "output_tokens": 891, "thinking_tokens": 661,
            "cache_read_tokens": 24842, "total_tokens": 19979,
        },
    })
    text, source = runner.extract_response("antigravity", stdout, None)
    assert (text, source) == ("", "empty_response")


def test_an_empty_response_through_main_fails_the_run(runner, tmp_path, monkeypatch):
    """An empty recognized answer must not read as a clean run through the exit gate."""
    stdout = json.dumps({"status": "SUCCESS", "response": "  "})
    monkeypatch.setattr(runner.subprocess, "run", _stub_run(stdout))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "antigravity", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"),
        "--families", "antigravity",
        "--model", "stub-model",
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() != 0


def test_completely_empty_stdout_does_not_read_as_a_clean_run(runner, tmp_path, monkeypatch):
    """A harness that writes nothing at all falls through extraction to a raw, empty string,
    a response-source name the exit gate never special-cased; the check has to be on the
    character count itself, not on a list of the names known to produce an empty one."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run(""))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "antigravity", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"), "--families", "antigravity",
        "--model", "stub-model", "--cwd", str(tmp_path),
        "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() != 0

    meta = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "antigravity" / "meta.json").read_text(encoding="utf-8"))
    assert meta["response_source"] == "raw_stdout"
    assert meta["fault"]


def test_a_whitespace_only_agent_message_stream_is_a_failed_run_not_a_clean_one(
    runner, tmp_path, monkeypatch
):
    """A codex stream whose only captured agent message is whitespace is not an answer:
    response_chars counts the stripped text, so this run must classify as a fault, not a clean
    zero-content success."""
    lines = [
        json.dumps({"type": "item.completed",
                    "item": {"id": "item_0", "type": "agent_message", "text": "   \n  "}}),
        json.dumps({"type": "turn.completed"}),
    ]
    monkeypatch.setattr(runner.subprocess, "run", _stub_run("\n".join(lines)))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "codex", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"), "--families", "codex",
        "--model", "stub-model", "--cwd", str(tmp_path),
        "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() != 0

    meta = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json").read_text(encoding="utf-8"))
    assert meta["response_source"] == "stream_agent_message"
    assert meta["response_chars"] == 0
    assert meta["fault"]


def test_a_runs_own_fault_verdict_is_recorded_in_its_meta_json(runner, tmp_path, monkeypatch):
    """The record a later reader inspects on its own must already carry why a run failed, not
    only the aggregate table printed at the end of this process's own stdout."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run("a real answer from the harness"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setitem(runner.BUILDERS, "codex", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    meta = runner.run_one("codex", "qid", "as-shipped", "question",
                          tmp_path, tmp_path / "out", 5, "stub-model", None)
    assert meta["fault"] == ""

    on_disk = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json").read_text(encoding="utf-8"))
    assert on_disk["fault"] == ""


def test_a_small_plain_text_stdout_still_returns_as_a_raw_stdout_answer_through_main(
    runner, tmp_path, monkeypatch
):
    """Hardening the extraction must not turn an ordinary short answer into a failure."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run("a short plain-text answer"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "codex", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"),
        "--families", "codex",
        "--model", "stub-model",
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() == 0

    meta = json.loads(
        (tmp_path / "out" / "qid" / "as-shipped" / "codex" / "meta.json").read_text(encoding="utf-8"))
    assert meta["response_source"] == "raw_stdout"
    assert meta["response_chars"] == len("a short plain-text answer")


def test_a_dict_payload_with_no_recognized_keys_still_returns_whole_payload(runner):
    """An unknown shape with none of the recognized keys still dumps whole: that recovery
    path for shapes this runner has not learned yet must stay intact."""
    stdout = json.dumps({"unexpected_field": "some value", "another_field": 42})
    text, source = runner.extract_response("antigravity", stdout, None)
    assert source == "whole_payload"
    assert "unexpected_field" in text


def test_a_non_string_answer_value_still_dumps_the_whole_payload(runner):
    """A recognized key holding a non-string (a content-block list, a null) is an unknown
    shape carrying possible data, never a confirmed-empty answer to discard."""
    stdout = json.dumps({"content": [{"type": "text", "text": "the real answer"}]})
    text, source = runner.extract_response("antigravity", stdout, None)
    assert source == "whole_payload"
    assert "the real answer" in text

    text, source = runner.extract_response("antigravity", json.dumps({"result": None}), None)
    assert source == "whole_payload"


def test_a_character_outside_the_console_codepage_survives_the_trip_to_a_stdin_child(
    runner, tmp_path, monkeypatch
) -> None:
    """The child gets utf-8 stdin and its utf-8 stdout decodes intact, whatever the locale."""
    echo_child = [sys.executable, "-c",
                  "import sys; sys.stdin.reconfigure(encoding='utf-8'); "
                  "sys.stdout.reconfigure(encoding='utf-8'); "
                  "sys.stdout.write(sys.stdin.read())"]
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setitem(runner.BUILDERS, "codex", lambda *a, **k: (echo_child, None))

    prompt = "a box-drawing rule: ───"
    meta = runner.run_one("codex", "qid", "as-shipped", prompt,
                          tmp_path, tmp_path / "out", 30, "stub-model", None)

    run_dir = tmp_path / "out" / "qid" / "as-shipped" / "codex"
    body = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    assert "─" in body
    assert (run_dir / "stdout.txt").read_text(encoding="utf-8") == body
    assert meta["exit_code"] == 0


# ── the codex model is named on every run ───────────────────────────────────


def test_an_unnamed_codex_model_with_no_config_is_refused_before_launch(runner, tmp_path, monkeypatch):
    """A run on the harness's built-in default would record no model, so it is refused by name."""
    monkeypatch.setattr(runner, "CODEX_CONFIG", tmp_path / "absent" / "config.toml")

    with pytest.raises(SystemExit, match="codex model is unnamed"):
        runner.resolve_codex_model(None)


def test_the_codex_model_is_read_from_the_config_and_its_source_recorded(runner, tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('model = "gpt-from-config"' + chr(10), encoding="utf-8")
    monkeypatch.setattr(runner, "CODEX_CONFIG", config)

    assert runner.resolve_codex_model(None) == ("gpt-from-config", str(config))
    assert runner.resolve_codex_model("gpt-from-flag") == ("gpt-from-flag", "flag")


def test_a_run_with_no_usable_answer_says_so_in_words_at_the_end_of_its_output(
    runner, tmp_path, monkeypatch, capsys
):
    """The exit code is not the only signal a refused run gets.

    A caller that pipes this script somewhere takes the pipeline's status from the last command,
    so the refusal has to be legible in the output itself, at the end, where a tail still shows it.
    """
    monkeypatch.setattr(runner.subprocess, "run",
                        _stub_run(json.dumps({"status": "SUCCESS", "response": "  "})))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "antigravity", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"), "--families", "antigravity",
        "--model", "stub-model", "--cwd", str(tmp_path),
        "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() != 0
    captured = capsys.readouterr()
    assert "RUN FAILED" in captured.out
    assert "antigravity" in captured.out.split("RUN FAILED")[1]
    assert "no answer text" in captured.out
    # Last, so a tail of the output cannot cut the verdict off in favour of a "wrote ..." line.
    assert captured.out.rstrip().splitlines()[-1].strip().startswith("Nothing here is reviewable")
    assert "RUN FAILED" in captured.err


# ── the tcip MCP block is read from .mcp.json, not restated ─────────────────


def test_load_tcip_mcp_config_reads_the_repos_own_mcp_json(runner):
    import json as _json

    declared = _json.loads((SCRIPT.parents[1] / ".mcp.json").read_text(encoding="utf-8"))
    assert runner.load_tcip_mcp_config() == {
        "mcpServers": {"tcip": declared["mcpServers"]["tcip"]}
    }


def test_load_tcip_mcp_config_refuses_when_mcp_json_is_missing(runner, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="cannot read the tcip MCP server block"):
        runner.load_tcip_mcp_config()


def test_load_tcip_mcp_config_refuses_when_the_tcip_block_is_missing(runner, tmp_path, monkeypatch):
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"claude-context": {}}}), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no mcpServers.tcip block"):
        runner.load_tcip_mcp_config()


def test_a_run_whose_families_all_answered_says_so_rather_than_staying_silent(
    runner, tmp_path, monkeypatch, capsys
):
    """The verdict must admit a good run too, not only refuse a bad one."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run("a real answer from the harness"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    monkeypatch.setitem(runner.BUILDERS, "antigravity", lambda *a, **k: (["stub", "argv"], None))
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"), "--families", "antigravity",
        "--model", "stub-model", "--cwd", str(tmp_path),
        "--out", str(tmp_path / "out"), "--timeout", "5",
    ])

    assert runner.main() == 0
    captured = capsys.readouterr()
    assert "RUN OK: 1 of 1 families answered." in captured.out
    assert "RUN FAILED" not in captured.out
    assert "RUN FAILED" not in captured.err


def test_an_attached_image_reaches_codex_as_a_flag_and_the_other_families_in_the_prompt(
        runner, tmp_path):
    """Every family is vision-capable and takes an image its own way: codex through exec -i, each
    flag closed by the next option so the trailing stdin marker survives the variadic option;
    antigravity through an @path in the prompt; Claude Code through a named path its own file
    reader opens. A run with no image builds exactly as before."""
    image = tmp_path / "render.png"
    image.write_bytes(b"png")
    condition = runner.CONDITIONS["no-tools"]
    prompt_file = tmp_path / "p.txt"

    argv, _ = runner.build_codex(prompt_file, tmp_path, tmp_path, condition, "gpt-5", "high", 5,
                                 images=(image,))
    at = argv.index("-i")
    assert argv[at + 1] == str(image)
    assert argv[at + 2].startswith("--")
    assert argv[-1] == "-"
    bare, _ = runner.build_codex(prompt_file, tmp_path, tmp_path, condition, "gpt-5", "high", 5)
    assert "-i" not in bare

    assert f"@{image}" in runner.image_preface("antigravity", (image,))
    assert str(image) in runner.image_preface("claude", (image,))
    assert str(image) in runner.image_preface("codex", (image,))
    assert runner.image_preface("claude", ()) == ""


def test_an_attached_image_prefaces_the_prompt_and_is_recorded_in_the_runs_meta(
        runner, tmp_path, monkeypatch):
    """The preface sits ahead of the byte-identical question, the builder receives the resolved
    paths, and meta.json names the images so a later reader knows what the family saw."""
    monkeypatch.setattr(runner.subprocess, "run", _stub_run("an answer"))
    monkeypatch.setattr(runner, "harness_version", lambda *a, **k: "stub-version")
    seen = {}

    def builder(prompt_file, run_dir, cwd, condition, model, effort, timeout, images=()):
        seen["images"] = images
        return ["stub", "argv"], None

    monkeypatch.setitem(runner.BUILDERS, "codex", builder)
    image = tmp_path / "render.png"
    image.write_bytes(b"png")

    meta = runner.run_one("codex", "qid", "as-shipped", "question",
                          tmp_path, tmp_path / "out", 5, "stub-model", None, images=[image])

    assert meta["images"] == [str(image.resolve())]
    assert seen["images"] == (image.resolve(),)
    prompt = (tmp_path / "out" / "qid" / "as-shipped" / "codex" / "prompt.txt").read_text(
        encoding="utf-8")
    assert prompt.startswith("The image(s) attached to this prompt are ")
    assert prompt.endswith("question")


def test_a_missing_image_is_refused_before_any_family_launches(runner, tmp_path, monkeypatch):
    """A family told to review an image that is not there would answer about nothing; the
    runner refuses by name instead of launching."""
    launched = []
    monkeypatch.setattr(runner, "run_one", lambda *a, **k: launched.append(a))
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: "/stub/harness")
    (tmp_path / "q.txt").write_text("question", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "cross_family_ask.py", "--question-id", "qid",
        "--prompt-file", str(tmp_path / "q.txt"), "--families", "codex",
        "--image", str(tmp_path / "absent.png"),
        "--cwd", str(tmp_path), "--out", str(tmp_path / "out"),
    ])

    with pytest.raises(SystemExit) as refused:
        runner.main()
    assert refused.value.code != 0
    assert launched == []
