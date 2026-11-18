"""tools/check_premise_cites.py: every premise block's header, its cites and the verdict it prints.

Every cite is read from a git repository the test commits, so `git show` is the real producer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools import check_premise_cites as cites

SOURCE = (
    "import os\n"  # 1
    "def alpha():\n"  # 2
    "    return beta_value\n"  # 3
    "\n"  # 4
    "def gamma():\n"  # 5
    "    return 1\n"  # 6
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
            "-C",
            str(repo),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _repository(root: Path, name: str = "repo") -> tuple[Path, str]:
    repo = root / name
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text(SOURCE, newline="\n")
    (repo / "b.txt").write_text("one\ntwo\nthree\n", newline="\n")
    (repo / "c.txt").write_text(
        "head still line one\ntarget_name\n",
        newline="\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "files")
    return repo, _git(repo, "rev-parse", "HEAD")


def _second_commit(repo: Path) -> str:
    (repo / "c.txt").write_text(
        "head still line one\nrenamed_target\nadded_name\n",
        newline="\n",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "rename and add")
    return _git(repo, "rev-parse", "HEAD")


def _block(commit: str, *premises: str) -> cites.PremiseBlock:
    bullets = "\n".join(f"- {p}" for p in premises)
    [block] = cites.parse_premise_blocks(
        f"Premises, read at {commit}:\n\n{bullets}\n"
    )
    return block


def _failures(
    repo: Path, commit: str, premise: str, **checkouts: Path
) -> list[str]:
    block = _block(commit, premise)
    return cites.check_premise(block.premises[0], block, checkouts, repo)


def test_a_one_commit_header_yields_its_commit_and_premise_cites() -> None:
    text = (
        "## Work order, premise half\n\n"
        "Premises, read at 0123abc:\n\n"
        "- at src/a.py:3 the code returns `beta_value`.\n"
        "- at src/b.py:4-9 the code does X, and at src/c.py:12 Y.\n\n"
        "Unplaced:\n\n- a later bullet that is not a premise\n"
    )

    [block] = cites.parse_premise_blocks(text)

    assert block.commits == {None: "0123abc"}
    assert len(block.premises) == 2
    assert cites.parse_cites(block.premises[0]) == [
        cites.Cite("src/a.py", 3, 3)
    ]
    assert cites.parse_cites(block.premises[1]) == [
        cites.Cite("src/b.py", 4, 9),
        cites.Cite("src/c.py", 12, 12),
    ]


def test_a_two_commit_labeled_header_yields_each_label_commit() -> None:
    text = (
        "Premises, read at tcip-agent 5807271 and claude-config db5350e. "
        "The mechanism lives in claude-config.\n\n"
        "- [claude-config] at x/y.md:30-31 the definition requires X.\n"
        "- [tcip-agent] at tools/README.md:3-5 each tool is run.\n\n"
        "Contradictions with the issue:\n\n- a bullet that is not a premise\n"
    )

    [block] = cites.parse_premise_blocks(text)

    assert block.commits == {
        "tcip-agent": "5807271",
        "claude-config": "db5350e",
    }
    assert block.premises == [
        "- [claude-config] at x/y.md:30-31 the definition requires X.",
        "- [tcip-agent] at tools/README.md:3-5 each tool is run.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Premises, read at the last release:\n\n- at a.py:1 X.\n",
        "Premises, read at 0123abc:\n\nNo bullets follow.\n",
        "Premises, read at 0123abc and a1b2c3d4:\n\n- at a.py:1 X.\n",
        "- at a.py:1 X, with no header.\n",
    ],
)
def test_a_block_with_no_commit_premise_or_header_is_refused(
    text: str,
) -> None:
    with pytest.raises(cites.PremiseBlockError):
        cites.parse_premise_blocks(text)


def test_a_cite_holding_its_identifier_passes(tmp_path: Path) -> None:
    repo, commit = _repository(tmp_path)

    assert _failures(repo, commit, "at a.py:3 returns `beta_value`.") == []


def test_a_cite_one_line_off_fails_naming_the_tokens(tmp_path: Path) -> None:
    repo, commit = _repository(tmp_path)

    failures = _failures(repo, commit, "at a.py:2 returns `beta_value`.")

    assert len(failures) == 1
    assert "none of beta_value appears in a.py:2-2" in failures[0]


def test_a_cite_past_the_files_end_fails(tmp_path: Path) -> None:
    repo, commit = _repository(tmp_path)

    failures = _failures(repo, commit, "at b.txt:2-4 the file says `two`.")

    assert len(failures) == 1
    assert "b.txt:2-4 is outside the file (3 lines)" in failures[0]


def test_a_path_absent_at_the_commit_fails(tmp_path: Path) -> None:
    repo, commit = _repository(tmp_path)

    failures = _failures(repo, commit, "at missing.py:1 the code does X.")

    assert len(failures) == 1
    assert "missing.py unreadable at" in failures[0]


def test_a_premise_with_no_backticked_identifier_passes_on_the_range(
    tmp_path: Path,
) -> None:
    repo, commit = _repository(tmp_path)

    failures = _failures(
        repo, commit, "at a.py:1-6 the code holds `42` and `(1, 2)`."
    )

    assert failures == []


def test_lines_are_counted_on_newlines_alone_as_git_counts_them(
    tmp_path: Path,
) -> None:
    repo, commit = _repository(tmp_path)

    holds = _failures(repo, commit, "at c.txt:2 names `target_name`.")
    past_end = _failures(repo, commit, "at c.txt:3 names `target_name`.")

    assert holds == []
    assert "c.txt:3-3 is outside the file (2 lines)" in past_end[0]


def test_backticked_call_expressions_are_judged_on_the_names_inside_them(
    tmp_path: Path,
) -> None:
    repo, commit = _repository(tmp_path)
    premise = (
        "at a.py:{} the code does `alpha(beta_value, k=None)` or `len(x)`."
    )

    holds = _failures(repo, commit, premise.format(3))
    off = _failures(repo, commit, premise.format(5))

    assert holds == []
    assert "none of alpha, beta_value, k, None, len, x appears" in off[0]


def test_a_wrapped_bullet_joins_its_lines_and_later_premises_are_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    text = (
        f"Premises, read at repo {commit}:\n\n"
        "- at a.py:3 the code returns\n  `beta_value`.\n"
        "- at a.py:2 the code returns `beta_value`.\n"
    )

    [block] = cites.parse_premise_blocks(text)
    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    assert block.premises[0] == "- at a.py:3 the code returns `beta_value`."
    assert len(block.premises) == 2
    assert exit_code == 1
    assert capsys.readouterr().out.endswith("FAIL\n")


def test_a_bare_line_cite_fails_as_not_in_the_writers_form(
    tmp_path: Path,
) -> None:
    repo, commit = _repository(tmp_path)

    failures = _failures(repo, commit, "the code at :12 does X.")

    assert failures == ["not in the writer's form: - the code at :12 does X."]


def test_a_labeled_repository_without_a_checkout_fails(
    tmp_path: Path,
) -> None:
    repo, commit = _repository(tmp_path)
    text = f"Premises, read at one {commit} and two {commit}:\n\n"
    [block] = cites.parse_premise_blocks(
        text + "- [two] at a.py:3 returns `beta_value`.\n"
    )

    failures = cites.check_premise(block.premises[0], block, {}, repo)
    with_checkout = cites.check_premise(
        block.premises[0], block, {"two": repo}, tmp_path
    )

    [single] = cites.parse_premise_blocks(
        f"Premises, read at one {commit}:\n\n- at a.py:3 returns `beta_value`.\n"
    )
    single_failures = cites.check_premise(single.premises[0], single, {}, repo)

    assert "no --checkout for repository 'two'" in failures[0]
    assert with_checkout == []
    assert "no --checkout for repository 'one'" in single_failures[0]


def test_main_exits_zero_on_an_all_passing_block_and_one_otherwise(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    header = f"Premises, read at repo {commit}:\n\n"
    passing = header + "- at a.py:3 returns `beta_value`.\n"
    failing = passing + "- at a.py:2 returns `beta_value`.\n"
    argv = ["12", "--checkout", f"repo={repo}"]

    passed = cites.main(argv, fetch=lambda issue: passing)
    passed_out = capsys.readouterr().out
    failed = cites.main(argv, fetch=lambda issue: failing)
    failed_out = capsys.readouterr().out
    unreadable = cites.main(argv, fetch=lambda issue: "no premises here")
    unreadable_out = capsys.readouterr().out

    passed_lines = passed_out.splitlines()
    assert passed == 0
    assert len(passed_lines) == 1
    assert passed_lines[0].startswith("PASS")
    words = "a premise that backticks any name has one or more of them on its cited lines"
    assert words in passed_lines[0]
    assert failed == 1
    assert failed_out.endswith("FAIL\n")
    assert "a.py:2-2" in failed_out
    assert unreadable == 1
    assert unreadable_out.endswith("FAIL\n")


def test_a_bad_cite_in_a_later_block_fails_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, first = _repository(tmp_path)
    second = _second_commit(repo)
    text = (
        f"Premises, read at repo {first}:\n\n"
        "- at a.py:3 returns `beta_value`.\n\n"
        "## Premise supplement\n\n"
        f"Premises, read at repo {second}:\n\n"
        "- at c.txt:4 names `added_name`.\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "c.txt:4-4 is outside the file (3 lines): - at c.txt:4" in out
    assert out.endswith("FAIL\n")


def test_each_block_is_read_at_its_own_commit_and_the_pass_counts_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, first = _repository(tmp_path)
    second = _second_commit(repo)
    text = (
        f"Premises, read at repo {first}:\n"
        "- at c.txt:2 names `target_name`.\n\n"
        f"Premises, read at repo {second}:\n"
        "- at c.txt:2 names `renamed_target`.\n"
        "- at c.txt:3 names `added_name`.\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out_lines = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(out_lines) == 1
    assert "2 premise blocks checked" in out_lines[0]
    assert "block 1 of" not in out_lines[0]


def _failure_line_of(out_lines: list[str], premise: str) -> str:
    [line] = [line for line in out_lines if premise in line]
    return line


def test_each_failing_premise_is_named_with_its_own_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, first = _repository(tmp_path)
    second = _second_commit(repo)
    first_header = f"Premises, read at repo {first}"
    second_header = f"Premises, read at repo {second}"
    first_premise = "- at c.txt:5 names `target_name`."
    second_premise = "- at c.txt:4 names `added_name`."
    text = (
        f"{first_header}:\n{first_premise}\n\n"
        f"{second_header}:\n{second_premise}\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out_lines = capsys.readouterr().out.splitlines()
    first_line = _failure_line_of(out_lines, first_premise)
    second_line = _failure_line_of(out_lines, second_premise)
    assert exit_code == 1
    assert out_lines[-1] == "FAIL"
    assert first_line.startswith(f'block 1 of 2, "{first_header}": ')
    assert second_line.startswith(f'block 2 of 2, "{second_header}": ')


def test_a_failing_premise_of_the_second_of_two_blocks_at_one_commit_is_named_second(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    header = f"Premises, read at repo {commit}"
    failing = "- at b.txt:9 the file says `two`."
    text = (
        f"{header}:\n- at a.py:3 returns `beta_value`.\n\n"
        f"{header}:\n{failing}\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out_lines = capsys.readouterr().out.splitlines()
    assert exit_code == 1
    assert len(out_lines) == 2
    assert out_lines[0].startswith(f'block 2 of 2, "{header}": ')
    assert failing in out_lines[0]


def test_a_header_whose_commit_is_on_the_next_line_is_named_on_one_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    failing = "- at c.txt:5 names `target_name`."
    text = f"Premises, read at\nrepo {commit}:\n{failing}\n"

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out_lines = capsys.readouterr().out.splitlines()
    line = _failure_line_of(out_lines, failing)
    assert exit_code == 1
    assert line.startswith(
        f'block 1 of 1, "Premises, read at repo {commit}": '
    )


def test_a_header_ending_in_a_period_and_a_line_break_keeps_its_first_bullet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    failing = "- at b.txt:9 the file says `two`."
    text = (
        f"Premises, read at repo {commit}.\n"
        f"{failing}\n"
        "- at a.py:3 returns `beta_value`.\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out = capsys.readouterr().out
    assert f"outside the file (3 lines): {failing}" in out
    assert exit_code == 1


def test_a_header_continued_onto_a_second_line_opens_a_checked_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, first = _repository(tmp_path)
    second = _second_commit(repo)
    failing = "- [other] at c.txt:4 names `added_name`."
    text = (
        f"Premises, read at repo {first}:\n- at a.py:3 returns `beta_value`.\n\n"
        f"Premises, read at repo {first} and\nother {second}:\n{failing}\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}", "--checkout", f"other={repo}"],
        fetch=lambda issue: text,
    )

    name = (
        f'block 2 of 2, "Premises, read at repo {first} and other {second}": '
    )
    out_lines = capsys.readouterr().out.splitlines()
    assert (
        f"{name}c.txt:4-4 is outside the file (3 lines): {failing}"
        in out_lines
    )
    assert exit_code == 1


def test_an_unterminated_header_line_followed_by_a_bullet_opens_no_block(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    text = (
        f"Premises, read at repo {commit}\n- at b.txt:9 the file says `two`.\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out = capsys.readouterr().out
    assert out.startswith("no 'Premises, read at <commit>:' header")
    assert exit_code == 1


def test_a_superseded_block_is_not_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, commit = _repository(tmp_path)
    text = (
        f"Superseded premises, read at repo {commit}:\n"
        "- at b.txt:9 a cite past the end of its file.\n\n"
        f"Premises, read at repo {commit}:\n"
        "- at a.py:3 returns `beta_value`.\n"
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out = capsys.readouterr().out.splitlines()
    assert exit_code == 0
    assert len(out) == 1
    assert "1 premise block checked" in out[0]


@pytest.mark.parametrize(
    ("later_block", "later_header"),
    [
        (
            "Premises, read at the last release:\n- at a.py:1 X.\n",
            "Premises, read at the last release",
        ),
        (
            "Premises, read at repo {commit} and repo {commit}:\n- at a.py:1 X.\n",
            "Premises, read at repo {commit} and repo {commit}",
        ),
        (
            "Premises, read at repo {commit}:\nNo bullets follow.\n",
            "Premises, read at repo {commit}",
        ),
    ],
)
def test_a_later_block_that_cannot_be_read_fails_the_run(
    later_block: str,
    later_header: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, commit = _repository(tmp_path)
    text = (
        f"Premises, read at repo {commit}:\n- at a.py:3 returns `beta_value`.\n\n"
        + later_block.format(commit=commit)
    )

    exit_code = cites.main(
        ["12", "--checkout", f"repo={repo}"], fetch=lambda issue: text
    )

    out = capsys.readouterr().out
    assert exit_code == 1
    assert out.endswith("FAIL\n")
    assert "PASS" not in out
    assert out.startswith(
        f'block 2 of 2, "{later_header.format(commit=commit)}": '
    )
