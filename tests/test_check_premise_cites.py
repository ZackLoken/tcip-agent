"""tools/check_premise_cites.py: the premise half's header, its cites and the verdict it prints.

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
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "files")
    return repo, _git(repo, "rev-parse", "HEAD")


def _half(commit: str, *premises: str) -> cites.PremiseHalf:
    bullets = "\n".join(f"- {p}" for p in premises)
    return cites.parse_premise_half(
        f"Premises, read at {commit}:\n\n{bullets}\n"
    )


def _failures(
    repo: Path, commit: str, premise: str, **checkouts: Path
) -> list[str]:
    half = _half(commit, premise)
    return cites.check_premise(half.premises[0], half, checkouts, repo)


def test_a_one_commit_header_yields_its_commit_and_premise_cites() -> None:
    text = (
        "## Work order, premise half\n\n"
        "Premises, read at 0123abc:\n\n"
        "- at src/a.py:3 the code returns `beta_value`.\n"
        "- at src/b.py:4-9 the code does X, and at src/c.py:12 Y.\n\n"
        "Unplaced:\n\n- a later bullet that is not a premise\n"
    )

    half = cites.parse_premise_half(text)

    assert half.commits == {None: "0123abc"}
    assert len(half.premises) == 2
    assert cites.parse_cites(half.premises[0]) == [
        cites.Cite("src/a.py", 3, 3)
    ]
    assert cites.parse_cites(half.premises[1]) == [
        cites.Cite("src/b.py", 4, 9),
        cites.Cite("src/c.py", 12, 12),
    ]


def test_a_two_commit_labeled_header_yields_each_label_commit() -> None:
    text = (
        "Premises, read at tcip-agent a26495a and claude-config db5350e. "
        "The mechanism lives in claude-config.\n\n"
        "- [claude-config] at x/y.md:30-31 the definition requires X.\n"
        "- [tcip-agent] at tools/README.md:3-5 each tool is run.\n\n"
        "Contradictions with the issue:\n\n- a bullet that is not a premise\n"
    )

    half = cites.parse_premise_half(text)

    assert half.commits == {
        "tcip-agent": "a26495a",
        "claude-config": "db5350e",
    }
    assert half.premises == [
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
def test_a_half_with_no_commit_premise_or_header_is_refused(
    text: str,
) -> None:
    with pytest.raises(cites.PremiseHalfError):
        cites.parse_premise_half(text)


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
        repo, commit, 'at a.py:1-6 the code is `import os` and `"quoted"`.'
    )

    assert failures == []


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
    half = cites.parse_premise_half(
        text + "- [two] at a.py:3 returns `beta_value`.\n"
    )

    failures = cites.check_premise(half.premises[0], half, {}, repo)
    with_checkout = cites.check_premise(
        half.premises[0], half, {"two": repo}, tmp_path
    )

    assert "no --checkout for repository 'two'" in failures[0]
    assert with_checkout == []


def test_main_exits_zero_on_an_all_passing_half_and_one_otherwise(
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

    assert (passed, passed_out) == (0, "PASS\n")
    assert failed == 1
    assert failed_out.endswith("FAIL\n")
    assert "a.py:2-2" in failed_out
    assert unreadable == 1
    assert unreadable_out.endswith("FAIL\n")
