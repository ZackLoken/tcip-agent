"""``scripts/conform_metrics_marker.py``: the one-off conform step for a root whose experiments
logged epoch rows before ``log_metrics`` started stamping ``status.json["metrics_logged"]``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import tcip_store as ts

PY_EXE = sys.executable
SCRIPT = str(Path(__file__).parent.parent / "scripts" / "conform_metrics_marker.py")


def _run(root: Path):
    return subprocess.run([PY_EXE, SCRIPT, str(root)], capture_output=True, text=True, env=None)


def _experiment_with_a_logged_row(experiment_id: str) -> None:
    from tcip_mcp.experiments import create_experiment, log_metrics

    create_experiment(experiment_id, {"model_source": {"builder": "tests.bespoke_models:x"}})
    log_metrics(experiment_id, 1, {"train_loss": 0.5})


def _strip_marker(experiment_id: str, root: Path) -> None:
    """Manufacture the pre-change state: a status record with a logged row but no marker, the
    shape every experiment on this machine had before log_metrics started stamping it. This is
    the one legitimate direct store write in this file, since it recreates a fact the runtime no
    longer produces on its own."""
    from tcip_mcp.experiments import status_key

    key = status_key(experiment_id, root=root)
    with ts.transaction(key) as txn:
        status = txn.read(key, default={})
        status.pop("metrics_logged", None)
        txn.write(key, status)


def test_conforms_an_experiment_whose_marker_predates_the_change(tmp_path):
    root = Path(os.environ["TCIP_STATE_ROOT"])
    _experiment_with_a_logged_row("exp1")
    _strip_marker("exp1", root)

    from tcip_mcp.experiments import status_key
    assert "metrics_logged" not in ts.read(status_key("exp1", root=root))

    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert "conformed: exp1" in result.stdout

    assert ts.read(status_key("exp1", root=root))["metrics_logged"] is True


def test_an_already_conformed_experiment_is_left_alone(tmp_path):
    """A rail must admit valid work: an experiment log_metrics already marked is reported
    untouched, not re-stamped or refused."""
    root = Path(os.environ["TCIP_STATE_ROOT"])
    _experiment_with_a_logged_row("exp2")

    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert "conformed: exp2" not in result.stdout
    assert "0 experiment(s) stamped" in result.stdout


def test_an_experiment_with_no_rows_is_never_stamped(tmp_path):
    """A created-but-not-yet-trained experiment (no rows, no marker) stays pristine; the conform
    script only stamps the marker where the log already proves it should not read that way."""
    root = Path(os.environ["TCIP_STATE_ROOT"])
    from tcip_mcp.experiments import create_experiment, status_key

    create_experiment("exp3", {"model_source": {"builder": "tests.bespoke_models:x"}})

    result = _run(root)
    assert result.returncode == 0, result.stderr
    assert "conformed: exp3" not in result.stdout
    assert "metrics_logged" not in ts.read(status_key("exp3", root=root))
