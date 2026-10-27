"""Records the pipelines write, each read back through its one declaration.

The plant registry, the source snapshot and a document scope are produced through the platform's
own producers and read back through their readers; a stored registry missing a field is refused
naming it. The dotted scorer and proposal factories are checked once for the class they build.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath

import pytest
from pydantic import ValidationError

import tcip_store as ts
from tcip_mcp.pipelines.postprocessing import plant_mapping
from tests._mapping_fixtures import (
    register_plant_registry_for,
    write_plots_csv,
)


def _registered(project: Path) -> str:
    """Register a two-plant CSV under ``project`` and return the registry's name."""
    csv_path = write_plots_csv(project / "plants.csv")
    return register_plant_registry_for(project, [csv_path])


def test_a_registered_plant_registry_reads_back_as_its_producer_wrote_it(
    tmp_path: Path,
) -> None:
    """The record ``register_plant_registry_record`` returns is the one ``load_registry`` reads."""
    csv_path = write_plots_csv(tmp_path / "plants.csv")

    written = plant_mapping.register_plant_registry_record(
        tmp_path, "rows", [csv_path], crop="currant", site="orchard"
    )

    assert plant_mapping.load_registry(tmp_path, "rows") == written
    assert written["csvs"][0]["sha256"] == (
        hashlib.sha256(csv_path.read_bytes()).hexdigest()
    )


def test_a_stored_plant_registry_missing_its_digest_is_refused_naming_it(
    tmp_path: Path,
) -> None:
    """A stored registry record without ``digest`` is refused by its reader, naming the field."""
    name = _registered(tmp_path)
    key = plant_mapping.plant_registry_key(tmp_path, name)
    stored = ts.read(key)
    assert isinstance(stored, dict)
    ts.replace(key, {k: v for k, v in stored.items() if k != "digest"})

    with pytest.raises(ValidationError, match="digest"):
        plant_mapping.load_registry(tmp_path, name)


def test_a_staged_source_snapshot_lays_out_each_copy_its_record_names(
    tmp_path: Path,
) -> None:
    """``run_layout`` places every copy the snapshot record names, at its recorded digest."""
    from tcip_mcp.pipelines.model_build import SNAPSHOT_DIR, staged_sources
    from tcip_mcp.pipelines.schemas import train_config
    from tests._chain_fixtures import (
        BESPOKE_MODELS,
        GT_ANCHOR_DETECTOR,
        training_config,
        weights_normalization,
    )
    from tests._training_values import drawn_split

    source = {
        "builder": GT_ANCHOR_DETECTOR,
        "task": "detection",
        "source_files": [BESPOKE_MODELS],
        "normalization": weights_normalization(),
    }
    spec = train_config(
        training_config(tmp_path, source, {"split": drawn_split()})
    )

    staged = staged_sources(spec, tmp_path)

    copies = staged.record["files"].values()
    assert {
        str(PurePosixPath(SNAPSHOT_DIR) / place)
        for place in staged.layout.places
    } == {copy["file"] for copy in copies}
    for copy in copies:
        data = (staged.layout.root.parent / copy["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == copy["sha256"]
        assert len(data) == copy["bytes"]


def test_a_document_scope_carries_the_subject_and_attributes_its_record_states() -> (
    None
):
    """A recorded document scope reads back as the subject and attributes the record states."""
    from tcip_mcp.pipelines.data.selection import ClassScope

    scope = ClassScope.of(
        {
            "scope": {
                "subject": "bud",
                "attributes": [
                    {
                        "name": "stage",
                        "type": "categorical",
                        "values": ["a", "b"],
                    }
                ],
            }
        }
    )

    read = scope.document_attributes("the test's scope")

    assert (read.subject, read.attributes) == (scope.subject, scope.attributes)


def test_a_document_scope_recording_no_attributes_is_refused_naming_them() -> (
    None
):
    """A scope naming a subject and recording no attributes is refused naming ``attributes``."""
    from tcip_mcp.pipelines.data.selection import ClassScope

    with pytest.raises(ValueError, match="records no attributes"):
        ClassScope(subject="bud").document_attributes("the test's scope")


class _NotAScorer:
    """A class no scorer block may build."""

    def __init__(self, task: str) -> None:
        self.task = task


def make_uncertainty_scorer(task: str) -> object:
    """Build an uncertainty scorer for ``task``, as an agent's dotted factory would."""
    from tcip_mcp.pipelines.active_learning.scorer import UncertaintyScorer

    return UncertaintyScorer(task)


def make_proposer() -> object:
    """Build a proposer whose every proposal is empty, as an agent's dotted factory would."""
    from tcip_mcp.pipelines.proposal import Proposal

    class _Empty:
        def propose(self, image_path: str, **params: object) -> Proposal:
            return Proposal([], None)

    return _Empty()


def make_nothing() -> object:
    """Build an object that proposes nothing and scores nothing."""
    return _NotAScorer("detection")


def test_a_dotted_scorer_factory_building_a_scorer_is_admitted() -> None:
    """A dotted factory that builds a ``BaseScorer`` resolves to that scorer."""
    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.active_learning.scorer import (
        UncertaintyScorer,
        resolve_scorer,
    )

    built = resolve_scorer(
        ScorerSpec(name=f"{__name__}:make_uncertainty_scorer"), "detection"
    )

    assert isinstance(built, UncertaintyScorer)


def test_a_dotted_scorer_factory_building_anything_else_is_refused_naming_it() -> (
    None
):
    """A dotted factory that builds no ``BaseScorer`` is refused naming the block's ``name``."""
    from tcip_mcp.pipelines.active_learning import ScorerSpec
    from tcip_mcp.pipelines.active_learning.scorer import resolve_scorer

    name = f"{__name__}:_NotAScorer"
    with pytest.raises(TypeError, match="_NotAScorer"):
        resolve_scorer(ScorerSpec(name=name), "detection")


def test_a_dotted_proposal_factory_building_a_proposer_is_admitted() -> None:
    """A dotted factory that builds a ``Proposer`` resolves to what it built."""
    from tcip_mcp.pipelines.proposal import (
        Proposal,
        Proposer,
        resolve_proposer,
    )

    built = resolve_proposer(f"{__name__}:make_proposer")

    assert isinstance(built, Proposer)
    assert built.propose("image.png") == Proposal([], None)


def test_a_dotted_proposal_factory_building_anything_else_is_refused_naming_it() -> (
    None
):
    """A dotted factory that builds no ``Proposer`` is refused naming the engine."""
    from tcip_mcp.pipelines.proposal import resolve_proposer

    with pytest.raises(TypeError, match="make_nothing"):
        resolve_proposer(f"{__name__}:make_nothing")
