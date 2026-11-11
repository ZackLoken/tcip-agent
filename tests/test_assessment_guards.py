"""The assessment's disjointness, criterion, revision and gate checks.

The chain's own producers build the chain's records here: a capture ingested, a selection drawn,
a tiny detector trained on it, the trait confirmed, the assessment run and the bucket published. A
reference is re-sided only through the admission's own sample producer.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

import pytest

from tcip_mcp.dataset_layout import UNDATED_BUCKET, image_dir

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL import ImageDraw

    from tcip_mcp.pipelines.data.band_groups import BandGroupRef
    from tcip_mcp.pipelines.data.selection import Selection
    from tcip_mcp.pipelines.execution import Execution, MaskCut, Pass
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.traits import TraitEntry
    from tcip_mcp.workspace import BoundProject

torch = pytest.importorskip("torch")
pytest.importorskip("pycocotools")

from tcip_annotation import json_io  # noqa: E402
from tcip_annotation.state import Annotation, Polygon  # noqa: E402
from tcip_mcp.pipelines.execution import Stated  # noqa: E402
from tests import _trait_fixtures as fx  # noqa: E402
from tests._chain_fixtures import (  # noqa: E402
    DATE,
    IMG,
    SUBJECT,
    assess,
    confirm_count_trait,
    draw_reference_selection,
    object_at,
    run_the_chain,
    synthetic_capture,
    train_on,
)
from tests._dense_op_fixtures import dense_records, good_cal_holdout  # noqa: E402
from tests._producer_fixtures import (  # noqa: E402
    two_roots_refusal,
    unrooted_refusal,
)
from tests._training_values import (  # noqa: E402
    COUNT_TEST,
    DISPERSION_QUANTILE,
    EQUIVALENCE_Z,
    MASK_CUT,
    RATIONALE,
    collection_pass,
)
from tests._verified_checkpoint_fixtures import SAMPLE_DETECTOR_PASS  # noqa: E402


def _trained(
    project: Path, experiment_id: str
) -> tuple[Path, Selection, Path, str]:
    """Ingest a capture, draw its selection and train a checkpoint on the selection's train side.

    Returns ``(images_dir, selection, selection_dir, checkpoint_path)``.
    """
    root = project / "ds"
    images_dir = synthetic_capture(project, root)
    selection_dir = project / "selection"
    selection = draw_reference_selection(project, root, selection_dir)
    return (
        images_dir,
        selection,
        selection_dir,
        train_on(selection_dir, project, experiment_id),
    )


def _add_frame(
    images_dir: Path, stem: str, *, like: str, same_pixels: bool
) -> None:
    """Add frame ``stem`` labeled with frame ``like``'s stored annotations.

    Its image is a byte copy of ``like``'s when ``same_pixels``, else a frame painted on its own
    background with a square at :func:`object_at` of ``like``'s index.
    """
    from tests._producer_fixtures import (
        image_label_key,
        label_image,
        painted_frame,
    )

    if same_pixels:
        shutil.copyfile(images_dir / f"{like}.png", images_dir / f"{stem}.png")
    else:
        x0, y0, size = object_at(int(like[1:]))
        painted_frame(
            IMG,
            IMG,
            (90, 60, 30),
            [((x0, y0, x0 + size, y0 + size), (250, 240, 200))],
        ).save(images_dir / f"{stem}.png")
    label_image(
        images_dir / f"{stem}.png",
        json_io.read_label_document(
            image_label_key(images_dir / f"{like}.png")
        ).annotations,
        IMG,
        IMG,
    )


def _resided(
    project: Path,
    images_dir: Path,
    drawn: Selection,
    extra: dict[str, str],
    out: Path,
) -> Path:
    """Write a reference selection of its own at ``out`` and return ``out``.

    It holds the drawn selection's calibration and holdout sides with the added frames ``extra``
    (member to side) joined to them, sided and grouped by the admission's own sample producer
    under the draw's own policy.
    """
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.selection import (
        REFERENCE_SIDES,
        Selection,
        write_selection,
    )
    from tcip_mcp.pipelines.data.splits import recorded_group_key_fn

    admitted = admit(images_dir, scope=drawn.scope)
    sides = {
        s.member: s.side for s in drawn.samples if s.side in REFERENCE_SIDES
    } | extra
    samples = admitted.samples(
        sides, recorded_group_key_fn(drawn.group_by, date=admitted.date)
    )
    write_selection(
        out,
        Selection(
            samples=tuple(samples),
            scope=drawn.scope,
            seed=drawn.seed,
            group_by=drawn.group_by,
            ratios=drawn.ratios,
            stratify_foreground=drawn.stratify_foreground,
            rationale=drawn.rationale,
            dataset_fingerprint=drawn.dataset_fingerprint,
        ),
        project=project,
    )
    return out


def _a_train_member(selection: Selection) -> str:
    return next(s.member for s in selection.samples if s.side == "train")


# -- identity: content, never annotation geometry --------------------------------------


def test_a_distinct_image_holding_a_training_frames_boxes_is_a_legitimate_reference(
    tmp_path: Path,
) -> None:
    """A distinct image holding a training frame's boxes is a disjoint reference.

    Two different images can carry identical boxes: the second is not the first, so a reference
    holding it is held out. Identity is the image's content, never the geometry drawn on it.
    """
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-twin")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    _add_frame(images_dir, "twin", like=like, same_pixels=False)
    reference = _resided(
        tmp_path,
        images_dir,
        drawn,
        {"twin": "holdout"},
        tmp_path / "reference",
    )

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    for side in ("training", "selection"):
        assert record["disjointness"][side] == {
            "groups": [],
            "source_digests": [],
        }, record
    assert "reference_shares_training" not in record["failures"]


def test_a_training_image_copied_under_a_new_name_into_the_reference_fails_disjointness(
    tmp_path: Path,
) -> None:
    """A training image copied under a new name into the reference fails disjointness.

    A reference holding a byte copy of a training frame under another name fails with
    ``reference_shares_training``.
    """
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-dup")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    _add_frame(images_dir, "dup", like=like, same_pixels=True)
    reference = _resided(
        tmp_path, images_dir, drawn, {"dup": "holdout"}, tmp_path / "reference"
    )

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    assert "reference_shares_training" in record["failures"], record[
        "failures"
    ]
    assert record["disjointness"]["training"]["groups"] == [], record[
        "disjointness"
    ]
    assert len(record["disjointness"]["training"]["source_digests"]) == 1, (
        record["disjointness"]
    )


def test_a_reference_frame_in_a_training_group_fails_even_when_its_pixels_are_its_own(
    tmp_path: Path,
) -> None:
    """A reference frame in a training group fails disjointness on the group alone.

    A holdout frame ``<member>_5_5`` of its own pixels, grouped by tile prefix with a training
    member, fails with ``reference_shares_training``: one shared group and no shared source
    digest.
    """
    images_dir, drawn, _sel, checkpoint = _trained(tmp_path, "exp-group")
    confirm_count_trait(tmp_path)
    like = _a_train_member(drawn)
    sibling = f"{like}_5_5"
    _add_frame(images_dir, sibling, like=like, same_pixels=False)
    reference = _resided(
        tmp_path,
        images_dir,
        drawn,
        {sibling: "holdout"},
        tmp_path / "reference",
    )

    record = assess(tmp_path, checkpoint, reference)

    assert "error" not in record, record
    assert "reference_shares_training" in record["failures"], record[
        "failures"
    ]
    assert record["disjointness"]["training"]["source_digests"] == [], record[
        "disjointness"
    ]
    assert len(record["disjointness"]["training"]["groups"]) == 1, record[
        "disjointness"
    ]


def test_a_foreign_checkpoint_fails_its_assessment_on_unknown_training_membership(
    tmp_path: Path,
) -> None:
    """A checkpoint no run of the project produced fails on its unknown training membership.

    A checkpoint registered through ``register_model``'s explicit mode is assessed over the
    project's reference: no run's completion registered it, so nothing states what it was trained
    on, the record fails with ``training_membership_unknown`` and its disjointness states the
    reference's two sides alone.
    """
    from tests._verified_checkpoint_fixtures import foreign_checkpoint

    root = tmp_path / "ds"
    synthetic_capture(tmp_path, root)
    selection_dir = tmp_path / "selection"
    draw_reference_selection(tmp_path, root, selection_dir)
    confirm_count_trait(tmp_path)
    checkpoint = foreign_checkpoint(tmp_path, name="foreign")

    record = assess(tmp_path, checkpoint, selection_dir)

    assert "error" not in record, record["error"]
    assert "training_membership_unknown" in record["failures"], record[
        "failures"
    ]
    disjointness = record["disjointness"]
    assert (disjointness["training"], disjointness["selection"]) == (
        None,
        None,
    ), disjointness


NEXT_STEP = "Draw it over one with draw_splits."
"""The step a selection's refusal of its samples' root names after the root reading's sentence."""


def test_a_selection_spanning_two_dataset_roots_refuses_naming_both(
    tmp_path: Path,
) -> None:
    """A selection whose samples lie under two dataset roots refuses at its write, naming both.

    A reference drawn over one dataset is joined, through the admission's own sample producer, by
    holdout frames of a second dataset; writing that selection refuses with the one root
    reading's sentence for its samples, both roots in their canonical spelling, followed by the
    draw_splits step, and the drawn selection stays as it was written.
    """
    from tcip_mcp.pipelines.data.label_queries import admit
    from tcip_mcp.pipelines.data.selection import (
        REFERENCE_SIDES,
        Selection,
        read_selection,
        selection_place,
        write_selection,
    )
    from tcip_mcp.pipelines.data.splits import recorded_group_key_fn
    from tests._producer_fixtures import (
        label_image,
        registry_of,
        registry_over,
        write_image,
    )
    from tests.test_physical_scale_assessment import (
        DATE as SCALE_DATE,
        SUBJECT as BAR,
        _rect,
        _reference,
    )

    fx.seed_delivery_traits(tmp_path)
    selection_dir, _csv_path = _reference(tmp_path)
    other = tmp_path / "ds2"
    images = image_dir(other, SCALE_DATE)
    images.mkdir(parents=True)
    registry_over(other, registry_of(BAR))
    for i in range(2):
        label_image(
            write_image(images / f"s{i}.png", (120, 120), (200, 20 + i, 20)),
            [Annotation(subject=BAR, geometry=Polygon(rings=[_rect(100.0)]))],
            120,
            120,
        )
    drawn = read_selection(selection_dir, project=tmp_path)
    joined = admit(images, scope=drawn.scope)
    spanning = Selection(
        samples=(
            *(s for s in drawn.samples if s.side in REFERENCE_SIDES),
            *joined.samples(
                {"s0": "holdout", "s1": "holdout"},
                recorded_group_key_fn(drawn.group_by, date=joined.date),
            ),
        ),
        scope=drawn.scope,
        seed=drawn.seed,
        group_by=drawn.group_by,
        ratios=drawn.ratios,
        stratify_foreground=drawn.stratify_foreground,
        rationale=drawn.rationale,
        dataset_fingerprint=drawn.dataset_fingerprint,
    )

    with pytest.raises(ValueError) as refused:
        write_selection(selection_dir, spanning, project=tmp_path)

    assert str(refused.value) == (
        f"{selection_place(selection_dir)}: "
        f"{two_roots_refusal([tmp_path / 'ds', other], 'its samples')} {NEXT_STEP}"
    ), refused.value
    assert "bucket" not in str(refused.value), refused.value
    assert read_selection(selection_dir, project=tmp_path) == drawn


def test_a_selection_with_a_sample_under_no_image_tree_refuses_at_its_write(
    tmp_path: Path,
) -> None:
    """A selection holding a sample under no dataset image tree refuses at its write.

    A drawn reference's first sample is re-pointed at a copy of its image outside any image tree;
    writing that selection refuses with the no-image-tree sentence for its samples, naming the
    copy, followed by the draw_splits step, and nothing is written at the new place.
    """
    from dataclasses import replace

    from tcip_mcp.pipelines.data.selection import (
        read_selection,
        selection_place,
        write_selection,
    )
    from tests.test_physical_scale_assessment import _reference

    selection_dir, _csv_path = _reference(tmp_path)
    drawn = read_selection(selection_dir, project=tmp_path)
    first, *rest = drawn.samples
    loose = tmp_path / "loose" / Path(first.source).name
    loose.parent.mkdir()
    shutil.copyfile(first.source, loose)
    out = tmp_path / "rewritten"

    with pytest.raises(ValueError) as refused:
        write_selection(
            out,
            replace(drawn, samples=(replace(first, source=str(loose)), *rest)),
            project=tmp_path,
        )

    assert str(refused.value) == (
        f"{selection_place(out)}: {unrooted_refusal(loose, 'its samples')} {NEXT_STEP}"
    ), refused.value
    assert "bucket" not in str(refused.value), refused.value
    assert not out.exists()


def test_a_selection_of_a_capture_and_its_roots_image_tree_writes_and_reads_back(
    tmp_path: Path,
) -> None:
    """A selection of one capture's samples and one directly in its root's image tree is one root's.

    A drawn reference's first sample is re-pointed at a copy of its image directly under the same
    root's ``images``; the selection writes and reads back as written.
    """
    from dataclasses import replace

    from tcip_mcp.dataset_layout import image_root
    from tcip_mcp.pipelines.data.selection import (
        read_selection,
        write_selection,
    )
    from tests.test_physical_scale_assessment import _reference

    selection_dir, _csv_path = _reference(tmp_path)
    drawn = read_selection(selection_dir, project=tmp_path)
    first, *rest = drawn.samples
    in_tree = image_root(tmp_path / "ds") / Path(first.source).name
    shutil.copyfile(first.source, in_tree)
    mixed = replace(
        drawn, samples=(replace(first, source=str(in_tree)), *rest)
    )
    out = tmp_path / "mixed"

    write_selection(out, mixed, project=tmp_path)

    assert read_selection(out, project=tmp_path) == mixed


@pytest.mark.parametrize(
    "registered",
    [("root",), ("tree",)],
    ids=["outer_root_carries_state", "inner_tree_carries_state"],
)
def test_a_selection_over_a_directory_named_images_writes_and_reads_back_with_one_root_registered(
    tmp_path: Path, registered: tuple[str, ...]
) -> None:
    """A selection over a directory named ``images`` writes and reads back with one root registered.

    Every sample of a drawn reference is re-pointed at a copy of its image in
    ``<grove>/images/images``; with ``<grove>`` registered the directory is its capture ``images``,
    and with ``<grove>/images`` registered it is that root's image tree. Either way the selection
    writes through ``write_selection`` and reads back as written.
    """
    from tcip_mcp.pipelines.data.selection import (
        read_selection,
        write_selection,
    )
    from tests.test_physical_scale_assessment import _reference
    from tests.test_web_inference_launch_root import _capture_named_images

    selection_dir, _csv_path = _reference(tmp_path)
    relocated, _copies = _copied_elsewhere(
        tmp_path,
        selection_dir,
        lambda i, name: Path("grove", "images", "images", f"{i}-{name}"),
    )
    _capture_named_images(tmp_path, registered=registered)
    out = tmp_path / "relocated"

    write_selection(out, relocated, project=tmp_path)

    assert read_selection(out, project=tmp_path) == relocated


def _assessed_with_no_checkpoint_file(
    project: Path, selection_dir: Path, *, confirmed: bool
) -> dict:
    """Return ``assess_checkpoint``'s arguments over ``selection_dir`` naming no checkpoint file.

    The assessment is for the count trait's per-image count under the sample collection pass and
    count test settings; the trait is confirmed first (:func:`confirm_count_trait`) when
    ``confirmed``, and otherwise left as the project holds it.
    """
    if confirmed:
        confirm_count_trait(project)
    return {
        "checkpoint_path": str(project / "absent.pt"),
        "trait": fx.COUNT_TRAIT,
        "delivery_kind": "per_image_count",
        "selection_dir": str(selection_dir),
        "equivalence_z": EQUIVALENCE_Z,
        "dispersion_quantile": DISPERSION_QUANTILE,
        "rationale": RATIONALE,
        "stated": collection_pass().model_dump(
            mode="json", exclude_unset=True
        ),
    }


@pytest.mark.parametrize(
    "confirmed", [True, False], ids=["trait_confirmed", "trait_unconfirmed"]
)
def test_an_assessment_under_a_root_never_registered_refuses_before_its_checkpoint_is_read(
    tmp_path: Path, bound: BoundProject, confirmed: bool
) -> None:
    """An assessment over a root never registered refuses before its trait or checkpoint is read.

    The selection is drawn over a capture ingested and never registered, the checkpoint path
    names no file, and the trait is confirmed or has no revision. Called through the server, the
    answer is the one sentence ``require_dataset_identity`` refuses the root with, and nothing is
    recorded or logged; registered, the same call meets the checkpoint's own refusal, or the
    trait's when it has no revision.
    """
    from tests._mapping_fixtures import COUNT_CROP
    from tests._producer_fixtures import (
        left_written,
        register_dataset_at,
        unregistered_refusal,
    )
    from tests._server_calls import called_through_the_server

    root = tmp_path / "ds"
    synthetic_capture(tmp_path, root, crop=None)
    selection_dir = tmp_path / "selection"
    draw_reference_selection(tmp_path, root, selection_dir)
    arguments = _assessed_with_no_checkpoint_file(
        tmp_path, selection_dir, confirmed=confirmed
    )
    before = left_written(tmp_path, root)

    refused = called_through_the_server(
        "assess_checkpoint", arguments, bound=bound
    )

    assert refused == {"error": unregistered_refusal(root)}, refused
    assert left_written(tmp_path, root) == before
    register_dataset_at(tmp_path, root, crop=COUNT_CROP)
    answer = called_through_the_server(
        "assess_checkpoint", arguments, bound=bound
    )
    assert isinstance(answer, dict), answer
    met = arguments["checkpoint_path"] if confirmed else fx.COUNT_TRAIT
    assert met in answer["error"], answer


@pytest.mark.parametrize(
    "confirmed", [True, False], ids=["trait_confirmed", "trait_unconfirmed"]
)
def test_an_assessment_reads_its_selection_before_its_checkpoint(
    tmp_path: Path, bound: BoundProject, confirmed: bool
) -> None:
    """An assessment naming no recorded selection meets it before its trait or checkpoint.

    The checkpoint path names no file, and the trait is confirmed or has no revision. Called
    through the server, the answer is the selection's refusal, naming its directory.
    """
    from tcip_mcp.pipelines.data.selection import selection_place
    from tests._server_calls import called_through_the_server

    selection_dir = tmp_path / "selection"

    refused = called_through_the_server(
        "assess_checkpoint",
        _assessed_with_no_checkpoint_file(
            tmp_path, selection_dir, confirmed=confirmed
        ),
        bound=bound,
    )

    assert isinstance(refused, dict), refused
    assert selection_place(selection_dir) in refused["error"], refused
    assert "no selection recorded there" in refused["error"], refused


def _copied_elsewhere(
    project: Path, selection_dir: Path, place: Callable[[int, str], Path]
) -> tuple[Selection, list[Path]]:
    """Return the selection at ``selection_dir`` with each sample's image copied elsewhere.

    Sample ``i``'s image is copied to ``project / place(i, its file name)`` and the sample names
    that copy as its source, every other field the drawn selection's own. Returns the selection
    and the copies, in sample order.
    """
    from dataclasses import replace

    from tcip_mcp.pipelines.data.selection import read_selection

    drawn = read_selection(selection_dir, project=project)
    copies = []
    for i, sample in enumerate(drawn.samples):
        copy = project / place(i, Path(sample.source).name)
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sample.source, copy)
        copies.append(copy)
    relocated = tuple(
        replace(s, source=str(c))
        for s, c in zip(drawn.samples, copies, strict=True)
    )
    return replace(drawn, samples=relocated), copies


def _relocated(
    project: Path,
    selection_dir: Path,
    place: Callable[[int, str], Path],
    out: Path,
) -> list[Path]:
    """Store at ``out`` the selection at ``selection_dir``, each sample's image copied elsewhere.

    The selection is :func:`_copied_elsewhere`'s. The document is the one
    ``stored_selection_document`` stores, written through the store past ``write_selection``, a
    fault injected past the selection's writer. Returns the copies, in sample order.
    """
    import tcip_store
    from tcip_mcp.pipelines.data.selection import (
        selection_key,
        stored_selection_document,
    )

    relocated, copies = _copied_elsewhere(project, selection_dir, place)
    tcip_store.replace(
        selection_key(out), stored_selection_document(relocated, project)
    )
    return copies


def _layout_refusal(directory: Path) -> str:
    """Return the sentence ``parse_capture_dir`` refuses the ambiguous ``directory`` with."""
    from tcip_mcp.dataset_layout import parse_capture_dir

    with pytest.raises(ValueError, match="register_dataset") as refused:
        parse_capture_dir(directory)
    return str(refused.value)


@pytest.mark.parametrize(
    ("place", "sentence", "step", "registered"),
    [
        (
            lambda i, name: Path("loose", f"{i}-{name}"),
            lambda copies: unrooted_refusal(copies[0], "its samples"),
            f" {NEXT_STEP}",
            None,
        ),
        (
            lambda i, name: Path(
                "a" if i == 0 else "b", "images", f"{i}-{name}"
            ),
            lambda copies: two_roots_refusal(
                {c.parent.parent for c in copies}, "its samples"
            ),
            f" {NEXT_STEP}",
            None,
        ),
        (
            lambda i, name: Path("grove", "images", "images", f"{i}-{name}"),
            lambda copies: _layout_refusal(copies[0].parent),
            "",
            (),
        ),
        (
            lambda i, name: Path("grove", "images", "images", f"{i}-{name}"),
            lambda copies: _layout_refusal(copies[0].parent),
            "",
            ("root", "tree"),
        ),
    ],
    ids=[
        "under_no_image_tree",
        "under_two_image_trees",
        "in_a_directory_neither_root_claims",
        "in_a_directory_both_roots_claim",
    ],
)
@pytest.mark.parametrize(
    "door",
    [
        "read_selection",
        "assess_checkpoint",
        "calibrate_physical_scale",
        "bound_launch",
        "doctor",
    ],
)
def test_a_selection_whose_samples_lie_under_no_one_root_refuses_at_its_readers(
    tmp_path: Path,
    bound: BoundProject,
    door: str,
    place: Callable[[int, str], Path],
    sentence: Callable[[list[Path]], str],
    step: str,
    registered: tuple[str, ...] | None,
) -> None:
    """A stored selection whose samples lie under no one root refuses at its reads.

    Each sample of a drawn selection names its own copy of its image: in no
    image tree, the first sample under one root's image tree and every other under a second's, or
    all in a capture named ``images`` that neither or both of its two readings' roots carry state
    for (``registered``, as the launch route's producer makes the layout), stored past the
    selection's writer (:func:`_relocated`). Each reader answers the selection's own refusal: its
    place and the sentence for its samples, followed by the draw_splits step for the first two,
    which a redraw answers; the third and fourth name register_dataset instead and no draw_splits.
    ``read_selection`` raises it; every other reader is a tool called through a client: the two
    assessment doors answer it as their error, ``preflight_config`` over a config bound to the
    selection (made over a detection selection before it is moved) answers it among its
    ``issues`` and not ``valid``, and ``diagnose_project`` answers it as an error finding over the
    project's ``splits``.
    """
    from tcip_mcp.pipelines.data.selection import (
        read_selection,
        selection_place,
    )
    from tests._chain_fixtures import training_config
    from tests._server_calls import called_through_the_server
    from tests._verified_checkpoint_fixtures import unbuilt_source
    from tests.test_doctor import _answered
    from tests.test_physical_scale_assessment import (
        SUBJECT as BAR,
        _author_tolerance,
        _reference,
    )
    from tests.test_selection_binding import (
        _draw,
        _two_subject_two_date_dataset,
    )
    from tests.test_web_inference_launch_root import _capture_named_images

    if door == "bound_launch":
        drawn = tmp_path / "drawn"
        _draw(tmp_path, _two_subject_two_date_dataset(tmp_path / "ds"), drawn)
        launch = training_config(
            tmp_path,
            unbuilt_source("detection"),
            {"split": {"selection_dir": str(drawn)}},
        )
    else:
        drawn, csv_path = _reference(tmp_path)
    selection_dir = tmp_path / "splits" / "relocated"
    copies = _relocated(tmp_path, drawn, place, selection_dir)
    if registered is not None:
        _capture_named_images(tmp_path, registered=registered)
    if door == "assess_checkpoint":
        served = called_through_the_server(
            door,
            _assessed_with_no_checkpoint_file(
                tmp_path, selection_dir, confirmed=True
            ),
            bound=bound,
        )
        assert isinstance(served, dict), served
        answer = served["error"]
    elif door == "calibrate_physical_scale":
        fx.seed_delivery_traits(tmp_path)
        _author_tolerance(tmp_path)
        served = called_through_the_server(
            door,
            {
                "trait": "plant_surface_area",
                "selection_dir": str(selection_dir),
                "reference_csv": str(csv_path),
                "unit": "mm",
                "reference_subject": BAR,
            },
            bound=bound,
        )
        assert isinstance(served, dict), served
        answer = served["error"]
    elif door == "read_selection":
        with pytest.raises(ValueError) as refused:
            read_selection(selection_dir, project=tmp_path)
        answer = str(refused.value)
    expected = f"{selection_place(selection_dir)}: {sentence(copies)}{step}"
    if door == "bound_launch":
        launch["data"]["split"]["selection_dir"] = str(selection_dir)
        served = called_through_the_server(
            "preflight_config", {"config": launch}, bound=bound
        )
        assert isinstance(served, dict), served
        assert served["valid"] is False, served
        issues = served["issues"]
        assert isinstance(issues, list) and expected in issues, served
        answer = expected
    elif door == "doctor":
        found = _answered(bound)
        finding = f"selection will not read: {expected}"
        assert ("error", finding) in found, found
        answer = finding
    else:
        assert answer == expected, answer
    assert isinstance(answer, str), answer
    assert "bucket" not in answer, answer
    assert ("draw_splits" in answer) == bool(step), answer
    assert ("register_dataset" in answer) == (not step), answer


def test_an_assessments_reference_records_the_id_its_root_is_registered_under(
    tmp_path: Path, bound: BoundProject
) -> None:
    """An assessment's reference records the id ``read_dataset_identity`` reads of its root.

    The assessment is ``assess_checkpoint``'s, called through the server over a trained detector's
    reference selection.
    """
    from tcip_mcp.dataset_layout import read_dataset_identity
    from tests._chain_fixtures import trained_detector
    from tests._server_calls import called_through_the_server

    root, _images, selection_dir, checkpoint = trained_detector(
        tmp_path, experiment_id="exp-reference-id"
    )
    arguments = _assessed_with_no_checkpoint_file(
        tmp_path, selection_dir, confirmed=False
    )

    assessed = called_through_the_server(
        "assess_checkpoint",
        {**arguments, "checkpoint_path": checkpoint},
        bound=bound,
    )

    identity = read_dataset_identity(root)
    assert identity is not None
    assert isinstance(assessed, dict) and "error" not in assessed, assessed
    reference = assessed["reference"]
    assert isinstance(reference, dict), reference
    assert reference["dataset_id"] == identity.id


def test_a_mosaic_reference_band_is_held_out_only_inside_a_non_training_region() -> (
    None
):
    """A mosaic reference rect is held out only inside a non-training region.

    Against a within-image split's recorded regions, a reference rect is held out when it lies
    wholly inside one non-training region; one straddling the training boundary is not.
    """
    from tcip_mcp.pipelines.operating_point import spatial_disjointness
    from tcip_mcp.pipelines.schemas import SpatialManifest

    regions = {
        "train_region": [(0, 0, 500, 1000)],
        "val_region": [(500, 0, 650, 1000)],
        "calibration_region": [(650, 0, 800, 1000)],
        "holdout_region": [(800, 0, 1000, 1000)],
    }
    spatial = SpatialManifest.model_construct(**regions)

    assert (
        spatial_disjointness(
            spatial,
            [(550, 100, 600, 300), (680, 100, 780, 300), (850, 100, 950, 300)],
        )
        == []
    )
    assert spatial_disjointness(spatial, [(400, 100, 600, 300)]) == [
        "[400, 100, 600, 300]"
    ]
    # A manifest not stating its regions whole refuses where it is read back.
    with pytest.raises(ValueError, match="calibration_region"):
        SpatialManifest.model_validate(
            {k: v for k, v in regions.items() if k != "calibration_region"}
        )


# -- the count and classifier criteria ------------------------------------------------


def _count_entry(**fields: object) -> TraitEntry:
    return fx.with_fields(fx.COUNT_SPEC, **fields)


def test_an_accurate_reference_passes_the_count_criterion() -> None:
    from tcip_mcp.pipelines.operating_point import count_criterion

    cal, hold = good_cal_holdout()
    _conf, _evidence, failures = count_criterion(
        cal,
        hold,
        _count_entry(count_error_tolerance=2.0),
        staged_conf_floor=0.01,
        **COUNT_TEST,
    )

    assert failures == [], failures


def test_a_known_count_bias_fails_the_outcome_tolerance_with_no_dispersion_at_all() -> (
    None
):
    """A known count bias with no dispersion fails the outcome tolerance.

    Every held-out image over-counts by the same ten objects: the bias is certain, its spread
    zero, and the count is still wrong.
    """
    from tcip_mcp.pipelines.operating_point import count_criterion

    cal, _ = good_cal_holdout()
    hold = dense_records(
        id_prefix="h", miss_pattern=[0] * 20, fp_pattern=[10] * 20, score=0.9
    )
    _conf, evidence, failures = count_criterion(
        cal,
        hold,
        _count_entry(count_error_tolerance=20.0),
        staged_conf_floor=0.01,
        **COUNT_TEST,
    )

    assert evidence.holdout_at_conf["count_bias_std_present"] == 0.0
    assert "count_bias_exceeds_tolerance" in failures, failures


def _items(image: str, calls: list[tuple[bool, bool]]) -> list[dict]:
    return [
        {"image_id": image, "is_true_positive": t, "is_pred_positive": p}
        for t, p in calls
    ]


def test_an_agreeing_classifier_passes_the_classifier_criterion() -> None:
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    calls = [(True, True), (True, True), (False, False), (False, False)]
    hold = _items("a", calls) + _items("b", calls)
    evidence, failures = classifier_criterion(
        hold, hold, _count_entry(), equivalence_z=EQUIVALENCE_Z
    )

    assert failures == [], (failures, evidence)


def test_compensating_classification_errors_cannot_validate_through_pooled_counts() -> (
    None
):
    """Compensating classification errors fail the classifier agreement floor.

    Each image's two positives called negative and two negatives called positive: its positive
    count is exactly right, and every call is wrong.
    """
    from tcip_mcp.pipelines.operating_point import classifier_criterion

    calls = [(True, False), (True, False), (False, True), (False, True)]
    hold = _items("a", calls) + _items("b", calls)
    evidence, failures = classifier_criterion(
        hold, hold, _count_entry(), equivalence_z=EQUIVALENCE_Z
    )

    assert evidence.positive_count_bias == 0.0
    assert "classifier_agreement_below_floor" in failures, (failures, evidence)


# -- the trait revision -----------------------------------------------------------------


def test_an_unauthored_tolerance_refuses_the_revision_with_the_breeders_question(
    tmp_path: Path,
) -> None:
    """A count revision leaving a tolerance unauthored is refused with the breeder's question.

    With ``count_bias_tolerance_frac`` unset, the proposal raises ``UnauthoredFieldError`` naming
    the field and its question, and the trait is not recorded.
    """
    from tcip_mcp.traits import QUESTIONS, UnauthoredFieldError, trait_names

    with pytest.raises(UnauthoredFieldError) as refused:
        confirm_count_trait(tmp_path, count_bias_tolerance_frac=None)

    assert "count_bias_tolerance_frac" in str(refused.value)
    assert QUESTIONS["count_bias_tolerance_frac"] in str(refused.value)
    assert fx.COUNT_TRAIT not in trait_names(tmp_path)


def test_an_iou_match_operationalization_is_proposed_only_with_its_jitter_and_margin(
    tmp_path: Path,
) -> None:
    """An ``iou_match`` proposal is refused without its jitter and margin and recorded with them.

    The fields an IoU match compares by are part of the criterion a count operationalization
    rests on: a proposal selecting ``iou_match`` without them is refused asking the breeder, and
    one stating them is recorded.
    """
    from tcip_mcp.traits import QUESTIONS, UnauthoredFieldError, trait_names

    with pytest.raises(UnauthoredFieldError) as refused:
        confirm_count_trait(tmp_path, localization="iou_match")
    assert "iou_jitter_px" in str(refused.value) and "iou_margin" in str(
        refused.value
    )
    assert QUESTIONS["iou_jitter_px"] in str(refused.value)
    assert fx.COUNT_TRAIT not in trait_names(tmp_path)

    confirmed = confirm_count_trait(
        tmp_path,
        localization="iou_match",
        iou_jitter_px=fx.IOU_JITTER_PX,
        iou_margin=fx.IOU_MARGIN,
    )
    assert confirmed.state.kind == "confirmed"
    assert (confirmed.entry.iou_jitter_px, confirmed.entry.iou_margin) == (
        fx.IOU_JITTER_PX,
        fx.IOU_MARGIN,
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("iou_jitter_px", -3.0),
        ("iou_jitter_px", "nan"),
        ("iou_margin", -0.1),
        ("iou_margin", "inf"),
    ],
)
def test_an_iou_field_that_is_no_finite_nonnegative_value_refuses(
    field: str, value: float | str
) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match=field):
        fx.with_fields(fx.COUNT_SPEC, **{field: float(value)})


def test_a_delivery_under_another_revision_than_the_assessments_refuses_naming_both(
    tmp_path: Path,
) -> None:
    from tcip_mcp.tools.inference_tools import deliver_per_image_counts

    chain = run_the_chain(tmp_path, experiment_id="exp-revision")
    confirm_count_trait(tmp_path, count_error_tolerance=3.0)
    out_csv = tmp_path / "counts.csv"

    delivered = deliver_per_image_counts(
        tmp_path,
        str(chain.root),
        chain.bucket,
        str(out_csv),
        trait=fx.COUNT_TRAIT,
        actor=None,
    )

    assert "error" in delivered, delivered
    assert (
        "revision 1" in delivered["error"]
        and "revision 2" in delivered["error"]
    ), delivered
    assert not out_csv.exists()


# -- the one gate over the producers ------------------------------------------------------


def test_one_producer_clears_the_gate_and_a_series_of_two_refuses(
    tmp_path: Path,
) -> None:
    """The gate clears one producer's bucket and refuses a series of two, naming each.

    Two checkpoints publish over one capture: the assessed one's bucket alone validates, and the
    two together refuse as more than one checkpoint, naming both buckets.
    """
    from tcip_mcp.buckets import read_bucket
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed
    from tcip_mcp.tools.inference_tools import run_inference

    nothing = Result((), (), population=())
    chain = run_the_chain(tmp_path, experiment_id="exp-series-a")
    other = train_on(chain.selection_dir, tmp_path, "exp-series-b")
    second = f"other/{DATE}"
    published = run_inference(
        tmp_path,
        checkpoint_path=other,
        images_dir=str(chain.images_dir),
        bucket=second,
        stated=Stated(**SAMPLE_DETECTOR_PASS),
        actor=None,
    )
    assert "error" not in published, published
    revision = latest_confirmed(fx.COUNT_TRAIT, tmp_path)
    first = chain.read()

    assert (
        gate(
            tmp_path,
            [first],
            delivery_kind="per_image_count",
            revision=revision,
            result=nothing,
        ).shipped.kind
        == "validated"
    )
    with pytest.raises(
        DeliveryRefusedError, match="more than one checkpoint"
    ) as refused:
        gate(
            tmp_path,
            [first, read_bucket(chain.root, second)],
            delivery_kind="per_image_count",
            revision=revision,
            result=nothing,
        )
    assert chain.bucket in str(refused.value) and second in str(refused.value)


def test_an_assessment_whose_retained_reference_is_gone_answers_for_nothing(
    tmp_path: Path,
) -> None:
    """The gate refuses a bucket whose assessment's retained reference is gone.

    An assessment whose retained reference copies are removed reads as a changed reference, so
    the gate refuses the bucket it validated.
    """
    from tcip_mcp.assessment import REFERENCE_DIR, assessment_dir
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed

    chain = run_the_chain(tmp_path, experiment_id="exp-forged")
    shutil.rmtree(
        assessment_dir(tmp_path, chain.assessment["assessment_id"])
        / REFERENCE_DIR
    )

    with pytest.raises(DeliveryRefusedError, match="changed since"):
        gate(
            tmp_path,
            [chain.read()],
            delivery_kind="per_image_count",
            revision=latest_confirmed(fx.COUNT_TRAIT, tmp_path),
            result=Result((), (), population=()),
        )


def test_a_reference_source_edited_after_the_assessment_answers_for_nothing(
    tmp_path: Path,
) -> None:
    """The gate refuses a bucket whose reference source was edited after its assessment.

    The pixels an assessment measured are part of its reference: a source image edited after it
    refuses the bucket it validated, naming the image, which cleared before the edit.
    """
    from PIL import Image

    from tcip_mcp.assessment import read_assessment
    from tcip_mcp.delivery import DeliveryRefusedError, Result, gate
    from tcip_mcp.operationalization import latest_confirmed

    chain = run_the_chain(tmp_path, experiment_id="exp-edited-source")
    nothing = Result((), (), population=())
    revision = latest_confirmed(fx.COUNT_TRAIT, tmp_path)
    bucket = chain.read()
    assert (
        gate(
            tmp_path,
            [bucket],
            delivery_kind="per_image_count",
            revision=revision,
            result=nothing,
        ).shipped.kind
        == "validated"
    )
    assessment = read_assessment(tmp_path, chain.assessment["assessment_id"])
    source = assessment.reference.samples[0].sample.source
    with Image.open(source) as frame:
        edited = frame.copy()
    edited.putpixel((0, 0), (255, 0, 0))
    edited.save(source)

    with pytest.raises(DeliveryRefusedError, match="changed since") as refused:
        gate(
            tmp_path,
            [bucket],
            delivery_kind="per_image_count",
            revision=revision,
            result=nothing,
        )
    assert source in str(refused.value)


# -- a missing fact refuses by name -----------------------------------------------------


def test_an_evaluation_record_missing_its_ground_truth_field_refuses_by_name() -> (
    None
):
    from pydantic import ValidationError

    from tcip_mcp.pipelines.training.evaluation import EvaluationRecord

    with pytest.raises(ValidationError, match="gt"):
        EvaluationRecord.model_validate(
            {
                "width": IMG,
                "height": IMG,
                "dt": [],
                "cap": None,
                "frame_count": 0,
                "image_id": None,
                "region": None,
            }
        )


def test_a_scalar_prediction_carrying_no_output_refuses_the_assessment_by_name(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from tcip_mcp.assessment import AssessmentRefusedError, _scalar
    from tcip_mcp.pipelines.data.selection import (
        ClassScope,
        Sample,
        WholeSource,
    )
    from tcip_mcp.pipelines.execution import Pass, execution_record
    from tcip_mcp.pipelines.inference.passes import HeadPass
    from tests._producer_fixtures import write_image

    class Silent:
        def predict_batch(
            self,
            sources: list[str | Path | BandGroupRef],
            execution: Execution,
            *,
            tile_batch_size: int,
        ) -> list[HeadPass]:
            return [
                HeadPass(
                    image=str(s),
                    width=8,
                    height=8,
                    outputs={"regression_values": []},
                )
                for s in sources
            ]

    checkpoint = SimpleNamespace(
        task="regression", payload={}, path="regressor.pt"
    )
    p = Pass(
        checkpoint=checkpoint,
        predictor=Silent(),
        scope=ClassScope(),
        execution=execution_record(checkpoint, Stated(), None, None),
        tile_batch_size=1,
    )
    image = write_image(tmp_path / "a.png", (8, 8))
    (tmp_path / "t.csv").write_text("image,value\na,1.0\n", encoding="utf-8")
    sample = Sample(
        member="a",
        source=str(image),
        ground_truth=str(tmp_path / "t.csv"),
        group="a",
        side="holdout",
        extent=WholeSource(),
        row_key="a",
    )
    entry = fx.COUNT_SPEC.model_copy(
        update={"regression_criterion": "r_squared"}
    )

    with pytest.raises(AssessmentRefusedError, match="_values"):
        _scalar(p, [sample], entry, {sample.location: "digest"})

    class Short:
        def predict_batch(
            self,
            sources: list[str | Path | BandGroupRef],
            execution: Execution,
            *,
            tile_batch_size: int,
        ) -> list[HeadPass]:
            return []

    p.predictor = Short()
    with pytest.raises(
        AssessmentRefusedError, match="0 predictions for 1 holdout samples"
    ):
        _scalar(p, [sample], entry, {sample.location: "digest"})


def test_a_polygon_that_fails_to_rasterize_refuses_rather_than_training_an_empty_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import ImageDraw

    from tests._producer_fixtures import dataset_over, label_image, write_image

    images_dir = image_dir(tmp_path, UNDATED_BUCKET)
    label_image(
        write_image(images_dir / "a.png", (IMG, IMG)),
        [
            Annotation(
                subject=SUBJECT,
                geometry=Polygon(rings=[[(2, 2), (20, 2), (20, 20)]]),
            )
        ],
        IMG,
        IMG,
    )
    dataset = dataset_over("instance_seg", images_dir, subject=SUBJECT)

    def refuse(
        self: ImageDraw.ImageDraw, *args: object, **kwargs: object
    ) -> NoReturn:
        raise ValueError("the ring would not draw")

    monkeypatch.setattr(ImageDraw.ImageDraw, "polygon", refuse)
    with pytest.raises(ValueError, match="would not draw"):
        dataset[0]


def test_an_instance_segmentation_assessment_reads_its_reference_through_polygons(
    tmp_path: Path,
) -> None:
    """An instance segmentation count fit reads polygons and refuses a box-only reference.

    An instance-segmentation checkpoint's count fit reads its reference through its own task's
    loader: a reference holding the subject as polygons is fitted, and one holding it only as
    boxes refuses by name.
    """
    from tcip_mcp import assessment
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.data.selection import source_digests
    from tcip_mcp.pipelines.execution import prepare
    from tests._chain_fixtures import (
        BESPOKE_INSTANCE_SEG,
        BESPOKE_MODELS,
        FRAME_STATISTICS,
    )
    from tests._producer_fixtures import samples_over
    from tests._verified_checkpoint_fixtures import (
        detection_images,
        registered_checkpoint,
    )

    checkpoint = load_registered_checkpoint(
        registered_checkpoint(
            tmp_path,
            model_source={
                "builder": BESPOKE_INSTANCE_SEG,
                "source_files": [BESPOKE_MODELS],
                "task": "instance_seg",
                "normalization": FRAME_STATISTICS,
            },
        ),
        project=tmp_path,
    )

    def fit(polygons: bool) -> tuple[Pass, dict, list[str]]:
        reference = detection_images(
            tmp_path.parent / f"reference-{polygons}",
            {"subject": SUBJECT},
            n=4,
            polygons=polygons,
        )
        samples = samples_over(reference["images_dir"], subject=SUBJECT)
        return assessment._detection(
            prepare(
                checkpoint,
                collection_pass(
                    tile=False,
                    masks=True,
                    mask_cut=MASK_CUT,
                ),
                device="cpu",
            ),
            samples[:2],
            samples[2:],
            fx.COUNT_SPEC,
            None,
            source_digests(samples),
            **COUNT_TEST,
        )

    _pass, measured, _failures = fit(polygons=True)
    assert measured.count is not None
    with pytest.raises(ValueError, match="instance_seg loader does not read"):
        fit(polygons=False)


def test_an_assessment_records_the_test_settings_and_the_reason_it_states(
    tmp_path: Path,
) -> None:
    """An assessment records the test settings and the reason it states, no door defaulting them.

    The equivalence multiplier and the dispersion quantile a count assessment states are the
    ones its record carries beside its reason, and neither door holds a value for any of them.
    """
    import inspect

    from tcip_mcp import assessment
    from tcip_mcp.tools import calibration_tools

    chain = run_the_chain(tmp_path, experiment_id="exp-stated-test")
    recorded = assessment.read_assessment(
        tmp_path, chain.assessment["assessment_id"]
    )

    count = recorded.criterion.count
    assert count is not None
    assert count.equivalence_z == EQUIVALENCE_Z
    assert (
        count.calibration_curve["dispersion_quantile"] == DISPERSION_QUANTILE
    )
    assert recorded.rationale == RATIONALE
    for door in (assessment.assess, calibration_tools.assess_checkpoint):
        parameters = inspect.signature(door).parameters
        for name in ("equivalence_z", "dispersion_quantile", "rationale"):
            assert parameters[name].default is inspect.Parameter.empty, (
                door,
                name,
            )


def test_a_detector_assessment_collects_at_the_conf_it_states_and_records_it_as_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A detector assessment collects at the conf it states and records it as the floor.

    Every prediction a detector assessment collects runs at the conf its statement carries, that
    conf is the floor its evidence records, and a statement carrying no conf refuses naming the
    field.
    """
    from tcip_mcp.pipelines.inference.generic_predictor import GenericPredictor
    from tcip_mcp.tools.calibration_tools import assess_checkpoint

    _images, _drawn, selection_dir, checkpoint = _trained(
        tmp_path, "exp-floor"
    )
    confirm_count_trait(tmp_path)
    stated_conf = 0.05
    collected: set[float] = set()
    real = GenericPredictor.predict_batch

    def recording(
        self: GenericPredictor,
        sources: list[str | Path | BandGroupRef],
        execution: Execution,
        **kwargs: int,
    ) -> list[dict]:
        collected.add(execution.conf)
        return real(self, sources, execution, **kwargs)

    monkeypatch.setattr(GenericPredictor, "predict_batch", recording)

    record = assess(tmp_path, checkpoint, selection_dir, conf=stated_conf)

    assert "error" not in record, record
    assert collected == {stated_conf}
    assert record["criterion"]["count"]["staged_conf_floor"] == stated_conf
    assert record["criterion"]["count"]["observed_min_score"] >= stated_conf
    unstated = assess_checkpoint(
        tmp_path,
        checkpoint_path=checkpoint,
        trait=fx.COUNT_TRAIT,
        delivery_kind="per_image_count",
        selection_dir=str(selection_dir),
        equivalence_z=EQUIVALENCE_Z,
        dispersion_quantile=DISPERSION_QUANTILE,
        rationale=RATIONALE,
        stated=Stated(),
        actor=None,
    )
    assert "states no conf" in unstated["error"], unstated


@pytest.mark.parametrize(
    ("tool", "bands"),
    [
        ("assess_checkpoint", {"selection_dir": "s"}),
        ("assess_reserved_regions", {"k_cal": 3, "k_test": 3}),
    ],
)
def test_the_assess_tools_take_a_dispersion_quantile_within_zero_and_one(
    tool: str, bands: dict[str, str] | dict[str, int]
) -> None:
    """The assess tools refuse a dispersion quantile of -0.1 or 1.5 by name, and not 0.0 or 1.0.

    Called through the MCP server, the answer at -0.1 and 1.5 names ``dispersion_quantile``, and
    the answer at 0.0 and 1.0 does not.
    """
    from tests._server_calls import called_through_the_server

    def called(quantile: float) -> str:
        return called_through_the_server(
            tool,
            {
                "checkpoint_path": "absent.pt",
                "trait": fx.COUNT_TRAIT,
                "delivery_kind": "per_image_count",
                **bands,
                "equivalence_z": EQUIVALENCE_Z,
                "dispersion_quantile": quantile,
                "rationale": RATIONALE,
                "stated": {},
            },
        )

    for outside in (-0.1, 1.5):
        assert "dispersion_quantile" in called(outside), outside
    for end in (0.0, 1.0):
        assert "dispersion_quantile" not in called(end), end


# -- a restored execution record runs exactly as recorded ---------------------------------


def test_a_restored_record_runs_as_recorded_and_a_changed_overlap_or_merge_refuses_by_name(
    tmp_path: Path,
) -> None:
    from tcip_mcp.model_registry import load_registered_checkpoint
    from tcip_mcp.pipelines.execution import ExecutionRefusedError, prepare

    _images, _drawn, _sel, checkpoint_path = _trained(tmp_path, "exp-restore")
    checkpoint = load_registered_checkpoint(checkpoint_path, project=tmp_path)

    def tiled(**merge: bool | int | float | str | MaskCut | None) -> Stated:
        return Stated(
            tile=True, overlap=0.2, **{**SAMPLE_DETECTOR_PASS, **merge}
        )

    recorded = prepare(checkpoint, tiled()).runnable().execution
    assert recorded.tiled, recorded
    assert recorded.postprocess == "nms"
    stated_merge = (
        prepare(checkpoint, tiled(postprocess="nmm")).runnable().execution
    )
    assert stated_merge.postprocess == "nmm"

    assert (
        prepare(
            checkpoint,
            Stated(overlap=0.2, rationale="a restored pass's own reason"),
            restored=recorded,
        )
        .runnable()
        .execution
        == recorded
    )
    with pytest.raises(
        ExecutionRefusedError, match="overlap stated 0.5, recorded 0.2"
    ):
        prepare(
            checkpoint,
            Stated(overlap=0.5, rationale=RATIONALE),
            restored=recorded,
        ).runnable()
    with pytest.raises(
        ExecutionRefusedError, match="postprocess stated 'nmm', recorded 'nms'"
    ):
        prepare(
            checkpoint,
            Stated(postprocess="nmm", rationale=RATIONALE),
            restored=recorded,
        ).runnable()
    # A blank merge is a name the vocabulary refuses.
    with pytest.raises(ValueError, match="Unknown cross-tile merge ''.*nms"):
        prepare(checkpoint, tiled(postprocess="")).runnable()
