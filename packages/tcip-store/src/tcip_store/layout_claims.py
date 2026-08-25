"""Which store could own which path under a root, as data rather than as locator inversion.

A locator answers "where does this key go" and inverts that placement, but it cannot answer
"which store owns this file": locator shapes collide, and thirteen stores place a single json
document under ``.tcip/state``. What tells them apart is the constant text each store's key
constructor spells and the directory chain its locator puts it under. That is stated here, one
claim per record and log store, as path templates a reader matches without importing the owning
module and without a live registry.

Two callers share the one matching implementation, so the rail and the planner cannot disagree
about which files a root still holds. The conform rail asks whether any claim of the root's
layout matches a file under it; the adoption planner asks the same question and then attributes
each match to the store whose template says the most about it. Parts recovery is not a claim's
job and stays with the locators.

The walk is template-directed: it descends only the constant segments a layout's templates
spell and lists a directory only where a template allows a varying segment, so checking a
dataset root reads ``.tcip`` rather than the image tree.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tcip_store.errors import StoreError
from tcip_store.file_backend import _is_bookkeeping, require_absolute_root
from tcip_store.registry import claim_generation, get_descriptor, registered_stores

ROOT = "root"
"""A project or dataset root: the directory holding ``images/``, ``annotations/`` and ``.tcip/``."""

STATE = "state"
"""A root's ``.tcip/state`` directory, the root the review shards and per-trait records hang off."""

EXPERIMENTS = "experiments"
"""A root's ``.tcip/experiments`` directory, the one root every experiment's members share."""

WORKSPACE = "workspace"
"""The workspace directory holding the project folders and the active-project marker."""

HPO_ROOT = "hpo_root"
"""A root's ``.tcip/hpo`` directory, holding one study result and one manifest per sweep."""

SWEEP = "sweep"
"""One sweep's directory under the hpo root, holding a directory per trial."""

SPLITS = "splits"
"""A partition's output directory: one document per split plus the manifest describing them."""

CURATED = "curated"
"""A curated dataset's output directory."""

RUN = "run"
"""A training or evaluation run's output directory."""

PREDICTION_BUCKET = "prediction_bucket"
"""One prediction bucket directory, where a run's operating-point stamps sit beside its output."""

LAYOUTS = (
    ROOT,
    STATE,
    EXPERIMENTS,
    WORKSPACE,
    HPO_ROOT,
    SWEEP,
    SPLITS,
    CURATED,
    RUN,
    PREDICTION_BUCKET,
)
"""Every kind of directory a root can be, for an operator naming one on a command line."""


@dataclass(frozen=True)
class PartPattern:
    """What one part of a store's key looks like across every entry the store holds.

    A part is either a constant the store's key constructor spells (``literal``), a varying
    value with a fixed opening the constructor puts there (``starts_with``), or free. The
    three are ordered by how much they say, which is how a file two stores' templates both
    match is attributed to the store that says more about it.
    """

    literal: str | None = None
    starts_with: str = ""

    def matches(self, part: str) -> bool:
        """Whether this part could belong to the store this pattern describes."""
        if self.literal is not None:
            return part == self.literal
        return part.startswith(self.starts_with)

    @property
    def specificity(self) -> int:
        """How much the pattern constrains, for choosing between two stores claiming one file."""
        if self.literal is not None:
            return 2
        return 1 if self.starts_with else 0


ANY = PartPattern()
"""A part whose value varies with no fixed opening."""


def literal(text: str) -> PartPattern:
    """A part that is the same constant in every entry of the store."""
    return PartPattern(literal=text)


@dataclass(frozen=True)
class Constant:
    """A path segment that reads the same in every entry a store holds."""

    text: str


@dataclass(frozen=True)
class Patterned:
    """A path segment carrying a key part: constant lead text, the part, constant tail text.

    The lead and the tail are what the locator puts around the part inside one segment: the
    stem a hash-named artifact opens with, and the extension the file carries.
    """

    pattern: PartPattern = ANY
    lead: str = ""
    tail: str = ""


Segment = Constant | Patterned
Template = tuple[Segment, ...]

_BARE_EXTENSION = re.compile(r"^\.[^.\\/]+$")


@dataclass(frozen=True)
class Claim:
    """Where one store's entries can sit under a root of ``layout``.

    A store carries more than one template when one shape cannot cover every legal entry: a
    review verdict places its shard under a bucket directory or directly under ``review/``
    depending on whether the review named a prediction bucket, and both depths are legal.
    """

    layout: str
    templates: tuple[Template, ...]


@dataclass(frozen=True)
class Claimant:
    """One store whose claim matches a file, and how much that claim says about it."""

    store: str
    specificity: int


@dataclass(frozen=True)
class ClaimedFile:
    """A file under a root that at least one claim of the root's layout matches."""

    path: Path
    claimants: tuple[Claimant, ...]

    @property
    def stores(self) -> tuple[str, ...]:
        """Every store that could own this file, sorted."""
        return tuple(claimant.store for claimant in self.claimants)


@dataclass(frozen=True)
class AnchoredMatch:
    """A path an anchored template of one store would place under ``root``."""

    root: Path
    store: str
    layout: str


def _matches_segment(segment: Segment, text: str) -> bool:
    """Whether one path segment could be the segment this matcher describes."""
    if isinstance(segment, Constant):
        return text == segment.text
    if not text.startswith(segment.lead) or not text.endswith(segment.tail):
        return False
    part = text[len(segment.lead) : len(text) - len(segment.tail)]
    return bool(part) and segment.pattern.matches(part)


def matches_template(template: Template, segments: Sequence[str]) -> bool:
    """Whether these path segments are one entry of the store this template describes."""
    if len(template) != len(segments):
        return False
    return all(_matches_segment(matcher, text) for matcher, text in zip(template, segments, strict=True))


def _segment_specificity(segment: Segment) -> int:
    """How much one segment constrains, ranked the way a key part's pattern is.

    The tail is not counted: every store of a kind carries the same extension, so it
    separates nothing.
    """
    if isinstance(segment, Constant):
        return 2
    return segment.pattern.specificity + (1 if segment.lead else 0)


def template_specificity(template: Template) -> int:
    """How much a whole template constrains, for choosing between two stores claiming one file."""
    return sum(_segment_specificity(segment) for segment in template)


def _anchors(segment: Segment) -> bool:
    """Whether this segment carries constant text an ordinary file would not share.

    A wholly constant segment anchors, and so does constant text around or inside the part.
    An extension alone does not: every ordinary file of that type carries it, so a template
    anchored on one would match every such file anywhere.
    """
    if isinstance(segment, Constant):
        return True
    if segment.lead or segment.pattern.literal is not None or segment.pattern.starts_with:
        return True
    return bool(segment.tail) and not _BARE_EXTENSION.match(segment.tail)


def anchored(template: Template) -> bool:
    """Whether a template says enough about a path to be matched against an arbitrary target."""
    return any(_anchors(segment) for segment in template)


def _rooted(*segments: str, suffix: str = "") -> Template:
    """A template of constant directories ending in one segment that carries the key part."""
    return (*(Constant(text) for text in segments), Patterned(ANY, tail=suffix))


def _named(*segments: str, name: str, suffix: str = "") -> Template:
    """A template of constant directories ending in a document the store always calls ``name``."""
    return (*(Constant(text) for text in segments), Patterned(literal(name), tail=suffix))


PLATFORM_CLAIMS: Mapping[str, Claim] = {
    "image_status": Claim(ROOT, (_named(".tcip", "state", name="image_status", suffix=".json"),)),
    "image_status_digest": Claim(
        ROOT, (_named(".tcip", "state", name="image_status_digest", suffix=".json"),)
    ),
    "view_coverage": Claim(ROOT, (_named(".tcip", "state", name="view_coverage", suffix=".json"),)),
    "region_completeness": Claim(
        ROOT, (_named(".tcip", "state", name="region_completeness", suffix=".json"),)
    ),
    "region_completeness_digest": Claim(
        ROOT, (_named(".tcip", "state", name="region_completeness_digest", suffix=".json"),)
    ),
    "gui_snapshot": Claim(ROOT, (_named(".tcip", "state", name="gui", suffix=".json"),)),
    "canvas_meta": Claim(ROOT, (_named(".tcip", "state", name="canvas_live", suffix=".json"),)),
    "canvas_geometry": Claim(ROOT, (_named(".tcip", "state", name="canvas_shapes", suffix=".json"),)),
    "project_status": Claim(ROOT, (_named(".tcip", "state", name="project_status", suffix=".json"),)),
    "annotation_stats": Claim(
        ROOT, (_named(".tcip", "state", name="annotation_stats", suffix=".json"),)
    ),
    "ray_dashboard": Claim(ROOT, (_named(".tcip", "state", name="ray_dashboard", suffix=".json"),)),
    "backend_port": Claim(ROOT, (_named(".tcip", "state", name="web_port", suffix=".txt"),)),
    "job_registry": Claim(ROOT, (_named(".tcip", "state", name="inference_jobs", suffix=".json"),)),
    "proposal_staging": Claim(
        ROOT,
        (
            (Constant(".tcip"), Constant("state"), Constant("proposals"),
             Patterned(ANY), Patterned(ANY, tail=".json")),
        ),
    ),
    "model_registry": Claim(ROOT, (_named(".tcip", "models", name="registry", suffix=".json"),)),
    "dataset_registry": Claim(ROOT, (_named(".tcip", name="datasets", suffix=".json"),)),
    "project_record": Claim(ROOT, (_named(".tcip", name="project", suffix=".json"),)),
    "audit_log": Claim(ROOT, (_named(".tcip", name="audit", suffix=".jsonl"),)),
    "learning_capture": Claim(ROOT, (_named(".tcip", name="learning_capture", suffix=".jsonl"),)),
    "friction_reports": Claim(ROOT, (_rooted(".tcip", "reports", suffix=".json"),)),
    "retrospectives": Claim(ROOT, (_rooted(".tcip", "retrospectives", suffix=".md"),)),
    "confidence_sweep": Claim(
        ROOT,
        (
            (
                Constant(".tcip"),
                Constant("artifacts"),
                Patterned(ANY, lead="operating_point_sweep_", tail=".json"),
            ),
        ),
    ),
    "cal_holdout_split_lock": Claim(
        ROOT,
        (
            (
                Constant(".tcip"),
                Constant("artifacts"),
                Patterned(ANY, lead="cal_holdout_split_", tail=".json"),
            ),
        ),
    ),
    "plant_mapping": Claim(STATE, (_named(name="plant_mapping", suffix=".json"),)),
    "review_verdicts": Claim(
        STATE,
        (
            (Constant("review"), Patterned(ANY), Patterned(ANY, tail=".json")),
            (Constant("review"), Patterned(ANY, tail=".json")),
        ),
    ),
    "trait_specs": Claim(
        STATE, ((Constant("trait_specs"), Patterned(ANY, tail=".json")),)
    ),
    "trait_operationalizations": Claim(
        STATE,
        ((Constant("trait_operationalizations"), Patterned(ANY), Patterned(ANY, tail=".json")),),
    ),
    "trait_spec_statements": Claim(
        STATE, ((Constant("trait_spec_statements"), Patterned(ANY, tail=".json")),)
    ),
    "delivery_events": Claim(
        STATE, ((Constant("delivery_events"), Patterned(ANY, tail=".json")),)
    ),
    "experiment_config": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("config"), tail=".json")),)
    ),
    "experiment_status": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("status"), tail=".json")),)
    ),
    "experiment_lineage": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("lineage"), tail=".json")),)
    ),
    "experiment_artifacts": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("artifacts"), tail=".json")),)
    ),
    "experiment_env": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("env"), tail=".json")),)
    ),
    "experiment_split": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("split"), tail=".json")),)
    ),
    "experiment_metrics": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("metrics"), tail=".jsonl")),)
    ),
    "experiment_validations": Claim(
        EXPERIMENTS, ((Patterned(ANY), Patterned(literal("validations"), tail=".jsonl")),)
    ),
    "model_snapshot_manifest": Claim(
        EXPERIMENTS,
        ((Patterned(ANY), Constant("model_src"), Patterned(literal("manifest"), tail=".json")),),
    ),
    "workspace_active_project": Claim(WORKSPACE, ((Patterned(literal(".active")),),)),
    "hpo_study_result": Claim(HPO_ROOT, (_rooted(suffix=".json"),)),
    "hpo_sweep_manifest": Claim(
        HPO_ROOT, ((Patterned(ANY), Patterned(literal("manifest"), tail=".json")),)
    ),
    "hpo_trial_config": Claim(
        SWEEP, ((Patterned(ANY), Patterned(literal("resolved_config"), tail=".json")),)
    ),
    "hpo_trial_metrics": Claim(
        SWEEP, ((Patterned(ANY), Patterned(literal("metrics"), tail=".jsonl")),)
    ),
    "split_manifest": Claim(SPLITS, (_named(name="split_manifest", suffix=".json"),)),
    "split_stem_list": Claim(SPLITS, (_rooted(suffix=".json"),)),
    "curated_manifest": Claim(CURATED, (_named(name="curated_manifest", suffix=".json"),)),
    "evaluation_results": Claim(RUN, (_named(name="test_results", suffix=".json"),)),
    "run_launch_config": Claim(RUN, (_named(name="launch_config", suffix=".json"),)),
    "operating_point_sidecar": Claim(
        PREDICTION_BUCKET, (_named(name="operating_point", suffix=".json"),)
    ),
    "classifier_operating_point_sidecar": Claim(
        PREDICTION_BUCKET, (_named(name="classifier_operating_point", suffix=".json"),)
    ),
    "ordinal_operating_point_sidecar": Claim(
        PREDICTION_BUCKET, (_named(name="ordinal_operating_point", suffix=".json"),)
    ),
    "regression_operating_point_sidecar": Claim(
        PREDICTION_BUCKET, (_named(name="regression_operating_point", suffix=".json"),)
    ),
    "resolve_scale_sidecar": Claim(
        PREDICTION_BUCKET, (_named(name="resolve_scale", suffix=".json"),)
    ),
}
"""One row per record and log store the platform declares, keyed by store name.

Each row is derived from that store's own locator plus the constants its key constructor
spells, and the bootstrap inventory test holds the two sides together: a golden key of every
platform store lands on a path its row matches, and near misses do not.
"""


def platform_claim_stores() -> frozenset[str]:
    """Every store the platform table already speaks for."""
    return frozenset(PLATFORM_CLAIMS)


_effective: tuple[int, Mapping[str, Claim]] | None = None


def effective_claims() -> Mapping[str, Claim]:
    """The platform table plus the declared claim of every registered store outside it.

    A store registered at runtime states its own claim in its descriptor, and that claim is
    in force for exactly as long as the store is registered, which is the lifetime in which
    its files can exist. The answer is recomputed whenever a declared claim has joined the
    catalogue, so nothing serves a claim set that has since grown.
    """
    global _effective
    generation = claim_generation()
    cached = _effective
    if cached is not None and cached[0] == generation:
        return cached[1]
    claims = dict(PLATFORM_CLAIMS)
    for name in registered_stores():
        declared = get_descriptor(name).claim
        if declared is not None:
            claims[name] = declared
    _effective = (generation, claims)
    return claims


def claim_of(store: str) -> Claim:
    """Where this store's entries can sit, or the refusal a store that never says earns.

    A record or log store with no claim cannot be placed under any layout, so which files under
    a root would be its own has no answer, and the rail refuses naming the declaration it owes
    rather than serving against a guess.
    """
    claim = effective_claims().get(store)
    if claim is None:
        raise StoreError(
            f"store {store!r} states no layout claim, so which files under a root would be "
            "its own cannot be answered and this backend would have to guess whether the "
            "root's state is already in a database. Declare claim=Claim(layout, templates) "
            "in the store's descriptor."
        )
    return claim


def layouts_of(stores: Sequence[str]) -> tuple[str, ...]:
    """The layouts these stores hang off, sorted and without repeats.

    More than one is ordinary: a directory serves whatever stores a caller roots there, and
    the shipped platform has directories that are a curated output and a dataset root at once.
    Each layout is what the rail then checks the root against, one at a time.
    """
    return tuple(sorted({claim_of(store).layout for store in stores}))


def claimed_files(
    root: str,
    layout: str,
    *,
    claims: Mapping[str, Claim] | None = None,
    limit: int = 0,
) -> tuple[ClaimedFile, ...]:
    """Every file under ``root`` that a claim of ``layout`` matches, with what could own it.

    The walk follows the templates rather than the tree: it stats the constant segments they
    spell and lists a directory only where some template allows a varying segment, so a
    dataset root costs its ``.tcip`` directory rather than its imagery. ``limit`` stops once
    that many are found, for a caller that only needs to know whether any exist.
    """
    directory = require_absolute_root(root)
    if not directory.is_dir():
        return ()
    in_force = effective_claims() if claims is None else claims
    live = [
        (store, claim, template)
        for store, claim in in_force.items()
        if claim.layout == layout
        for template in claim.templates
    ]
    found: list[ClaimedFile] = []
    if live:
        _descend(directory, tuple(live), found, limit)
    return tuple(found)


def layouts_in_play(held: Sequence[str], serving: Sequence[str]) -> frozenset[str]:
    """The kinds of root a directory demonstrably serves, for reasoning about what sits in it.

    A directory serves whatever stores a caller roots there, and which kinds those are is
    answered by the stores a database already holds plus the ones the operation is serving now.
    A held store nothing in this process declares contributes nothing, since where its files
    would sit is exactly what an unstated claim cannot say.
    """
    claims = effective_claims()
    found = {claims[store].layout for store in held if store in claims}
    return frozenset(found | set(serving))


def contested_claimants(
    root: str, path: Path, in_play: frozenset[str]
) -> tuple[str, ...]:
    """The stores that could equally own this file, among the layouts this root serves.

    Claim templates collide across layouts: a free directory's ``metrics.jsonl`` is an
    experiment's metrics log under one layout and a trial's under another, and at a directory
    serving both, nothing about the file says which. Only the claims that say the most about it
    are contenders, the way the planner picks a winner and refuses only a tie, and only the
    layouts this root actually serves are considered, so a shape some other kind of root would
    have claimed is not held against a directory that is not one.
    """
    claims = effective_claims()
    contenders = [
        claimant
        for claimant in claimants_of(root, path, claims=claims)
        if claims[claimant.store].layout in in_play
    ]
    if not contenders:
        return ()
    best = max(claimant.specificity for claimant in contenders)
    return tuple(sorted(c.store for c in contenders if c.specificity == best))


def claimants_of(
    root: str, path: Path, *, claims: Mapping[str, Claim] | None = None
) -> tuple[Claimant, ...]:
    """Every store, under any layout, whose claim matches this file under this root.

    One directory serves whatever stores a caller roots there, and two layouts' templates can
    describe the same path (a free directory's ``metrics.jsonl`` is an experiment's metrics log
    under one layout and a trial's under another). Whose file it is cannot be told apart there,
    which is what the accounting has to notice rather than pick between.
    """
    directory = require_absolute_root(root)
    try:
        segments = path.relative_to(directory).parts
    except ValueError:
        return ()
    in_force = effective_claims() if claims is None else claims
    best: dict[str, int] = {}
    for store, claim in in_force.items():
        for template in claim.templates:
            if not matches_template(template, segments):
                continue
            score = template_specificity(template)
            if score > best.get(store, -1):
                best[store] = score
    return tuple(Claimant(store, best[store]) for store in sorted(best))


def unconformed_files(root: str, layout: str, *, limit: int = 0) -> tuple[Path, ...]:
    """Files under ``root`` whose state is still the file layout rather than a database.

    Each one is evidence this root holds record or log state a database beside it would not
    see. Blob files are not evidence: no blob store carries a claim, because a blob's bytes
    stay a file under every backend.
    """
    return tuple(item.path for item in claimed_files(root, layout, limit=limit))


_Live = tuple[tuple[str, Claim, Template], ...]


def _descend(directory: Path, live: _Live, found: list[ClaimedFile], limit: int) -> None:
    """Match one directory level against the templates still alive, then recurse where they lead."""
    heads = [template[0] for _, _, template in live]
    entries: list[tuple[str, bool, bool]] = []
    if all(isinstance(head, Constant) for head in heads):
        for name in {head.text for head in heads if isinstance(head, Constant)}:
            path = directory / name
            entries.append((name, path.is_dir(), path.is_file()))
    else:
        try:
            entries = [(entry.name, entry.is_dir(), entry.is_file()) for entry in os.scandir(directory)]
        except OSError:
            return
    for name, is_dir, is_file in sorted(entries):
        if _is_bookkeeping(name):
            continue
        matched = [item for item in live if _matches_segment(item[2][0], name)]
        if not matched:
            continue
        if is_file:
            claimants = _claimants([item for item in matched if len(item[2]) == 1])
            if claimants:
                found.append(ClaimedFile(path=directory / name, claimants=claimants))
                if limit and len(found) >= limit:
                    return
        if is_dir:
            deeper = tuple(
                (store, claim, template[1:]) for store, claim, template in matched if len(template) > 1
            )
            if deeper:
                _descend(directory / name, deeper, found, limit)
                if limit and len(found) >= limit:
                    return


def _claimants(matched: Sequence[tuple[str, Claim, Template]]) -> tuple[Claimant, ...]:
    """One entry per store whose template matched, carrying that template's specificity."""
    best: dict[str, int] = {}
    for store, _claim, template in matched:
        score = template_specificity(template)
        if score > best.get(store, -1):
            best[store] = score
    return tuple(Claimant(store, best[store]) for store in sorted(best))


def anchored_matches(
    target: Path, *, claims: Mapping[str, Claim] | None = None
) -> tuple[AnchoredMatch, ...]:
    """Every anchored claim an absolute path's own tail satisfies, with the root it implies.

    Asked of a path rather than of a key, because a caller-named output target roots wherever
    the caller pointed it and a claim's root is whatever sits above the segments it spells.
    Unanchored templates take no part: a template whose only constant text is an extension
    matches every ordinary file of that type, so honouring it here would tax every legitimate
    write.
    """
    in_force = effective_claims() if claims is None else claims
    segments = target.parts
    matches: list[AnchoredMatch] = []
    for store, claim in in_force.items():
        for template in claim.templates:
            if not anchored(template) or len(template) >= len(segments):
                continue
            above = len(segments) - len(template)
            if matches_template(template, segments[above:]):
                matches.append(
                    AnchoredMatch(root=Path(*segments[:above]), store=store, layout=claim.layout)
                )
    return tuple(matches)


__all__ = [
    "ANY",
    "AnchoredMatch",
    "CURATED",
    "Claim",
    "ClaimedFile",
    "Claimant",
    "Constant",
    "EXPERIMENTS",
    "HPO_ROOT",
    "LAYOUTS",
    "PLATFORM_CLAIMS",
    "PREDICTION_BUCKET",
    "PartPattern",
    "Patterned",
    "ROOT",
    "RUN",
    "SPLITS",
    "STATE",
    "SWEEP",
    "Segment",
    "Template",
    "WORKSPACE",
    "anchored",
    "anchored_matches",
    "claim_of",
    "claimants_of",
    "claimed_files",
    "contested_claimants",
    "effective_claims",
    "layouts_in_play",
    "layouts_of",
    "literal",
    "matches_template",
    "platform_claim_stores",
    "template_specificity",
    "unconformed_files",
]
