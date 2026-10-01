"""Building a loader over data on disk the way every door in the platform does.

A loader is built from the samples the producer named, never from a place to look, so a test that
needs one over a directory or a table admits through ``label_queries.admit`` first and builds from
what it answers. These helpers are that one route, so a fixture and the production path cannot
drift into admitting different membership.
"""

from __future__ import annotations

from typing import Any


def admit_over(
    images_dir, ground_truth, *, subject: str | None = None, attribute: str | None = None,
    members: list[str] | None = None,
):
    """The admission over one place holding ground truth under a fresh statement of subject and
    attribute (:func:`~tcip_mcp.pipelines.data.label_queries.stated_scope`), refusing an empty one
    by name."""
    from tcip_mcp.pipelines.data.label_queries import admit, require_admitted, stated_scope

    admitted = admit(images_dir, ground_truth,
                     scope=stated_scope(ground_truth, subject, attribute), members=members)
    require_admitted(admitted)
    return admitted


def samples_over(
    images_dir, ground_truth, *, subject: str | None = None, attribute: str | None = None,
    members: list[str] | None = None,
):
    """Every admitted member of one place as samples, each on the training side."""
    return admit_over(images_dir, ground_truth, subject=subject, attribute=attribute,
                      members=members).every_sample()


def run_over(
    task: str, images_dir, ground_truth, *, subject: str | None = None,
    attribute: str | None = None, members: list[str] | None = None,
    stated: dict[str, Any] | None = None, **kwargs: Any,
):
    """A loader for ``task`` over one place holding ground truth and the data section a run over
    it records (its ``scope`` and sizes), both through the producer.

    ``stated`` is what a config would state about the sizes (a band count, a class count); the
    rest are resolved off the admitted samples the way a run resolves them."""
    from dataclasses import asdict

    from tcip_mcp.pipelines.data.datasets import build_dataset
    from tcip_mcp.pipelines.data.split_construction import run_sizes

    admitted = admit_over(images_dir, ground_truth, subject=subject, attribute=attribute,
                          members=members)
    samples = admitted.every_sample()
    data = {**(stated or {}), "scope": asdict(admitted.scope)}
    sizes = run_sizes(task, data, samples, kwargs.get("dataset_source"))
    return build_dataset(task, samples=samples, scope=admitted.scope, sizes=sizes, **kwargs), data


def dataset_over(task: str, images_dir, ground_truth, **kwargs: Any):
    """The loader :func:`run_over` builds."""
    return run_over(task, images_dir, ground_truth, **kwargs)[0]
