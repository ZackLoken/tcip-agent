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
    """The admission over one place holding ground truth, refusing an empty one by name."""
    from tcip_mcp.pipelines.data.label_queries import admit, require_admitted

    admitted = admit(images_dir, ground_truth, subject=subject, attribute=attribute,
                     members=members)
    require_admitted(admitted)
    return admitted


def samples_over(
    images_dir, ground_truth, *, subject: str | None = None, attribute: str | None = None,
    members: list[str] | None = None,
):
    """Every admitted member of one place as samples, each on the training side."""
    return admit_over(images_dir, ground_truth, subject=subject, attribute=attribute,
                      members=members).every_sample()


def admission_of(members: list[str]):
    """The producer's own record over named members whose files are never written to disk, for a
    test driving a door's own pass rather than the admission under it. Every projection a door
    takes off it (its samples, its counts, its scope) is the producer's own."""
    from tcip_mcp.pipelines.data.label_queries import Admission, Admitted
    from tcip_mcp.pipelines.data.selection import DOCUMENT

    return Admission(
        shape=DOCUMENT, images_dir="images", ground_truth="labels",
        records=[Admitted(member=name, source=f"{name}.jpg", ground_truth=f"{name}.json")
                 for name in members],
        counts={"annotated": len(members)},
    )


def dataset_over(
    task: str, images_dir, ground_truth, *, subject: str | None = None,
    attribute: str | None = None, members: list[str] | None = None, **kwargs: Any,
):
    """A loader for ``task`` over one place holding ground truth, through the producer."""
    from tcip_mcp.pipelines.data.datasets import build_dataset

    admitted = admit_over(images_dir, ground_truth, subject=subject, attribute=attribute,
                          members=members)
    return build_dataset(task, samples=admitted.every_sample(), scope=admitted.scope, **kwargs)
