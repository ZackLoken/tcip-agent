"""The answers a route gives a checkpoint's admission that fails, and a registry it cannot read."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from collections.abc import Iterator


def registry_unreadable(exc: Exception) -> HTTPException:
    """Return the 409 answering a registry that could not be read, carrying ``exc``'s sentence."""
    return HTTPException(409, f"registry unreadable: {exc}")


@contextmanager
def answered_admission() -> Iterator[None]:
    """Answer each failure of a checkpoint's admission inside the block with its sentence.

    No regular file at the checkpoint's location
    (:class:`~tcip_mcp.model_registry.CheckpointNotFoundError`) answers 404, a refusal
    (``ValueError``) 400, and any other store error or operating-system error 409
    (:func:`registry_unreadable`). A busy store and a version conflict propagate to the app's
    own handlers.
    """
    from tcip_mcp.model_registry import CheckpointNotFoundError
    from tcip_store import StoreBusyError, StoreError, VersionConflictError

    try:
        yield
    except CheckpointNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except (StoreBusyError, VersionConflictError):
        raise
    except (StoreError, OSError) as exc:
        raise registry_unreadable(exc) from exc
