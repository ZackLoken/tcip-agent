"""Route modules for the tcip-web FastAPI backend.

Each submodule registers its own :class:`APIRouter`; :func:`register_all`
mounts them on the app.
"""

from __future__ import annotations

from fastapi import FastAPI

from tcip_web.routes import (
    annotate,
    classes,
    dataset,
    images,
    inference,
    meta,
    results,
    review,
    sessions,
    training,
    tuning,
)


def register_all(app: FastAPI) -> None:
    app.include_router(dataset.router)
    app.include_router(images.router)
    app.include_router(annotate.router)
    app.include_router(review.router)
    app.include_router(training.router)
    app.include_router(inference.router)
    app.include_router(results.router)
    app.include_router(tuning.router)
    app.include_router(classes.router)
    app.include_router(sessions.router)
    app.include_router(meta.router)
