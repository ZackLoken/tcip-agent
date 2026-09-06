---
name: Design proposal
about: A change touching a persisted format, a refusal, or a delivery gate
title: "[design] "
labels: design
---

CONTRIBUTING.md requires design review before code for any change touching a persisted field, a
refusal, an operating-point stamp, or a delivery gate. This template is that review.

## What changes

Name the persisted format, refusal, or delivery gate this touches, and what changes about it.

## Why

What is broken, missing, or wrong today that this fixes or adds.

## Persisted-format impact

If this touches a store's on-disk shape: is it inside the store's frozen version 1, or does it
need a version bump (`frozen-formats.json`, `tools/generate_frozen_manifest.py`)? A bump states
its own obligations: how an existing reader still reads an old-version document, and how existing
on-disk state is conformed (a `cli/` command shipped with the bump and deleted once every root
that needed it is conformed, never a runtime migration shim).

## Refusal or gate impact

If this adds or changes a refusal: what it refuses, and the legitimate call that must still
succeed afterward (a rail change ships with a test proving valid work still passes, constructed
through the platform's own producer).

## Alternatives considered

What else you considered and why this is the one worth building.
