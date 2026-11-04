---
name: Design proposal
about: A change touching a persisted format, a refusal, an operating-point stamp, or a delivery gate
title: "[design] "
labels: design
---

A change touching a persisted format, a refusal, an operating-point stamp or a delivery gate is
designed here before any code is written. This template is that design.

## What changes

Name the persisted format, refusal, operating-point stamp, or delivery gate this touches, and what
changes about it.

## Why

What is broken, missing, or wrong today that this fixes or adds.

## Persisted-format impact

If this re-shapes a record: its one producer and its one reader change together, and the records
already written are regenerated or discarded with the change, never read through a default or a
migration shim.

## Refusal or gate impact

If this adds or changes a refusal: what it refuses, and the legitimate call that must still succeed
afterward (a rail change ships with a test proving valid work still passes, constructed through the
platform's own producer).

## Alternatives considered

What else you considered and why this is the one worth building.

## Reads by another model family

Before code, run the design past a model family other than its author's with
`tools/cross_family_ask.py`, and past two families when it changes a persisted format. Record each
read's verdict here, and how the design answers each finding.
