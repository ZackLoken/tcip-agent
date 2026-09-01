"""Pixel bounds for what the platform serves to a screen or writes as an agent-facing artifact.

One module so the serving route, the visualization tools, and any validator agree on the same
numbers instead of each pinning its own copy of them.

``DISPLAY_MAX_EDGE`` bounds the longest output edge of a display-bound read. Source: the 4096
width cap the frontend previously hardcoded; a documented default pending a real derivation.

``DISPLAY_MAX_PIXELS`` caps the output-pixel area of an explicit region request. Source: the edge
bound squared.

``VIZ_ARTIFACT_MAX_EDGE`` bounds agent-facing visualization artifacts, which are read back
through the model's own image reading. Source: agent image-reading practicality; deliberately
smaller than the display bound, which serves a human screen.
"""

DISPLAY_MAX_EDGE = 4096

DISPLAY_MAX_PIXELS = DISPLAY_MAX_EDGE ** 2

VIZ_ARTIFACT_MAX_EDGE = 1024
