"""Image tiling for high-resolution images.

Splits large images into overlapping tiles for detection, then
reassembles predictions using NMS across tile boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image


@dataclass
class Tile:
    """A single tile with its position in the source image."""
    image: Image.Image
    x_offset: int
    y_offset: int
    width: int
    height: int


def generate_tiles(
    image: Image.Image,
    tile_size: int = 1024,
    overlap: int = 128,
) -> list[Tile]:
    """Split an image into overlapping tiles.

    Args:
        image: Source PIL image.
        tile_size: Width and height of each tile.
        overlap: Overlap in pixels between adjacent tiles.
    """
    w, h = image.size
    stride = tile_size - overlap
    tiles: list[Tile] = []

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            crop = image.crop((x, y, x2, y2))
            tiles.append(Tile(image=crop, x_offset=x, y_offset=y, width=x2 - x, height=y2 - y))

    return tiles


def merge_tile_predictions(
    tiles: list[Tile],
    tile_predictions: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Merge predictions from multiple tiles using NMS.

    Args:
        tiles: The tile objects with offset info.
        tile_predictions: Per-tile prediction dicts with 'boxes', 'scores', 'labels'.
        iou_threshold: IoU threshold for NMS deduplication.
    """
    all_boxes: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for tile, preds in zip(tiles, tile_predictions):
        boxes = preds.get("boxes", torch.zeros(0, 4))
        if len(boxes) == 0:
            continue
        # Offset boxes to full-image coordinates
        offset = torch.tensor([tile.x_offset, tile.y_offset, tile.x_offset, tile.y_offset], dtype=boxes.dtype)
        all_boxes.append(boxes + offset)
        all_scores.append(preds.get("scores", torch.zeros(len(boxes))))
        all_labels.append(preds.get("labels", torch.zeros(len(boxes), dtype=torch.int64)))

    if not all_boxes:
        return {"boxes": torch.zeros(0, 4), "scores": torch.zeros(0), "labels": torch.zeros(0, dtype=torch.int64)}

    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # Per-class NMS
    from torchvision.ops import nms

    keep_indices: list[int] = []
    for cls in labels.unique():
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]
        cls_indices = torch.where(cls_mask)[0]
        keep = nms(cls_boxes, cls_scores, iou_threshold)
        keep_indices.extend(cls_indices[keep].tolist())

    keep_indices.sort()
    keep = torch.tensor(keep_indices, dtype=torch.long)

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }
