import { memo } from "react";

import { subjectColor } from "@/api/classes";
import { BoxOverlay } from "@/components/annotate/BoxOverlay";
import { PointOverlay } from "@/components/annotate/PointOverlay";
import { PolygonOverlay } from "@/components/annotate/PolygonOverlay";
import { pointShapeVisible } from "@/lib/canvasSync";
import { derivedBoxFromPolygon } from "@/lib/polygonGeometry";
import type { Box, Mode, PointShape, PolygonShape } from "@/store/types";

/**
 * The committed boxes + polygons (content layer). Memoized, and crucially, the mouse
 * cursor is not one of its props, so a mouse move (which only updates cursor-following
 * overlays) does not re-render/reconcile these hundreds–thousands of Konva nodes. It
 * re-renders only when the shapes, selection/hover, active subject, or zoom-derived stroke
 * sizes actually change.
 */
interface AnnotationShapesProps {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  mode: Mode;
  activeSubject: string | null;
  selectedPolygonIdx: number | null;
  selectedBoxIdx: number | null;
  selectedPointIdx: number | null;
  hoveredIdx: number | null;
  hoveredBoxIdx: number | null;
  hoveredDerivedIdx: number | null;
  hoveredPointIdx: number | null;
  draggingIdx: number | undefined;
  renderLabels: boolean;
  boxStroke: number;
  polyStroke: number;
  vertR: number;
  selVertR: number;
  labelSize: number;
  pointCoreR: number;
  pointSelCoreR: number;
  pointTickInner: number;
  pointTickOuter: number;
  scaleLineW: number;
}

export const AnnotationShapes = memo(function AnnotationShapes({
  boxes,
  polygons,
  points,
  mode,
  activeSubject,
  selectedPolygonIdx,
  selectedBoxIdx,
  selectedPointIdx,
  hoveredIdx,
  hoveredBoxIdx,
  hoveredDerivedIdx,
  hoveredPointIdx,
  draggingIdx,
  renderLabels,
  boxStroke,
  polyStroke,
  vertR,
  selVertR,
  labelSize,
  pointCoreR,
  pointSelCoreR,
  pointTickInner,
  pointTickOuter,
  scaleLineW,
}: AnnotationShapesProps) {
  if (!renderLabels) return null;
  return (
    <>
      {/* Boxes (only the active subject in box mode). The legend carries the standing
          symbology; a shape is named on the canvas only while selected or hovered. */}
      {mode === "box" &&
        boxes.map((b, i) =>
          b.subject === activeSubject ? (
            <BoxOverlay
              key={`box-${i}`}
              box={b}
              stroke={i === selectedBoxIdx ? "#00BFFF" : subjectColor(b.subject)}
              width={boxStroke}
              labelSize={labelSize}
              label={b.subject}
              showLabel={i === selectedBoxIdx || i === hoveredBoxIdx}
              selected={i === selectedBoxIdx}
              handleR={selVertR}
            />
          ) : null,
        )}

      {/* Read-only derived boxes: each active-subject polygon's bounding box, shown in box mode so a
          polygon's detection footprint is visible while boxing. Render-only, derived from
          polygonBbox here and never added to canvas.boxes, so it can't be selected/edited/deleted or
          saved (handle-less). Dashed marks it as read-only, distinct from a real editable box
          (solid), the same convention in-progress/under-review shapes already use. */}
      {mode === "box" &&
        polygons.map((p, i) =>
          p.subject === activeSubject ? (
            <BoxOverlay
              key={`derived-${i}`}
              box={derivedBoxFromPolygon(p)}
              stroke={subjectColor(p.subject)}
              width={boxStroke}
              labelSize={labelSize}
              label={p.subject}
              showLabel={i === hoveredDerivedIdx}
              dashed
            />
          ) : null,
        )}

      {/* Polygons */}
      {polygons.map((p, i) => {
        const selected = selectedPolygonIdx === i;
        const hovered = hoveredIdx === i;
        const dragging = draggingIdx === i;
        // Outside polygon mode only the selected polygon shows (the shape being inspected)
        if (mode !== "polygon" && !selected) return null;
        // In polygon mode filter to the active subject unless selected
        if (mode === "polygon" && !selected && p.subject !== activeSubject) return null;
        const showVerts = selected || hovered || dragging;
        return (
          <PolygonOverlay
            key={`poly-${i}`}
            polygon={p}
            stroke={selected ? "#00BFFF" : subjectColor(p.subject)}
            width={polyStroke}
            vertexRadius={selected ? selVertR : vertR}
            showVertices={showVerts}
            labelSize={labelSize}
            label={p.subject}
            showLabel={selected || hovered}
          />
        );
      })}

      {/* Points: the same visibility rule the agent's mirror uses (pointShapeVisible) */}
      {points.map((p, i) => {
        const selected = selectedPointIdx === i;
        if (
          !pointShapeVisible({
            mode,
            subject: p.subject,
            activeSubject: activeSubject ?? "",
            selected,
          })
        )
          return null;
        return (
          <PointOverlay
            key={`point-${i}`}
            point={p}
            stroke={selected ? "#00BFFF" : subjectColor(p.subject)}
            coreR={selected ? pointSelCoreR : pointCoreR}
            tickInner={pointTickInner}
            tickOuter={pointTickOuter}
            lineW={scaleLineW * 1.6}
            labelSize={labelSize}
            label={p.subject}
            showLabel={selected || i === hoveredPointIdx}
          />
        );
      })}
    </>
  );
});
