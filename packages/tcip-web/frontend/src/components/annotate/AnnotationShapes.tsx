import { memo } from "react";

import { BoxOverlay } from "@/components/annotate/BoxOverlay";
import { PointOverlay } from "@/components/annotate/PointOverlay";
import { PolygonOverlay } from "@/components/annotate/PolygonOverlay";
import { shapeVisible } from "@/lib/canvasSync";
import { derivedBoxFromPolygon } from "@/lib/polygonGeometry";
import type { ReviewItem } from "@/lib/reviewItems";
import { useSubjectColors } from "@/lib/subjectColors";
import {
  authorshipLabel,
  lineStyleOf,
  outlineColor,
  type ReviewStatus,
  type strokeWidths,
} from "@/lib/symbology";
import type { Box, Mode, PointShape, PolygonShape } from "@/store/types";

/**
 * The committed boxes, polygons and points (content layer). Memoized, and crucially, the mouse
 * cursor is not one of its props, so a mouse move (which only updates cursor-following
 * overlays) does not re-render/reconcile these hundreds to thousands of Konva nodes. It
 * re-renders only when the shapes, the focus, the active subject, the statuses or the
 * zoom-derived sizes change.
 */
interface AnnotationShapesProps {
  boxes: Box[];
  polygons: PolygonShape[];
  points: PointShape[];
  mode: Mode;
  activeSubject: string | null;
  /** Each array's review statuses, aligned with it (`reviewStatuses`). */
  statuses: {
    boxes: (ReviewStatus | null)[];
    polygons: (ReviewStatus | null)[];
    points: (ReviewStatus | null)[];
  };
  /** The one focused item; an annotation among these draws its halo, label and handles. */
  focused: ReviewItem | null;
  hoveredIdx: number | null;
  draggingIdx: number | undefined;
  renderLabels: boolean;
  widths: ReturnType<typeof strokeWidths>;
}

export const AnnotationShapes = memo(function AnnotationShapes({
  boxes,
  polygons,
  points,
  mode,
  activeSubject,
  statuses,
  focused,
  hoveredIdx,
  draggingIdx,
  renderLabels,
  widths,
}: AnnotationShapesProps) {
  useSubjectColors(); // re-render on a recolor: outlineColor() below reads subjectColor fresh
  if (!renderLabels) return null;
  const isFocused = (shape: ReviewItem["shape"], i: number) =>
    focused?.kind === "annotation" && focused.shape === shape && focused.ref === i;
  const visible = (kind: "box" | "derived" | "polygon" | "point", subject: string, at: boolean) =>
    shapeVisible({ kind, mode, subject, activeSubject: activeSubject ?? "", focused: at });
  // The legend carries the standing symbology; a shape is named on the canvas only while focused.
  const named = (shape: { subject: string; authorship?: string | null }, at: boolean) => ({
    style: lineStyleOf(shape.authorship),
    labelSize: widths.labelSize,
    label: authorshipLabel(shape.subject, shape.authorship),
    showLabel: at,
    focused: at,
  });
  return (
    <>
      {boxes.map((b, i) => {
        const at = isFocused("box", i);
        if (!visible("box", b.subject, at)) return null;
        return (
          <BoxOverlay
            key={`box-${i}`}
            box={b}
            {...named(b, at)}
            stroke={outlineColor(b.subject, statuses.boxes[i])}
            width={widths.boxStroke}
            handleR={widths.selVertR}
          />
        );
      })}

      {/* Read-only derived boxes: a polygon's bounding box, shown while boxing so its detection
          footprint is visible. Derived from the rings here and never added to canvas.boxes, so
          it can't be focused, edited, deleted or saved; it draws in the polygon's own style. */}
      {polygons.map((p, i) =>
        visible("derived", p.subject, false) ? (
          <BoxOverlay
            key={`derived-${i}`}
            box={derivedBoxFromPolygon(p)}
            {...named(p, false)}
            stroke={outlineColor(p.subject, statuses.polygons[i])}
            width={widths.boxStroke}
          />
        ) : null,
      )}

      {polygons.map((p, i) => {
        const at = isFocused("polygon", i);
        if (!visible("polygon", p.subject, at)) return null;
        return (
          <PolygonOverlay
            key={`poly-${i}`}
            polygon={p}
            {...named(p, at)}
            stroke={outlineColor(p.subject, statuses.polygons[i])}
            width={widths.polyStroke}
            vertexRadius={at ? widths.selVertR : widths.vertR}
            showVertices={at || hoveredIdx === i || draggingIdx === i}
          />
        );
      })}

      {points.map((p, i) => {
        const at = isFocused("point", i);
        if (!visible("point", p.subject, at)) return null;
        return (
          <PointOverlay
            key={`point-${i}`}
            point={p}
            {...named(p, at)}
            stroke={outlineColor(p.subject, statuses.points[i])}
            coreR={at ? widths.pointSelCoreR : widths.pointCoreR}
            tickInner={widths.pointTickInner}
            tickOuter={widths.pointTickOuter}
            lineW={widths.scaleLineW * 1.6}
          />
        );
      })}
    </>
  );
});
