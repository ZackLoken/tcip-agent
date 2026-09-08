import { Fragment, memo, type ReactNode } from "react";
import { Rect } from "react-konva";

import { HaloLabel } from "@/components/HaloLabel";
import { ReviewLine } from "@/components/review/ReviewLine";
import { ReviewPoint } from "@/components/review/ReviewPoint";
import { ReviewRect } from "@/components/review/ReviewRect";
import type { ReviewColors } from "@/lib/reviewColors";
import {
  annotationGeometry,
  detectionAdmitted,
  detGtAnnotation,
  detPredAnnotation,
  type ReviewGeom,
} from "@/lib/reviewGeometry";
import { useStore } from "@/store";
import type { MatchesResponse } from "@/store/types";

interface OverlayProps {
  matches: MatchesResponse;
  focusedIdx: number;
  showGT: boolean;
  showPred: boolean;
  colors: ReviewColors;
  /** While editing, the picked-up shape is hidden here; it renders live in the edit overlay. */
  suppressFocusedGt?: boolean;
  suppressFocusedPred?: boolean;
  /** The bucket's own validated count operating point (admission_rule_of), or null: a detection
   *  at or above it draws a corner mark on its prediction's geometry. Null under a classified
   *  scope or an unvalidated bucket, so no mark is ever drawn there. */
  admissionConf?: number | null;
}

/** A prediction's top-left corner (a box's own, or a polygon's first ring's first vertex) and its
 *  shorter side, for sizing the corner mark; null for a point (never marked) or an empty ring. */
function markAnchor(geom: ReviewGeom): { x: number; y: number; shortSide: number } | null {
  if (geom.kind === "box") {
    const [x1, y1, x2, y2] = geom.box;
    return { x: x1, y: y1, shortSide: Math.min(x2 - x1, y2 - y1) };
  }
  if (geom.kind === "polygon") {
    const ring = geom.rings[0];
    if (!ring || !ring.length) return null;
    const xs = ring.map(([x]) => x);
    const ys = ring.map(([, y]) => y);
    return {
      x: ring[0][0],
      y: ring[0][1],
      shortSide: Math.min(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)),
    };
  }
  return null;
}

/** Memoized, and scale is read from the store internally (not a prop) so pan/zoom re-renders
 *  of ReviewTab don't rebuild this O(detection count) shape list; it re-runs only when the
 *  matches/filters/colors props actually change (or its own scale subscription fires). */
export const ReviewOverlays = memo(function ReviewOverlays({
  matches,
  focusedIdx,
  showGT,
  showPred,
  colors,
  suppressFocusedGt,
  suppressFocusedPred,
  admissionConf,
}: OverlayProps) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const ACTIVE_COLOR = colors.active;

  /** A filled square at the prediction's own top-left corner, in the detection's outcome colour
   *  (never the active colour, so the mark keeps its meaning on the focused detection): the
   *  channel that names a rule pre-admitted this box, distinct from colour, dash and fill. A
   *  thin white outline keeps it legible on a same-colour corner (a tp's box coinciding with its
   *  ground truth, or under the focused dash) where the fill alone would vanish into the box. */
  const drawAdmittedMark = (geom: ReviewGeom | null, outcome: string): ReactNode => {
    const anchor = geom ? markAnchor(geom) : null;
    if (!anchor) return null;
    const side = Math.min(6 * lw, anchor.shortSide / 3);
    if (side <= 0) return null;
    return (
      <Rect
        key="admitted"
        x={anchor.x}
        y={anchor.y}
        width={side}
        height={side}
        fill={outcome}
        stroke="#ffffff"
        strokeWidth={lw}
      />
    );
  };

  /** Every detection renders by its own annotation's geometry: a box stays a box, a polygon stays
   *  a polygon, a point stays a point, and no kind is hidden (hiding one is an unreviewed
   *  false-negative). Every ring of a polygon draws too, in the same stroke: a verdict on an
   *  occlusion-split shape is a verdict on all of it, so a truncated render would be a verdict on
   *  something the reviewer never saw. */
  const drawGeom = (
    key: string,
    geom: ReviewGeom | null,
    stroke: string,
    weight: number,
    dashed: boolean,
    fill: string | undefined,
  ): ReactNode => {
    if (!geom) return null;
    if (geom.kind === "point") {
      return <ReviewPoint key={key} point={geom.point} stroke={stroke} lw={lw} weight={weight} />;
    }
    if (geom.kind === "box") {
      return (
        <ReviewRect
          key={key}
          box={geom.box}
          stroke={stroke}
          lw={lw}
          weight={weight}
          dashed={dashed}
          fill={fill}
        />
      );
    }
    return (
      <Fragment key={key}>
        {geom.rings.map((ring, ri) => (
          <ReviewLine
            key={ri}
            points={ring}
            stroke={stroke}
            lw={lw}
            weight={weight}
            dashed={dashed}
            fill={fill}
          />
        ))}
      </Fragment>
    );
  };

  // Non-active first, the active detection last so its blue overlay sits on top.
  const order = matches.detections
    .map((_, i) => i)
    .sort((a, b) => (a === focusedIdx ? 1 : 0) - (b === focusedIdx ? 1 : 0));

  return (
    <>
      {order.map((i) => {
        const d = matches.detections[i];
        const active = i === focusedIdx;
        const outcome = colors[d.det_type];
        const weight = active ? 3 : 2;
        const nodes: ReactNode[] = [];

        if (d.det_type === "fp") {
          // FP = a prediction with no GT. Solid outcome red as context; the detection under
          // review turns dashed blue (see the review legend).
          if (showPred && !(active && suppressFocusedPred)) {
            const stroke = active ? ACTIVE_COLOR : outcome;
            nodes.push(
              drawGeom(
                "fp",
                annotationGeometry(detPredAnnotation(d, matches)),
                stroke,
                weight,
                active,
                `${stroke}26`,
              ),
            );
          }
        } else {
          // TP / FN = ground truth, solid. Active FN turns blue; active TP keeps its green GT.
          if (showGT && !(active && suppressFocusedGt)) {
            const activeFn = active && d.det_type === "fn";
            const stroke = activeFn ? ACTIVE_COLOR : outcome;
            /** The active FN has no prediction, so its GT is the thing under review; draw it
             *  dashed blue like every other under-review shape so it matches the "Under review"
             *  legend entry instead of reading as a solid outcome box. A faint blue wash reads
             *  through. */
            const fill = activeFn ? `${ACTIVE_COLOR}26` : d.reviewed ? `${outcome}26` : undefined;
            nodes.push(
              drawGeom(
                "gt",
                annotationGeometry(detGtAnnotation(d, matches)),
                stroke,
                weight,
                activeFn,
                fill,
              ),
            );
          }
          // The TP under review also shows its prediction as a dashed-blue overlay (pred vs GT).
          if (active && d.det_type === "tp" && showPred && !suppressFocusedPred) {
            nodes.push(
              drawGeom(
                "tp-pred",
                annotationGeometry(detPredAnnotation(d, matches)),
                ACTIVE_COLOR,
                3,
                true,
                `${ACTIVE_COLOR}26`,
              ),
            );
          }
        }

        // A mark is drawn only when a shape of this detection was actually drawn above: an
        // admitted tp with ground truth hidden has no shape to anchor a lone square on.
        const shapeDrawn = nodes.length > 0;
        // Admitted: scored at or above the bucket's own validated count operating point, under a
        // detector scope, drawn only under showPred since it is the prediction's own mark.
        const admitted =
          shapeDrawn &&
          showPred &&
          !matches.attribute &&
          detectionAdmitted(d, matches, admissionConf ?? null) &&
          !(active && suppressFocusedPred);
        if (admitted) {
          nodes.push(drawAdmittedMark(annotationGeometry(detPredAnnotation(d, matches)), outcome));
        }

        if (active && shapeDrawn) {
          nodes.push(
            <HaloLabel
              key="lbl"
              x={d.bbox[0]}
              y={d.bbox[1]}
              text={
                `${d.class_name}${d.conf !== null ? ` ${d.conf.toFixed(2)}` : ""}` +
                (admitted ? `, admitted at or above ${admissionConf!.toFixed(2)}` : "")
              }
              fill={ACTIVE_COLOR}
              size={11 * lw}
            />,
          );
        }

        return <Fragment key={`det-${i}`}>{nodes}</Fragment>;
      })}
    </>
  );
});
