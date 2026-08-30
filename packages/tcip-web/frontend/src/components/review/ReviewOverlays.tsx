import { Fragment, memo, type ReactNode } from "react";

import { HaloLabel } from "@/components/HaloLabel";
import { ReviewLine } from "@/components/review/ReviewLine";
import { ReviewPoint } from "@/components/review/ReviewPoint";
import { ReviewRect } from "@/components/review/ReviewRect";
import type { ReviewColors } from "@/lib/reviewColors";
import {
  annotationGeometry,
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
}: OverlayProps) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const ACTIVE_COLOR = colors.active;

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

        if (active) {
          nodes.push(
            <HaloLabel
              key="lbl"
              x={d.bbox[0]}
              y={d.bbox[1]}
              text={`${d.class_name}${d.conf !== null ? ` ${d.conf.toFixed(2)}` : ""}`}
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
