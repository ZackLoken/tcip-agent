import { memo } from "react";

import { subjectColor } from "@/api/subjects";
import { BoxOverlay } from "@/components/annotate/BoxOverlay";
import { PolygonOverlay } from "@/components/annotate/PolygonOverlay";
import { annotationsToCanvas } from "@/lib/labelSerde";
import type { Proposal } from "@/store/types";

/** The outline opacity of a proposal the bucket's operating point does not admit. Provisional: a
 *  display choice, set by eye, that only has to read as dimmer than a full stroke. */
const BELOW_OPERATING_POINT_OPACITY = 0.4;

/** The proposals the canvas shows, drawn by the annotation overlays in the tool's dotted stroke,
 *  dimmed when the bucket's operating point does not admit them; the selected one draws in the
 *  selection color with its label. */
export const ProposalShapes = memo(function ProposalShapes({
  proposals,
  selected,
  strokeW,
  scaleLineW,
}: {
  proposals: Proposal[];
  selected: number | null;
  strokeW: number;
  scaleLineW: number;
}) {
  const labelSize = 11 * scaleLineW;
  return (
    <>
      {proposals.map((p) => {
        const isSelected = p.index === selected;
        const stroke = isSelected ? "#00BFFF" : subjectColor(p.subject);
        const label =
          `${p.subject} proposal${p.score != null ? ` ${p.score.toFixed(2)}` : ""}` +
          (p.paired !== null ? " (pairs with an annotation)" : "");
        const strokeOpacity = p.admitted ? undefined : BELOW_OPERATING_POINT_OPACITY;
        const { boxes, polygons } = annotationsToCanvas([p]);
        return [
          ...boxes.map((b) => (
            <BoxOverlay
              key={`proposal-${p.index}`}
              box={b}
              stroke={stroke}
              width={strokeW}
              labelSize={labelSize}
              label={label}
              showLabel={isSelected}
              dashed="tool"
              strokeOpacity={strokeOpacity}
            />
          )),
          ...polygons.map((poly) => (
            <PolygonOverlay
              key={`proposal-${p.index}`}
              polygon={poly}
              stroke={stroke}
              width={strokeW}
              vertexRadius={0}
              showVertices={false}
              labelSize={labelSize}
              label={label}
              showLabel={isSelected}
              dashed="tool"
              strokeOpacity={strokeOpacity}
            />
          )),
        ];
      })}
    </>
  );
});
