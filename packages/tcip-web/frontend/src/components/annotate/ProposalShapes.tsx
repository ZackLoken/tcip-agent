import { memo } from "react";

import { BoxOverlay } from "@/components/annotate/BoxOverlay";
import { PolygonOverlay } from "@/components/annotate/PolygonOverlay";
import { annotationsToCanvas } from "@/lib/labelSerde";
import { focusesProposal, proposalMatch, type ReviewItem } from "@/lib/reviewItems";
import { outlineColor, proposalLabel, type strokeWidths } from "@/lib/symbology";
import type { Proposal } from "@/store/types";

/** The proposals the canvas shows, drawn in the tool's dotted stroke and their match color; the
 *  focused one draws its halo, and it and every unpaired proposal carry a label. */
export const ProposalShapes = memo(function ProposalShapes({
  proposals,
  focused,
  widths,
}: {
  proposals: Proposal[];
  focused: ReviewItem | null;
  widths: ReturnType<typeof strokeWidths>;
}) {
  return (
    <>
      {proposals.map((p) => {
        const isFocused = focusesProposal(focused, p.index);
        const label = proposalLabel(p, isFocused);
        const shared = {
          stroke: outlineColor(p.subject, proposalMatch(p)),
          style: "dotted" as const,
          labelSize: widths.labelSize,
          label: label ?? "",
          showLabel: label !== null,
          focused: isFocused,
        };
        const { boxes, polygons } = annotationsToCanvas([p]);
        return [
          ...boxes.map((b) => (
            <BoxOverlay key={`proposal-${p.index}`} box={b} width={widths.boxStroke} {...shared} />
          )),
          ...polygons.map((poly) => (
            <PolygonOverlay
              key={`proposal-${p.index}`}
              polygon={poly}
              width={widths.polyStroke}
              vertexRadius={0}
              showVertices={false}
              {...shared}
            />
          )),
        ];
      })}
    </>
  );
});
