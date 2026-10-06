import { memo } from "react";

import { HaloLabel } from "@/components/HaloLabel";
import { FLAG_MARK } from "@/lib/symbology";
import type { Flag } from "@/store/types";

/** The mark of each open flag that has a place on the image, drawn at that place. */
export const FlagMarks = memo(function FlagMarks({
  places,
  size,
}: {
  places: { at: [number, number]; flag: Flag }[];
  size: number;
}) {
  return (
    <>
      {places.map(({ at, flag }) => (
        <HaloLabel
          key={flag.id}
          x={at[0]}
          y={at[1]}
          text={FLAG_MARK.glyph}
          fill={FLAG_MARK.color}
          size={size * 1.6}
        />
      ))}
    </>
  );
});
