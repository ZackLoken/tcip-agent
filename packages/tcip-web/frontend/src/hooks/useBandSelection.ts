import { useCallback, useMemo } from "react";

import type { ImageBandsResponse } from "@/api/client";
import {
  bandSetSignature,
  defaultBandSelection,
  isPlainColourFrame,
  type BandSelection,
} from "@/lib/bandSelection";
import { useStore } from "@/store";

/** A stable no-op setter for the non-applicable case, so a consumer that puts the returned setter
 *  in a dependency array never sees a new function identity on a render that changes nothing. */
function noopSetter(): void {}

/**
 * The breeder's band selection for the image `bandsInfo` describes, held once per band-set
 * signature (the image's band names, in order) rather than per component, so a composite chosen
 * in one tab is the one the other renders over the same band set, and a detour through a
 * differently-banded image (or a plain colour photo) neither applies the old selection nor
 * destroys it.
 *
 * `null` (with a no-op setter) while `bandsInfo` itself is `null` (loading, path-less, a failed
 * fetch), the frame is a plain colour photo with no band choice to make, or the frame has three or
 * fewer bands: nothing is read or written in any of those cases, so a tab switch, which remounts
 * this hook, neither wipes nor reseeds anything. Absent a stored selection for a set, the default
 * (`defaultBandSelection`) is returned but not recorded; the first change records it.
 */
export function useBandSelection(
  bandsInfo: ImageBandsResponse | null,
): [BandSelection | null, (next: BandSelection) => void] {
  const byBandSet = useStore((s) => s.bandSelection.byBandSet);
  const setBandSelectionFor = useStore((s) => s.setBandSelectionFor);

  const applicable = !!bandsInfo && bandsInfo.band_count > 3 && !isPlainColourFrame(bandsInfo);
  const signature = applicable ? bandSetSignature(bandsInfo.bands) : "";

  // Memoized so a consumer that keys a dependency array off the returned selection or setter
  // does not see a new object/function identity on every render that changes nothing.
  const defaultSelection = useMemo(
    () => (applicable && bandsInfo ? defaultBandSelection(bandsInfo.bands) : null),
    [applicable, signature], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const setSelection = useCallback(
    (next: BandSelection) => setBandSelectionFor(signature, next),
    [setBandSelectionFor, signature],
  );

  if (!applicable) {
    return [null, noopSetter];
  }
  return [byBandSet[signature] ?? defaultSelection, setSelection];
}
