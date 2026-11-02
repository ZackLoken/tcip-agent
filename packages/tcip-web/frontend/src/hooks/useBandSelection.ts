import { useCallback, useMemo } from "react";

import { type ImageBands, PLAIN } from "@/api/types.generated";
import {
  bandSetSignature,
  defaultBandSelection,
  NO_BAND_CHOICE,
  type BandChoice,
  type BandSelection,
} from "@/lib/bandSelection";
import { useStore } from "@/store";

/**
 * The breeder's band choice for the image `bandsInfo` describes, its selection held once per
 * band-set signature (the image's band names, in order) rather than per component, so a composite
 * chosen in one tab is the one the other renders over the same band set, and a detour through a
 * differently-banded image (or a plain color photo) neither applies the old selection nor
 * destroys it.
 *
 * `NO_BAND_CHOICE` while `bandsInfo` itself is `null` (loading, path-less, a failed fetch) or the
 * frame displays as its own pixels (the server's plain read): nothing is read or written then, so
 * a tab switch, which remounts this hook, neither wipes nor reseeds anything. Absent a stored
 * selection for a set, the default (`defaultBandSelection`) is answered but not recorded; the first
 * choice records it. The selection and the setter keep their identity across renders that change
 * neither the band set nor its stored selection.
 */
export function useBandSelection(bandsInfo: ImageBands | null): BandChoice {
  const signature = bandsInfo ? bandSetSignature(bandsInfo.bands) : "";
  const stored = useStore((s) => s.bandSelection.byBandSet[signature]);
  const setBandSelectionFor = useStore((s) => s.setBandSelectionFor);

  const fallback = useMemo(
    () => bandsInfo && defaultBandSelection(bandsInfo),
    [signature], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const choose = useCallback(
    (next: BandSelection) => setBandSelectionFor(signature, next),
    [setBandSelectionFor, signature],
  );

  return useMemo(
    () =>
      bandsInfo === null || bandsInfo.read.kind === PLAIN
        ? NO_BAND_CHOICE
        : { kind: "banded", info: bandsInfo, selection: stored ?? fallback!, choose },
    [bandsInfo, stored, fallback, choose],
  );
}
