import type { StateCreator } from "zustand";

import type { BandSelection } from "@/lib/bandSelection";
import type { AppState } from "@/store/appState";

interface BandSelectionState {
  /** The breeder's chosen band composite for each distinct band set this session has seen, keyed
   *  by bandSetSignature (the image's band names, in order). Absent means no change has been made
   *  for that set yet; useBandSelection is the one reader/writer, never the map directly. */
  byBandSet: Record<string, BandSelection>;
}

export interface BandSelectionSlice {
  /** Band-composite selection, held per band-set signature; see useBandSelection. */
  bandSelection: BandSelectionState;
  setBandSelectionFor: (signature: string, selection: BandSelection) => void;
}

export const createBandSelectionSlice: StateCreator<AppState, [], [], BandSelectionSlice> = (
  set,
) => ({
  bandSelection: { byBandSet: {} },
  setBandSelectionFor: (signature, selection) =>
    set((s) => ({
      bandSelection: { byBandSet: { ...s.bandSelection.byBandSet, [signature]: selection } },
    })),
});
