import type { TraitsListing } from "@/api/inference";
import listings from "@/test/traitListings.json";

/**
 * The trait listings `tools/generate_trait_fixture.py` served from the platform's own proposal,
 * confirmation and listing: `setup` holds `stem` with revision 1 confirmed and revision 2
 * proposed, `unconfirmed` holds `subject_a` proposed only, and `results` the same revision
 * confirmed.
 */
export const TRAIT_LISTINGS = listings as Record<
  "setup" | "unconfirmed" | "results",
  TraitsListing
>;
