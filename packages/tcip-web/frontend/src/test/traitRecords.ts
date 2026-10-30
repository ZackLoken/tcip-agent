import type { ConfirmedRevision, RevisionMovedRefusal, TraitsListing } from "@/api/types.generated";
import type { Refusal } from "@/test/refusals";
import produced from "@/test/traitListings.json";

/**
 * The trait listings `tools/generate_trait_fixture.py` served from the platform's own proposal,
 * confirmation and listing: `setup` holds `stem` with revision 1 confirmed and revision 2
 * proposed, `unconfirmed` holds `trait_a` proposed only, `results` the same revision
 * confirmed, and `unreadable` a trait whose record will not read.
 */
export const TRAIT_LISTINGS = produced.listings as unknown as Record<
  "setup" | "unconfirmed" | "results" | "unreadable",
  TraitsListing
>;

/** The confirmation route's answers over `setup`'s revisions, once its listing was served: its
 *  refusal of revision 2 named by revision 1's hash, revision 2 `confirmed`, revision 1
 *  `withdrawn`, and revision 2 withdrawn while its audit line failed (`unrecorded`). */
export const CONFIRM_ANSWERS = produced.answers as unknown as {
  revision_moved: Refusal<RevisionMovedRefusal>;
  confirmed: ConfirmedRevision;
  withdrawn: ConfirmedRevision;
  unrecorded: ConfirmedRevision;
};
