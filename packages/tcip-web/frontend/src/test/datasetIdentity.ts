import type { AuditGapRefusal, RegisteredDataset, RootIdentity } from "@/api/types.generated";
import produced from "@/test/datasetIdentityAnswers.json";
import type { Refusal } from "@/test/refusals";

/**
 * The dataset identity reads and registrations `tools/generate_dataset_identity_fixture.py`
 * served in one project: `fresh` read unregistered; `mapped` read registered before any mapping
 * (`unmapped`), with one mapping over it (`mapped`), and with a mapping whose record will not read
 * and one whose plant registry will not read beside it (`mapped_damaged`); `damaged` read
 * refused; `fresh` registered, a crop the vocabulary does not hold refused, `damaged`'s id
 * restored, a change of `mapped`'s crop refused unconfirmed and made confirmed, and a registration
 * committed while its audit line failed (`unrecorded`).
 */
export const IDENTITY_ANSWERS = produced as unknown as {
  unregistered: RootIdentity;
  unmapped: RootIdentity;
  mapped: RootIdentity;
  mapped_damaged: RootIdentity;
  refused: RootIdentity;
  unknown_crop: Refusal<string>;
  registered: RegisteredDataset;
  repaired: RegisteredDataset;
  crop_unconfirmed: Refusal<string>;
  crop_changed: RegisteredDataset;
  unrecorded: Refusal<AuditGapRefusal>;
};
