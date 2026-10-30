import type {
  AuditGapRefusal,
  SessionPlace,
  SessionsListing,
  SessionWrite,
} from "@/api/types.generated";
import type { Refusal } from "@/test/refusals";
import produced from "@/test/sessionAnswers.json";

/** What `tools/generate_sessions_fixture.py` received from the sessions route: a contribution
 *  `landed` in a new session, the same contribution `repeated` once that session ended, an
 *  `empty` contribution with nothing to record, which lands in no session, the `audit_gap`
 *  refusal of a contribution that landed in a new session without its line, and the refusals of
 *  a contribution naming the ended session (`ended_named`) and of another person's naming the
 *  session the unlogged one landed in (`another_named`); the sessions the load route then lists
 *  (`listing`), and its refusal before any project was open (`unopened`). */
// A JSON import reads the status as a string, which the literal union admits only through
// `unknown`.
export const SESSION_ANSWERS = produced as unknown as {
  landed: SessionWrite;
  repeated: SessionWrite;
  empty: SessionWrite;
  audit_gap: Refusal<AuditGapRefusal & { committed: SessionWrite }>;
  ended_named: Refusal<string>;
  another_named: Refusal<string>;
  listing: SessionsListing;
  unopened: Refusal<string>;
};

/** The produced answers by the session they name: the `first` session landed in and the same
 *  session answered ended (`firstEnded`), and the `held` session the unlogged contribution landed
 *  in, opened after the first. */
export const SESSION_WRITES = {
  first: SESSION_ANSWERS.landed,
  firstEnded: SESSION_ANSWERS.repeated,
  held: SESSION_ANSWERS.audit_gap.body.detail.committed,
};

/** The session the produced answer `write` landed in; an answer that landed in none throws. */
export function sessionOf(write: SessionWrite): SessionPlace {
  if (!write.session) throw new Error("the produced answer landed in no session");
  return write.session;
}
