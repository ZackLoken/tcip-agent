import type { PriorityQueueJob, ScorerCatalog, SelectAnswer } from "@/api/types.generated";
import produced from "@/test/queueAnswers.json";
import type { Refusal } from "@/test/refusals";

/** What `tools/generate_queue_fixture.py` received from the review queue routes over the open
 *  fixture's photographs: the scorer catalog (`scorers`), the same with a sample factory taking a
 *  boolean, a list of floats and an optional float registered as `typed` (`typed_scorers`), a
 *  launch of it the launch route refused for a value of the wrong type, one not taken and its
 *  required ones unstated (`refused_launch`), the photographs' selection for the subject whose
 *  second photograph is finished (`selection`), a queue launched there (`launched`),
 *  polled while it ranks (`running`) and once it ended (`completed`), a diversity queue over a
 *  subject no photograph is finished for, ended on the scorer's refusal (`failed`), and the
 *  selection of the dataset's later capture, whose one photograph is named as the first is
 *  (`later_selection`). */
// A JSON import reads the status as a string, which the literal union admits only through
// `unknown`.
export const QUEUE_ANSWERS = produced as unknown as {
  scorers: ScorerCatalog;
  typed_scorers: ScorerCatalog;
  refused_launch: Refusal<unknown>;
  selection: SelectAnswer;
  launched: PriorityQueueJob;
  running: PriorityQueueJob;
  completed: PriorityQueueJob;
  failed: PriorityQueueJob;
  later_selection: SelectAnswer;
};
