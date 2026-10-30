import type {
  BucketExistsRefusal,
  InferenceJob,
  InferenceJobFrame,
  InferenceJobs,
  MissingJobFrame,
} from "@/api/types.generated";
import produced from "@/test/inferenceAnswers.json";
import type { Refusal } from "@/test/refusals";

/** What `tools/generate_inference_fixture.py` received from the inference routes. A pass that
 *  fails is listed running before its engine starts (`failing_started`) and streams its failure
 *  (`failing_final`). A held pass is launched (`launched`) and listed beside the failed one before
 *  its engine reports (`held_started`) and at one image of nine (`running`), as its stream sends it
 *  (`progress`); a second launch naming its bucket is refused (`bucket_exists`); it is canceled
 *  while it still runs (`cancel`), then listed (`canceled`) and streamed (`final`) once it ends. A
 *  launch naming a checkpoint that does not exist is refused (`missing_checkpoint`), and a stream
 *  to a job no one launched answers so (`unknown_job`). */
// A JSON import reads the status as a string, which the literal union admits only through
// `unknown`.
export const INFERENCE_ANSWERS = produced as unknown as {
  failing_started: InferenceJobs;
  failing_final: InferenceJobFrame;
  launched: InferenceJob;
  held_started: InferenceJobs;
  running: InferenceJobs;
  progress: InferenceJobFrame;
  bucket_exists: Refusal<BucketExistsRefusal>;
  cancel: InferenceJob;
  canceled: InferenceJobs;
  final: InferenceJobFrame;
  missing_checkpoint: Refusal<string>;
  unknown_job: MissingJobFrame;
};
