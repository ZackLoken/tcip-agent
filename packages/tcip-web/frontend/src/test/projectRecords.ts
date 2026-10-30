import type { api } from "@/api/client";
import type {
  OpenAnswer,
  ProjectNotOpenRefusal,
  ProjectRemoval,
  RecordCorrection,
} from "@/api/types.generated";
import { openable, type OpenableProject } from "@/lib/openProject";
import fixtures from "@/test/projectFixtures.json";
import type { Refusal } from "@/test/refusals";

type Listing = Awaited<ReturnType<typeof api.projects.list>>;

/**
 * What `tools/generate_project_fixture.py` served from the platform's own doors: the `listing`
 * `GET /api/projects` answers for two created projects, the `correction` the invariances route
 * answers for one correction of the second, the route's `refusal` of an out-of-range block, the
 * second's open answer (`corrected_open`), and, with the second open, the canvas push route's
 * `project_not_open` refusal of a push naming the first; then, in the picker's workspace, its `empty_listing` before any project, its `picker`
 * listing and `picker_open` once `Valley farm` is open, the `removed` answer of a removal, the
 * `renamed` answer of `Valley farm`'s rename, and the `opened` answers of `Valley farm`, `Ridge
 * farm` and `Empty block`, in that order.
 */
export const PROJECT_FIXTURES = fixtures as unknown as {
  listing: Listing;
  correction: RecordCorrection;
  refusal: Refusal<unknown[]>;
  corrected_open: OpenAnswer;
  project_not_open: Refusal<ProjectNotOpenRefusal>;
  empty_listing: Listing;
  picker: Listing;
  opened: OpenAnswer[];
  picker_open: Listing;
  removed: ProjectRemoval;
  renamed: RecordCorrection;
};

const OPENABLE = PROJECT_FIXTURES.picker.projects.filter(openable);

/** The picker listing's project whose record names it `name`. */
const named = (name: string) => OPENABLE.find((p) => p.record.display_name === name)!;

/** The picker listing's projects: `valley` labeled with two subjects on its first date and one on
 *  its second, published over on the second, beside an unlabeled newest date; `ridge` holding two
 *  unlabeled dates; `empty` no image; and `stray` the directory whose record will not read. */
export const PICKER = {
  valley: named("Valley farm"),
  ridge: named("Ridge farm"),
  empty: named("Empty block"),
  stray: PROJECT_FIXTURES.picker.projects.find((p) => p.record === null)!,
};

/** The produced open answer of the listed project `p`; a project the generator opened no answer
 *  of throws. */
export function openedAs(p: OpenableProject): OpenAnswer {
  const found = PROJECT_FIXTURES.opened.find((answer) => answer.project.id === p.record.id);
  if (!found) throw new Error(`the generator opened no answer of ${p.record.display_name}`);
  return found;
}
