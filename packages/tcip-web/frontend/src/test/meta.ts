import type { MemoryPage_FrictionReport_, MemoryPage_Retrospective_ } from "@/api/types.generated";
import produced from "@/test/metaAnswers.json";

/** What `tools/generate_meta_fixture.py` received from the meta routes: the page of a project's
 *  one friction report (`reports`), and of its two retrospectives, the second a record that will
 *  not read (`retrospectives`). */
// A JSON import reads the category as a string, which the literal union admits only through
// `unknown`.
export const META_ANSWERS = produced as unknown as {
  reports: MemoryPage_FrictionReport_;
  retrospectives: MemoryPage_Retrospective_;
};
