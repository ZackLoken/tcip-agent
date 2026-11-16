import { render, renderHook } from "@testing-library/react";
import type { ReactElement } from "react";

import { useJobListings } from "@/hooks/useJobListings";

/** Render `tab` with the job listings reader mounted beside it, as the project shell mounts both,
 *  so the tab's listings come from the routes the test answers. */
export function renderBesideListings(tab: ReactElement) {
  renderHook(useJobListings);
  return render(tab);
}
