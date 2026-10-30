import type { OverviewJob, ServingCell, ViewReads } from "@/api/types.generated";
import produced from "@/test/imageAnswers.json";
import type { Refusal } from "@/test/refusals";

/** What the image route answered: its status and the image headers it sent. */
interface ServedImage {
  status: number;
  headers: Record<string, string>;
}

/** What `tools/generate_images_fixture.py` received from the image routes: the native cells a
 *  view within the display's cap crosses (`native`), the overview level's cells a whole view of a
 *  built raster serves (`level`), and the cells of a view down a one-pixel-wide raster (`skinny`);
 *  the image route's status and image headers serving the first raster whole (`image`) and the
 *  one-pixel-wide raster whole (`image_skinny`), refusing a
 *  raster past the cap with no pyramid (`image_overviews_required`) and a file that does not exist
 *  (`image_missing`); an overview build joined while it runs (`build`), read running (`running`)
 *  and once completed (`completed`), a build over a file that is no raster (`failed`), and the
 *  refusal of a job no one started (`unknown`). */
// A JSON import reads the status as a string, which the literal union admits only through
// `unknown`.
export const IMAGE_ANSWERS = produced as unknown as {
  native: ViewReads;
  level: ViewReads;
  skinny: ViewReads;
  image: ServedImage;
  image_skinny: ServedImage;
  image_overviews_required: ServedImage;
  image_missing: ServedImage;
  build: OverviewJob;
  running: OverviewJob;
  completed: OverviewJob;
  failed: OverviewJob;
  unknown: Refusal<string>;
};

/** The native extent `reads` cover together: the frame of the raster a whole view of it read. */
export function extentOf(reads: ServingCell[]): { w: number; h: number } {
  return {
    w: Math.max(...reads.map((cell) => cell.nx1)),
    h: Math.max(...reads.map((cell) => cell.ny1)),
  };
}
