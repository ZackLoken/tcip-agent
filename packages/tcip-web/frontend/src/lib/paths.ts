/** The paths the browser reads off the dataset selection. */

import type { DatasetSelection } from "@/store/types";

/** Where one image's bytes live under an already-resolved image directory. */
export function inImagesDir(imagesDir: string, imageName: string): string {
  return `${imagesDir}/${imageName}`;
}

/** Where the bytes of one image on the selected date live, or null when nothing is selected. */
export function imagePath(dataset: DatasetSelection, imageName: string | null): string | null {
  return imageName && dataset.images_dir ? inImagesDir(dataset.images_dir, imageName) : null;
}

/** Whether a path names a file directly inside a directory, both already-resolved strings from
 *  the same backend (so a shared separator convention needs no normalizing here). Null either
 *  side answers false rather than throwing, since a canvas or a dataset can carry no path yet. */
export function pathInDir(path: string | null, dir: string | null): boolean {
  return !!path && !!dir && path.startsWith(`${dir}/`);
}

/** The image the selection currently points at: its name and where its bytes live. */
export function currentImage(dataset: DatasetSelection): {
  name: string | null;
  path: string | null;
} {
  const name = dataset.image_list[dataset.current_image_index] ?? null;
  return { name, path: imagePath(dataset, name) };
}
