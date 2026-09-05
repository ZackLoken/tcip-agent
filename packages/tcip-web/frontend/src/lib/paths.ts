/** The paths the browser handles: pasted filesystem paths, and the dataset's own records. */

import type { DatasetSelection } from "@/store/types";

/**
 * Trim whitespace and strip surrounding quotes. Windows "Copy as path" wraps the path in
 * double quotes, and pasting from an address bar often adds stray spaces; either makes
 * the backend reject an otherwise-correct path, so clean it before use.
 */
export function cleanPath(raw: string): string {
  return raw
    .trim()
    .replace(/^["']+|["']+$/g, "")
    .trim();
}

/**
 * The extension one image's label or prediction record is written under.
 *
 * The backend's dataset_layout owns this rule and resolves every directory the browser is given;
 * this is the one place the browser states the rest of the record's name, and
 * tests/test_frontend_dataset_vocabulary.py holds it equal to the backend's own value.
 */
export const RECORD_EXT = ".json";

function stemOf(imageName: string): string {
  return imageName.replace(/\.[^.]+$/, "");
}

function inDir(dir: string | null, fileName: string): string | null {
  return dir ? `${dir}/${fileName}` : null;
}

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

/** One image's ground-truth record on the selected date, or null when there is no label dir. */
export function labelPath(dataset: DatasetSelection, imageName: string | null): string | null {
  return imageName ? inDir(dataset.annotations_dir, stemOf(imageName) + RECORD_EXT) : null;
}

/** One image's prediction record in the selected model bucket, or null when none is selected. */
export function predictionPath(dataset: DatasetSelection, imageName: string | null): string | null {
  return imageName ? inDir(dataset.predictions_dir, stemOf(imageName) + RECORD_EXT) : null;
}

/** The image the selection currently points at: its name and where its bytes live. */
export function currentImage(dataset: DatasetSelection): {
  name: string | null;
  path: string | null;
} {
  const name = dataset.image_list[dataset.current_image_index] ?? null;
  return { name, path: imagePath(dataset, name) };
}
