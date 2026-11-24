/**
 * The frontend's rule for a test file and the source directory it judges paths against, importing
 * no installed package.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";

/**
 * Return the source directory of the frontend at `root`, relative to it: the directory its
 * `tsconfig.json` includes first.
 *
 * @param {string} root
 */
export function sourceDirectory(root) {
  return JSON.parse(readFileSync(join(root, "tsconfig.json"), "utf8")).include[0];
}

/**
 * Return whether `relative`, a path relative to the source directory, is a test file: one whose
 * name carries `.test.`, or one inside the directory's `test` folder.
 *
 * @param {string} relative
 */
export function isTestFile(relative) {
  const parts = relative.split(/[\\/]/);
  return parts.at(-1).includes(".test.") || parts[0] === "test";
}
