/**
 * The frontend's rule for a test file and the source directory it judges paths against, importing
 * no installed package.
 *
 * Run as `node scripts/testFiles.mjs`, it reads a JSON array of paths relative to the frontend's
 * root from standard input and prints a JSON array of whether each is a test file, judged on its
 * path relative to the source directory.
 */
import { readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { pathToFileURL } from "node:url";

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

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const src = sourceDirectory(join(import.meta.dirname, ".."));
  const paths = JSON.parse(readFileSync(0, "utf8"));
  process.stdout.write(JSON.stringify(paths.map((path) => isTestFile(relative(src, path)))));
}
