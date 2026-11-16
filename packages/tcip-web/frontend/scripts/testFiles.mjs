/**
 * The frontend's rule for a test file, importing no installed package.
 *
 * Run as `node scripts/testFiles.mjs`, it reads a JSON array of paths relative to the frontend's
 * root from standard input and prints a JSON array of whether each is a test file, judged on its
 * path relative to the source directory `tsconfig.json` includes first.
 */
import { readFileSync } from "node:fs";
import { join, relative } from "node:path";
import { pathToFileURL } from "node:url";

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
  const root = join(import.meta.dirname, "..");
  const [src] = JSON.parse(readFileSync(join(root, "tsconfig.json"), "utf8")).include;
  const paths = JSON.parse(readFileSync(0, "utf8"));
  process.stdout.write(JSON.stringify(paths.map((path) => isTestFile(relative(src, path)))));
}
