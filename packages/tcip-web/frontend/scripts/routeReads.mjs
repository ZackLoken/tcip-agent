/**
 * The frontend's non-test sources, their reads of the generated route module and their backend
 * path literals, as the TypeScript compiler parses them.
 *
 * A non-test source is a `.ts` or `.tsx` file under the source directory whose name carries no
 * `.test.` and that lies outside the directory's `test` folder. A read is a property access whose
 * object is the identifier `ROUTES`; a destructuring or an element access of `ROUTES` is not one.
 * A backend path literal is a string or template literal starting `/api/` or `/ws/`.
 *
 * Run as `node scripts/routeReads.mjs <src>`, it prints one JSON object over the non-test sources
 * under `<src>`: `sources`, their sorted absolute paths; `reads`, the sorted names they read; and
 * `literals`, each backend path literal as its `file` and `text`.
 */
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import ts from "typescript";

/**
 * Return whether `relative`, a path relative to the source directory, is a non-test source.
 *
 * @param {string} relative
 */
export function isNonTestSource(relative) {
  const parts = relative.split(/[\\/]/);
  return /\.tsx?$/.test(relative) && !parts.at(-1).includes(".test.") && parts[0] !== "test";
}

/**
 * Return the non-test sources under the source directory `src`, as sorted absolute paths.
 *
 * @param {string} src
 */
export function nonTestSources(src) {
  return readdirSync(src, { recursive: true })
    .filter(isNonTestSource)
    .map((file) => join(src, file))
    .sort();
}

/** Call `see` on every syntax node of each of `files`, with the file it lies in. */
function eachNode(files, see) {
  const visit = (file) => (node) => {
    see(node, file);
    ts.forEachChild(node, visit(file));
  };
  for (const file of files) {
    const text = readFileSync(file, "utf8");
    visit(file)(ts.createSourceFile(file, text, ts.ScriptTarget.Latest));
  }
}

/** Return the sorted names `files` read as `ROUTES.<name>`. */
export function routeReads(files) {
  const names = new Set();
  eachNode(files, (node) => {
    if (
      ts.isPropertyAccessExpression(node) &&
      ts.isIdentifier(node.expression) &&
      node.expression.text === "ROUTES"
    ) {
      names.add(node.name.text);
    }
  });
  return [...names].sort();
}

/** Return each backend path literal in `files` as its `file` and its `text`. */
export function backendPathLiterals(files) {
  const found = [];
  eachNode(files, (node, file) => {
    const text =
      ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)
        ? node.text
        : ts.isTemplateExpression(node)
          ? node.head.text
          : null;
    if (text !== null && /^\/(api|ws)\//.test(text)) found.push({ file, text });
  });
  return found;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const sources = nonTestSources(process.argv[2]);
  process.stdout.write(
    JSON.stringify({
      sources,
      reads: routeReads(sources),
      literals: backendPathLiterals(sources),
    }),
  );
}
