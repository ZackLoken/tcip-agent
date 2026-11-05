/**
 * The identity of the frontend source a bundle is built from, and the build record that carries it.
 *
 * The source is the files that shape the bundle: every file under the source directory
 * `tsconfig.json` includes first that is no test file (`isTestFile`), every file under `public`,
 * and the root's `index.html`,
 * `tsconfig.json`, `package.json`, `package-lock.json` and `vite.config.ts`. Its identity is a
 * SHA-256 over each file's root-relative POSIX path and the SHA-256 of its bytes, in path order;
 * a file absent from the root is absent from the identity.
 *
 * The build record is `{ source, record }`: the identity and the record's own file name. Registered
 * in the Vite configuration, `bundleRecord` has `vite build` write it into the output directory
 * beside the bundle, after the output is emptied.
 *
 * Run as `node scripts/bundleSource.mjs <root>`, it prints the record `<root>`'s source would
 * be built with.
 */
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, posix } from "node:path";
import { pathToFileURL } from "node:url";

import { isTestFile } from "./routeReads.mjs";

/** The build record's file name, written into the output directory. */
export const BUILD_RECORD = "build.json";

const ROOT_FILES = [
  "index.html",
  "tsconfig.json",
  "package.json",
  "package-lock.json",
  "vite.config.ts",
];

/**
 * Return the files of `directory`, a root-relative path, as root-relative POSIX paths, keeping
 * those whose path relative to `directory` `keep` accepts.
 *
 * @param {string} root
 * @param {string} directory
 * @param {(relative: string) => boolean} keep
 */
function filesUnder(root, directory, keep = () => true) {
  if (!existsSync(join(root, directory))) return [];
  return readdirSync(join(root, directory), { recursive: true })
    .filter((entry) => statSync(join(root, directory, entry)).isFile() && keep(entry))
    .map((entry) => posix.join(directory, ...entry.split(/[\\/]/)));
}

/**
 * Return the source files under `root` that shape its bundle, as sorted root-relative POSIX paths.
 *
 * @param {string} root
 */
export function sourceFiles(root) {
  const [src] = JSON.parse(readFileSync(join(root, "tsconfig.json"), "utf8")).include;
  return [
    ...filesUnder(root, src, (entry) => !isTestFile(entry)),
    ...filesUnder(root, "public"),
    ...ROOT_FILES.filter((file) => existsSync(join(root, file))),
  ].sort();
}

/**
 * Return the hex SHA-256 identity of the source under `root`.
 *
 * @param {string} root
 */
export function sourceHash(root) {
  const identity = createHash("sha256");
  for (const file of sourceFiles(root)) {
    const content = createHash("sha256")
      .update(readFileSync(join(root, file)))
      .digest("hex");
    identity.update(`${file}\0${content}\n`);
  }
  return identity.digest("hex");
}

/**
 * Return the build record of the source under `root`.
 *
 * @param {string} root
 */
export function buildRecord(root) {
  return { source: sourceHash(root), record: BUILD_RECORD };
}

/** Return the Vite plugin that has a build write its source's record into the output directory. */
export function bundleRecord() {
  let root = "";
  return {
    name: "bundle-record",
    apply: "build",
    configResolved(config) {
      root = config.root;
    },
    generateBundle() {
      this.emitFile({
        type: "asset",
        fileName: BUILD_RECORD,
        source: JSON.stringify(buildRecord(root)),
      });
    },
  };
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [root] = process.argv.slice(2);
  process.stdout.write(JSON.stringify(buildRecord(root)));
}
