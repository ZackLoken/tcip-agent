/**
 * A project's Vite configuration, as Vite's own serve and build entries resolve it.
 *
 * Run as `node scripts/viteConfig.mjs <root>`, it prints one JSON object for the configuration
 * under `<root>`: `proxy`, the dev server's proxy rules as `vite serve` resolves them; and
 * `outDir`, the absolute directory, and `assetsDir`, the subdirectory of it, the build writes as
 * `vite build` resolves them.
 */
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { createBuilder, resolveConfig } from "vite";

/**
 * Return the configuration under `root` as `vite serve` resolves it, before any server starts.
 *
 * @param {string} root
 */
export function servedConfig(root) {
  return resolveConfig({ root, logLevel: "silent" }, "serve");
}

/**
 * Return the configuration under `root` as Vite's builder resolves it, nothing built.
 *
 * @param {string} root
 */
export async function builtConfig(root) {
  return (await createBuilder({ root, logLevel: "silent" })).config;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [root] = process.argv.slice(2);
  const built = await builtConfig(root);
  process.stdout.write(
    JSON.stringify({
      proxy: (await servedConfig(root)).server.proxy,
      outDir: resolve(built.root, built.build.outDir),
      assetsDir: built.build.assetsDir,
    }),
  );
}
