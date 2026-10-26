/**
 * The dev server's proxy rules, as Vite loads them from a project's own configuration.
 *
 * Run as `node scripts/devServerProxy.mjs <root>`, it prints the `server.proxy` of the Vite
 * configuration under `<root>`, loaded for `vite serve`, as one JSON object.
 */
import { pathToFileURL } from "node:url";

import { loadConfigFromFile } from "vite";

/**
 * Return the `server.proxy` of the Vite configuration file under `root`, loaded for `vite serve`.
 *
 * @param {string} root
 */
export async function devServerProxy(root) {
  const loaded = await loadConfigFromFile(
    { command: "serve", mode: "development" },
    undefined,
    root,
    "silent",
  );
  return loaded.config.server.proxy;
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  process.stdout.write(JSON.stringify(await devServerProxy(process.argv[2])));
}
