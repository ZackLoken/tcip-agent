import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

// Relative, not the "@" alias: that alias is declared inside this very config and is not
// available to the config file's own imports at load time.
import { DEV_PROXY } from "./src/api/devProxy.generated";

const BACKEND = process.env.TCIP_WEB_BACKEND ?? "http://127.0.0.1:8765";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      DEV_PROXY.map(({ path, ws }) => [path, { target: BACKEND, changeOrigin: true, ws }]),
    ),
  },
  build: {
    // Same path is restated in the repo root .gitignore and this dir's .prettierignore.
    outDir: "../static",
    emptyOutDir: true,
    // No sourcemaps in the shipped bundle: keeps static/ (and any wheel) lean. Use
    // `npm run dev` (HMR + sourcemaps) for debugging instead.
    sourcemap: false,
  },
});
