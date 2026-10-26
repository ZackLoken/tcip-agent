import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Relative: the config's own imports resolve before it applies tsconfig.json's paths.
import { BACKEND_PORT, DEV_PROXY, LOOPBACK_HOST } from "./src/api/devProxy.generated";

const BACKEND = process.env.TCIP_WEB_BACKEND ?? `http://${LOOPBACK_HOST}:${BACKEND_PORT}`;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { tsconfigPaths: true },
  server: {
    host: LOOPBACK_HOST,
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
