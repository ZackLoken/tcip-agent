import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

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
    proxy: {
      "/api": { target: BACKEND, changeOrigin: true },
      "/ws": { target: BACKEND, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
    // No sourcemaps in the shipped bundle: keeps static/ (and any wheel) lean. Use
    // `npm run dev` (HMR + sourcemaps) for debugging instead.
    sourcemap: false,
  },
});
