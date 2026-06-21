import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Deploy target: GitHub Pages at the apex of pinnlab.fasl-work.com -> base "/".
export default defineConfig({
  base: "/",
  plugins: [react()],
  build: { outDir: "dist", target: "es2022", chunkSizeWarningLimit: 1500 },
  // onnxruntime-web ships large .wasm/.mjs assets; keep them out of the optimizer.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
