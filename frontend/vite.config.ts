import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Relative base so the bundle works both on the project-site subpath (fsantibanezleal.github.io/CAOS_PINNLAB/)
// and on a custom-domain apex (pinnlab.fasl-work.com) with no rebuild.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", target: "es2022", chunkSizeWarningLimit: 1500 },
  // onnxruntime-web ships large .wasm/.mjs assets; keep them out of the optimizer.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
});
