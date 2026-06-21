// Overlay the committed pipeline artifacts into frontend/public/data so Vite serves them — the web's replay lane
// (committed field traces + manifests) and live lane (exported ONNX). This enforces CONTRACT 2: the SPA loads ONLY
// these baked artifacts, never recomputes. Run automatically by the predev/prebuild npm hooks.
import { cpSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const out = resolve(here, "public", "data");

if (existsSync(out)) rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// data/derived/ holds manifests/ (index + per-case) and <case>/field.json replay artifacts.
const derived = resolve(repo, "data", "derived");
if (!existsSync(derived)) {
  console.error("copy-data: data/derived not found — run the precompute pipeline first.");
  process.exit(1);
}
cpSync(derived, resolve(out, "derived"), { recursive: true });

// models/ holds the exported <case>.onnx for the live lane.
const models = resolve(repo, "models");
if (existsSync(models)) cpSync(models, resolve(out, "models"), { recursive: true });

console.log("copy-data: overlaid data/derived + models into frontend/public/data");
