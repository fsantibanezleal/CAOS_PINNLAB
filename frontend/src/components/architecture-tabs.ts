// Architecture-modal tab registry (the ⓘ "how it was built" panel: Veta/Circuita pattern). Each tab pairs ONE
// hand-authored themed SVG (every colour a CSS variable, fetched from svg/tech/ and inlined so the active theme
// paints it) with a compact bilingual explanation; the methods tab also renders text blocks BELOW the diagram.
// This panel is what demonstrates the app is REAL: the actual modules, stages and data flow, not a demo facade.

export interface TabBlockDef {
  titleKey: string;
  textKey: string;
}

export interface TabDef {
  id: string;
  labelKey: string;
  textKey: string;
  svg: string; // under svg/tech/
  blocks?: TabBlockDef[];
}

export const TABS: TabDef[] = [
  { id: "overview", labelKey: "arch.tab.overview", textKey: "arch.overview.text", svg: "system-overview.svg" },
  { id: "web", labelKey: "arch.tab.web", textKey: "arch.web.text", svg: "web-architecture.svg" },
  { id: "pipeline", labelKey: "arch.tab.pipeline", textKey: "arch.pipeline.text", svg: "offline-pipeline.svg" },
  { id: "bridge", labelKey: "arch.tab.bridge", textKey: "arch.bridge.text", svg: "train-onnx-web.svg" },
  { id: "gate", labelKey: "arch.tab.gate", textKey: "arch.gate.text", svg: "lane-gate.svg" },
  {
    id: "methods",
    labelKey: "arch.tab.methods",
    textKey: "arch.methods.text",
    svg: "method-matrix.svg",
    blocks: [
      { titleKey: "arch.methods.constraints.title", textKey: "arch.methods.constraints.text" },
      { titleKey: "arch.methods.adaptive.title", textKey: "arch.methods.adaptive.text" },
      { titleKey: "arch.methods.inverse.title", textKey: "arch.methods.inverse.text" },
      { titleKey: "arch.methods.uq.title", textKey: "arch.methods.uq.text" },
      { titleKey: "arch.methods.real.title", textKey: "arch.methods.real.text" },
      { titleKey: "arch.methods.honesty.title", textKey: "arch.methods.honesty.text" },
    ],
  },
];
