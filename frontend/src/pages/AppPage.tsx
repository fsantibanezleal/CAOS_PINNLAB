import katex from "katex";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";

import { FieldHeatmap } from "../components/FieldHeatmap";
import type { CaseIndex, CaseManifest, FieldTrace } from "../lib/contract";
import { loadIndex, loadManifest, loadTrace } from "../lib/data";

function honestyClass(flag: string): string {
  return flag.startsWith("validated") ? "validated" : "synthetic";
}

export function AppPage() {
  const { t } = useTranslation();
  const [index, setIndex] = useState<CaseIndex | null>(null);
  const [caseId, setCaseId] = useState<string>("");
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [trace, setTrace] = useState<FieldTrace | null>(null);
  const [tSlice, setTSlice] = useState(0);
  const [outIdx, setOutIdx] = useState(0);

  useEffect(() => {
    loadIndex().then((ix) => {
      setIndex(ix);
      setCaseId(ix.cases[0]?.case_id ?? "");
    });
  }, []);

  useEffect(() => {
    if (!caseId) return;
    setManifest(null);
    setTrace(null);
    setTSlice(0);
    setOutIdx(0);
    loadManifest(caseId).then((m) => {
      setManifest(m);
      loadTrace(m.artifact.path).then(setTrace);
    });
  }, [caseId]);

  // primary output field; if the case is 3D (x,y,t), slice by the time index
  const field2d = useMemo<number[][] | null>(() => {
    if (!trace || !manifest) return null;
    const name = manifest.outputs[outIdx] ?? manifest.outputs[0];
    const f = trace.fields[name];
    if (!Array.isArray(f) || !Array.isArray(f[0])) return null;
    if (Array.isArray((f[0] as unknown[])[0])) {
      const arr = f as unknown as number[][][];
      const nt = arr[0][0].length;
      const ti = Math.min(tSlice, nt - 1);
      return arr.map((row) => row.map((col) => col[ti]));
    }
    return f as number[][];
  }, [trace, manifest, tSlice, outIdx]);

  const is3d = manifest?.inputs.length === 3;
  const eqHtml = useMemo(() => {
    if (!manifest) return "";
    try {
      return katex.renderToString(manifest.governing_equations, { throwOnError: false, displayMode: true });
    } catch {
      return manifest.governing_equations;
    }
  }, [manifest]);

  const l2 = manifest?.metrics.l2_relative;

  return (
    <div className="layout">
      <aside className="panel">
        <h3>{t("app.selectCase")}</h3>
        <div className="chips">
          {index?.cases.map((c) => (
            <button
              key={c.case_id}
              className={`chip${c.case_id === caseId ? " on" : ""}`}
              onClick={() => setCaseId(c.case_id)}
              title={c.title}
            >
              {c.case_id}
            </button>
          ))}
        </div>
        {manifest && (
          <div className="kv" style={{ marginTop: 14 }}>
            <span className="k">{t("app.method")}</span>
            <span className="v mono">{manifest.method}</span>
            <span className="k">{t("app.engine")}</span>
            <span className="v mono">{manifest.engine.framework}</span>
            <span className="k">{t("app.lane")}</span>
            <span className="v">
              <span className={`tag ${manifest.lane}`}>{manifest.lane}</span>
            </span>
            <span className="k">{t("app.dataLabel")}</span>
            <span className="v">
              <span className={`tag ${honestyClass(manifest.real_or_synthetic)}`}>
                {t(`honesty.${manifest.real_or_synthetic}`, { defaultValue: manifest.real_or_synthetic })}
              </span>
            </span>
            {l2 !== undefined && (
              <>
                <span className="k">{t("app.relL2")}</span>
                <span className="v mono">{Number(l2).toExponential(2)}</span>
              </>
            )}
            <span className="k">{t("app.onnxParity")}</span>
            <span className="v mono">{Number(manifest.onnx.parity_max_abs).toExponential(2)}</span>
          </div>
        )}
      </aside>

      <section className="panel">
        {!manifest && <p className="muted">{t("app.loading")}</p>}
        {manifest && (
          <>
            <h2 style={{ marginTop: 0 }}>{manifest.title}</h2>
            <div className="eq" dangerouslySetInnerHTML={{ __html: eqHtml }} />
            {manifest.outputs.length > 1 && (
              <div className="row" style={{ marginTop: 12, alignItems: "center" }}>
                <span className="muted mono" style={{ fontSize: 13 }}>{t("app.field")}</span>
                <div className="chips" style={{ margin: 0 }}>
                  {manifest.outputs.map((o, i) => (
                    <button
                      key={o}
                      className={`chip${i === outIdx ? " on" : ""}`}
                      onClick={() => setOutIdx(i)}
                      title={o}
                    >
                      {o}
                    </button>
                  ))}
                </div>
              </div>
            )}
            <div className="field-wrap" style={{ marginTop: 14 }}>
              {field2d ? <FieldHeatmap field={field2d} /> : <p className="muted">{t("app.loading")}</p>}
              {is3d && trace && (
                <div className="row">
                  <span className="muted mono">t</span>
                  <input
                    type="range"
                    min={0}
                    max={(trace.axes[trace.dims[2]]?.length ?? 1) - 1}
                    value={tSlice}
                    onChange={(e) => setTSlice(Number(e.target.value))}
                  />
                </div>
              )}
              <p className="muted" style={{ fontSize: 13 }}>
                {manifest.expected_band}
              </p>
            </div>
          </>
        )}
      </section>
    </div>
  );
}
