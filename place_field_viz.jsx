import { useState } from "react";

const C = {
  bg:"#050810", card:"#111827", border:"#1e2d45",
  accent1:"#00d4ff", accent2:"#7c3aed", accent3:"#10b981",
  warn:"#f59e0b", danger:"#ef4444", text:"#e2e8f0", muted:"#64748b",
};

// Real data from eval_place_cell_receptive_fields.py
const PARTICLES = [
  {id:0,  peak_n:-8.1,  peak_e:141.7, fw:26.6, tuning:1.148, type:"diffuse"},
  {id:1,  peak_n:-11.5, peak_e:147.6, fw:17.1, tuning:1.208, type:"regional"},
  {id:2,  peak_n:3.4,   peak_e:152.0, fw:9.8,  tuning:1.224, type:"regional"},
  {id:3,  peak_n:-5.2,  peak_e:155.2, fw:21.5, tuning:1.417, type:"regional"},
  {id:4,  peak_n:-6.8,  peak_e:154.4, fw:14.6, tuning:1.226, type:"regional"},
  {id:5,  peak_n:-6.1,  peak_e:144.8, fw:25.2, tuning:1.321, type:"regional"},
  {id:6,  peak_n:-17.4, peak_e:145.8, fw:19.3, tuning:1.308, type:"regional"},
  {id:7,  peak_n:-6.8,  peak_e:109.4, fw:69.1, tuning:1.219, type:"diffuse"},
  {id:8,  peak_n:-8.3,  peak_e:117.5, fw:57.2, tuning:1.358, type:"diffuse"},
  {id:9,  peak_n:-13.6, peak_e:146.2, fw:12.8, tuning:1.106, type:"diffuse"},
  {id:10, peak_n:-15.9, peak_e:137.8, fw:22.1, tuning:1.136, type:"diffuse"},
  {id:11, peak_n:-11.7, peak_e:140.9, fw:37.9, tuning:1.262, type:"regional"},
  {id:12, peak_n:2.7,   peak_e:150.1, fw:25.2, tuning:1.199, type:"diffuse"},
  {id:13, peak_n:-3.0,  peak_e:105.2, fw:66.0, tuning:1.245, type:"diffuse"},
  {id:14, peak_n:1.5,   peak_e:148.6, fw:13.6, tuning:7.162, type:"place-like"},
  {id:15, peak_n:-5.7,  peak_e:158.0, fw:19.2, tuning:1.283, type:"regional"},
];

const TYPE_COLOR = { "place-like": C.accent3, regional: C.accent1, diffuse: C.muted };
const TYPE_ICON  = { "place-like": "◉", regional: "◎", diffuse: "○" };

// Space bounds from output
const BOUNDS = { nMin:-39, nMax:15, eMin:-2, eMax:176 };

function mapToSvg(n, e, W, H, pad=24) {
  const x = pad + ((e - BOUNDS.eMin) / (BOUNDS.eMax - BOUNDS.eMin)) * (W - 2*pad);
  const y = H - pad - ((n - BOUNDS.nMin) / (BOUNDS.nMax - BOUNDS.nMin)) * (H - 2*pad);
  return [x, y];
}

// Gaussian blob radius scaled by field width
function fieldRadius(fw, W) {
  return (fw / (BOUNDS.eMax - BOUNDS.eMin)) * (W - 48) * 0.9;
}

export default function PlaceFieldViz() {
  const [sel, setSel] = useState(14); // default: p14 (the place cell)
  const [showAll, setShowAll] = useState(false);
  const W = 460, H = 200, pad = 24;

  const selP = PARTICLES[sel];
  const color = TYPE_COLOR[selP.type];
  const [cx, cy] = mapToSvg(selP.peak_n, selP.peak_e, W, H, pad);
  const r = fieldRadius(selP.fw, W);

  const displayParticles = showAll ? PARTICLES : [selP];

  return (
    <div style={{ background: C.bg, color: C.text, fontFamily: "'DM Sans',system-ui,sans-serif",
                  padding: 24, minHeight: "100vh" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');*{box-sizing:border-box;margin:0;padding:0}`}</style>

      {/* Header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 600, letterSpacing: "-0.02em" }}>
          NeMo-WM Place Cell Analysis
        </h1>
        <p style={{ fontSize: 12, color: C.muted, marginTop: 3 }}>
          K=16 particle receptive fields · RECON Berkeley campus · 1,933 frames · 50 trajectories
        </p>
      </div>

      {/* Summary stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 10, marginBottom: 20 }}>
        {[
          { label: "Place-like cells", value: "1/16", sub: "p14 · tuning 7.16×", color: C.accent3 },
          { label: "Regional cells",   value: "8/16", sub: "12–37m field width", color: C.accent1 },
          { label: "Diffuse cells",    value: "7/16", sub: "57–69m broad tuning", color: C.muted },
          { label: "Space covered",    value: "54×178m", sub: "Berkeley campus corridor", color: C.warn },
        ].map(s => (
          <div key={s.label} style={{ background: C.card, border: `1px solid ${C.border}`,
                                      borderRadius: 10, padding: "12px 16px",
                                      borderLeft: `3px solid ${s.color}` }}>
            <div style={{ fontSize: 9, color: C.muted, letterSpacing: "0.1em",
                          textTransform: "uppercase", marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontSize: 20, fontFamily: "'DM Mono',monospace",
                          color: s.color, fontWeight: 600 }}>{s.value}</div>
            <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>{s.sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 16 }}>

        {/* Main map */}
        <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 12, padding: 16 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 600, color: C.text }}>
                Spatial Receptive Field — p{String(sel).padStart(2,"0")}
              </div>
              <div style={{ fontSize: 10, color: C.muted, marginTop: 2 }}>
                {TYPE_ICON[selP.type]} {selP.type} · tuning {selP.tuning.toFixed(3)}× ·
                peak (N={selP.peak_n.toFixed(1)}m, E={selP.peak_e.toFixed(1)}m) ·
                field width {selP.fw.toFixed(1)}m
              </div>
            </div>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: C.muted,
                             cursor: "pointer" }}>
              <input type="checkbox" checked={showAll}
                     onChange={e => setShowAll(e.target.checked)}
                     style={{ accentColor: C.accent1 }} />
              Show all 16
            </label>
          </div>

          <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: "block" }}>
            <defs>
              {PARTICLES.map(p => {
                const [pcx, pcy] = mapToSvg(p.peak_n, p.peak_e, W, H, pad);
                const pr = fieldRadius(p.fw, W);
                const pc = TYPE_COLOR[p.type];
                return (
                  <radialGradient key={p.id} id={`rg${p.id}`} cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor={pc} stopOpacity={p.id===14?0.55:0.25}/>
                    <stop offset="60%" stopColor={pc} stopOpacity={p.id===14?0.20:0.08}/>
                    <stop offset="100%" stopColor={pc} stopOpacity="0"/>
                  </radialGradient>
                );
              })}
            </defs>

            {/* Background grid */}
            <rect x={pad} y={pad} width={W-2*pad} height={H-2*pad}
                  fill="#0d1829" rx={4}/>
            {[-30,-20,-10,0,10].map(n => {
              const [,gy] = mapToSvg(n, BOUNDS.eMin, W, H, pad);
              return <line key={n} x1={pad} y1={gy} x2={W-pad} y2={gy}
                           stroke="#1e2d45" strokeWidth={0.5}/>;
            })}
            {[0,50,100,150].map(e => {
              const [gx] = mapToSvg(0, e, W, H, pad);
              return <line key={e} x1={gx} y1={pad} x2={gx} y2={H-pad}
                           stroke="#1e2d45" strokeWidth={0.5}/>;
            })}

            {/* Axis labels */}
            <text x={pad+2} y={H-pad+12} fill={C.muted} fontSize={8}>E=0m</text>
            <text x={W-pad-18} y={H-pad+12} fill={C.muted} fontSize={8}>E=176m</text>
            {[mapToSvg(-30,BOUNDS.eMin,W,H,pad), mapToSvg(10,BOUNDS.eMin,W,H,pad)].map(([,gy],i) => (
              <text key={i} x={2} y={gy+3} fill={C.muted} fontSize={8}>{i===0?"N=-30":"N=10"}</text>
            ))}

            {/* Field blobs */}
            {(showAll ? PARTICLES : [selP]).map(p => {
              const [pcx, pcy] = mapToSvg(p.peak_n, p.peak_e, W, H, pad);
              const pr = fieldRadius(p.fw, W);
              return (
                <ellipse key={p.id}
                  cx={pcx} cy={pcy}
                  rx={pr} ry={pr * 0.7}
                  fill={`url(#rg${p.id})`}
                  style={{ cursor: "pointer" }}
                  onClick={() => setSel(p.id)}
                />
              );
            })}

            {/* Peak markers */}
            {(showAll ? PARTICLES : [selP]).map(p => {
              const [pcx, pcy] = mapToSvg(p.peak_n, p.peak_e, W, H, pad);
              const pc = TYPE_COLOR[p.type];
              const isPlace = p.type === "place-like";
              return (
                <g key={p.id} style={{ cursor: "pointer" }} onClick={() => setSel(p.id)}>
                  {isPlace && (
                    <circle cx={pcx} cy={pcy} r={10} fill="none"
                            stroke={pc} strokeWidth={1} strokeDasharray="3,3" opacity={0.6}/>
                  )}
                  <circle cx={pcx} cy={pcy} r={p.id===sel?5:3}
                          fill={pc} stroke={C.bg} strokeWidth={1.5}
                          opacity={showAll&&p.id!==sel?0.5:1}/>
                  {(p.id===sel||!showAll) && (
                    <text x={pcx+7} y={pcy-4} fill={pc} fontSize={9} fontFamily="monospace">
                      p{String(p.id).padStart(2,"0")}
                    </text>
                  )}
                  {showAll && p.type==="place-like" && (
                    <text x={pcx+7} y={pcy+4} fill={pc} fontSize={8}>★</text>
                  )}
                </g>
              );
            })}
          </svg>

          <div style={{ fontSize: 10, color: C.muted, marginTop: 8, display: "flex", gap: 16 }}>
            <span><span style={{ color: C.accent3 }}>◉ place-like</span> — sharp localised field</span>
            <span><span style={{ color: C.accent1 }}>◎ regional</span> — moderate selectivity</span>
            <span><span style={{ color: C.muted }}>○ diffuse</span> — broad/path-general</span>
            <span style={{ marginLeft: "auto" }}>Click marker to select particle</span>
          </div>
        </div>

        {/* Right panel: particle list + p14 callout */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

          {/* p14 callout */}
          <div style={{ background: `${C.accent3}12`, border: `1px solid ${C.accent3}44`,
                        borderRadius: 10, padding: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 600, color: C.accent3, marginBottom: 6 }}>
              ◉ p14 — Genuine Place Cell
            </div>
            <div style={{ fontSize: 10, color: C.accent3, opacity: 0.8, lineHeight: 1.6 }}>
              Tuning index 7.16× — 5× higher than next best (p03: 1.42×).
              Field width 13.6m. Peak at (N=1.5m, E=148.6m).
            </div>
            <div style={{ fontSize: 10, color: C.muted, marginTop: 8, lineHeight: 1.5 }}>
              Biological parallel: O'Keefe & Dostrovsky (1971) first identified
              hippocampal place cells with localised spatial receptive fields.
              p14 shows the same computational signature in a learned world model.
            </div>
          </div>

          {/* Particle list */}
          <div style={{ background: C.card, border: `1px solid ${C.border}`,
                        borderRadius: 10, padding: 14, flex: 1, overflow: "auto" }}>
            <div style={{ fontSize: 10, color: C.muted, letterSpacing: "0.08em",
                          textTransform: "uppercase", marginBottom: 10 }}>All 16 particles</div>
            {PARTICLES.map(p => {
              const pc = TYPE_COLOR[p.type];
              const barW = Math.min(100, (p.tuning / 7.5) * 100);
              return (
                <div key={p.id} onClick={() => setSel(p.id)} style={{
                  display: "flex", alignItems: "center", gap: 8,
                  padding: "5px 4px", borderRadius: 4, cursor: "pointer",
                  background: sel===p.id ? `${pc}15` : "transparent",
                  border: sel===p.id ? `1px solid ${pc}44` : "1px solid transparent",
                  marginBottom: 2, transition: "all 0.15s",
                }}>
                  <span style={{ fontSize: 10, color: pc, width: 28,
                                 fontFamily: "monospace", fontWeight: 600 }}>
                    p{String(p.id).padStart(2,"0")}
                  </span>
                  <div style={{ flex: 1, background: "#0d1829", borderRadius: 2, height: 5 }}>
                    <div style={{ width: `${barW}%`, height: "100%", background: pc,
                                  borderRadius: 2, transition: "width 0.6s",
                                  boxShadow: p.type==="place-like"?`0 0 6px ${pc}`:"none" }}/>
                  </div>
                  <span style={{ fontSize: 9, fontFamily: "monospace",
                                 color: pc, width: 36, textAlign: "right" }}>
                    {p.tuning.toFixed(2)}×
                  </span>
                  <span style={{ fontSize: 9, color: C.muted, width: 18 }}>
                    {TYPE_ICON[p.type]}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Biological interpretation */}
      <div style={{ marginTop: 16, background: C.card, border: `1px solid ${C.border}`,
                    borderRadius: 12, padding: 16 }}>
        <div style={{ fontSize: 11, fontWeight: 600, color: C.text, marginBottom: 8,
                      letterSpacing: "0.05em", textTransform: "uppercase" }}>
          Biological Interpretation
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 12 }}>
          {[
            { title: "Population coding", color: C.accent3,
              text: "1 place-like + 8 regional particles form a distributed code over the Berkeley campus corridor. The ensemble collectively represents navigational position — consistent with hippocampal population coding (O'Keefe & Nadel, 1978)." },
            { title: "Spatial tiling", color: C.accent1,
              text: "Regional particles cluster around E=140–160m, tiling the most-traversed zone. p07, p08, p13 (broad, E=105–120m) represent the less-visited western section — sparser tiling where fewer training frames were collected." },
            { title: "Emergence without supervision", color: C.warn,
              text: "Place field structure emerged from VLM-grounded contrastive training (no GPS supervision). The particle encoder was never told where the robot was — spatial selectivity is a byproduct of learning temporal proximity." },
          ].map(item => (
            <div key={item.title} style={{ padding: 12, background: `${item.color}08`,
                                           borderRadius: 8, border: `1px solid ${item.color}22` }}>
              <div style={{ fontSize: 11, color: item.color, fontWeight: 600, marginBottom: 6 }}>
                {item.title}
              </div>
              <div style={{ fontSize: 10, color: C.muted, lineHeight: 1.6 }}>{item.text}</div>
            </div>
          ))}
        </div>
      </div>

      <div style={{ marginTop: 14, fontSize: 10, color: C.muted, fontFamily: "monospace",
                    display: "flex", justifyContent: "space-between" }}>
        <span>1,933 frames · 50 trajectories · 37.915°N Berkeley CA</span>
        <span>NeMo-WM neuroscience ablation series · eval_place_cell_receptive_fields.py</span>
      </div>
    </div>
  );
}
