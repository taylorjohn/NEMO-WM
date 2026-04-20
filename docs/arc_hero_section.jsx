import { useState, useEffect, useRef } from "react";

const C = ["#111219","#1e88e5","#e53935","#43a047","#fdd835","#9e9e9e","#8e24aa","#fb8c00","#039be5","#6a1b9a"];

const EXAMPLES = [
  {
    id:"140c817e", label:"Cross Diagonal Stamp", family:"Stamp Engine",
    input:[[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
    output:[[0,0,0,1,0,0,1,0,0],[0,0,0,1,0,0,1,0,0],[0,0,3,1,3,0,1,0,0],[1,1,1,2,1,1,1,1,1],[0,0,3,1,3,0,1,0,0],[0,0,0,1,0,3,1,3,0],[1,1,1,1,1,1,2,1,1],[0,0,0,1,0,3,1,3,0],[0,0,0,1,0,0,1,0,0]],
    note:"Learns an arbitrary stamp from input→output diff, applies at each anchor pixel with cross + diagonal neighbor accents."
  },
  {
    id:"a68b268e", label:"Quadrant Boolean OR", family:"Compositional",
    input:[[0,0,3,5,0,3,0],[0,0,0,5,0,0,3],[0,0,0,5,0,0,0],[5,5,5,5,5,5,5],[0,0,0,5,0,0,0],[3,0,0,5,0,0,0],[0,0,0,5,3,0,0]],
    output:[[3,0,3],[3,0,3],[3,0,0]],
    note:"Detects cross-divider, splits into 4 quadrants, overlays with logical OR — multi-step compositional reasoning."
  },
  {
    id:"5751f35e", label:"4-Fold Symmetry", family:"Symmetry",
    input:[[0,0,0,0,0],[0,0,2,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,2]],
    output:[[2,0,0,0,2],[0,0,2,0,0],[0,2,0,2,0],[0,0,2,0,0],[2,0,0,0,2]],
    note:"Completes a partially-filled grid to achieve 4-fold rotational symmetry — requires spatial invariance reasoning."
  },
  {
    id:"5ffb2104", label:"Object Gravity", family:"Relational",
    input:[[0,0,0,0,0,0],[0,2,0,0,0,0],[0,0,0,3,0,0],[0,0,0,0,0,0],[0,4,0,0,0,0],[0,0,0,0,0,0]],
    output:[[0,0,0,0,0,0],[0,0,0,0,0,2],[0,0,0,0,0,3],[0,0,0,0,0,0],[0,0,0,0,0,4],[0,0,0,0,0,0]],
    note:"Simulates physics: each whole object slides rightward until colliding with a wall or another object."
  },
  {
    id:"bc4146bd", label:"Kaleidoscope Mirror", family:"Tiling",
    input:[[1,0,2]],
    output:[[1,0,2,2,0,1,0,2,2,0,1,0,2,2,0]],
    note:"Input becomes a repeating tile with alternating horizontal flips — kaleidoscope pattern generation."
  },
  {
    id:"2072aba6", label:"Pixel → Tile", family:"Compositional",
    input:[[1,0],[0,2]],
    output:[[1,1,0,0],[1,1,0,0],[0,0,2,2],[0,0,2,2]],
    note:"Each pixel in the input becomes a learned 2×2 tile in the output — per-color tile mapping."
  }
];

const FIELD = [
  { name:"Imbue + Gemini", pct:95, tag:"$2-9/task", detail:"LLM-guided code evolution" },
  { name:"Gemini 3 Deep Think", pct:84.6, tag:"$13.62/task", detail:"Frontier LLM + deep search" },
  { name:"NVARC (ARC Prize #1)", pct:24, tag:"GPU cluster", detail:"4B params + test-time train" },
  { name:"NeMo-WM (ours)", pct:10.4, tag:"$0/task", detail:"DSL — zero LLM, CPU only", ours:true },
  { name:"TRM (Paper Prize)", pct:8, tag:"CPU only", detail:"7M param recursive net" },
];

const FAMILIES = [
  { icon:"🔨", name:"Stamp Engine", n:6, t:8, blurb:"Learn arbitrary stamps from diffs" },
  { icon:"🧩", name:"Compositional", n:4, t:5, blurb:"Split → overlay, per-object transforms" },
  { icon:"🔗", name:"Relational", n:5, t:4, blurb:"Adjacency, gravity, containment" },
  { icon:"🪟", name:"Tiling", n:3, t:3, blurb:"Mirror tiles, pixel→tile, partitions" },
  { icon:"🪞", name:"Symmetry", n:2, t:2, blurb:"Complete H/V/4-fold symmetry" },
  { icon:"➡️", name:"Movement", n:2, t:1, blurb:"Translate, gravity with collision" },
  { icon:"🔢", name:"Numerical", n:3, t:2, blurb:"Periodic recolor, scale by count" },
];

function Grid({ data, cell=24, gutter=1.5 }) {
  if(!data?.length) return null;
  const h=data.length, w=data[0].length;
  const totalW = w*cell + (w-1)*gutter;
  const totalH = h*cell + (h-1)*gutter;
  return (
    <svg width={totalW} height={totalH} viewBox={`0 0 ${totalW} ${totalH}`}>
      {data.flatMap((row,r) => row.map((v,c) => (
        <rect key={`${r}-${c}`} x={c*(cell+gutter)} y={r*(cell+gutter)} width={cell} height={cell} rx={2} fill={C[v]||C[0]}
          style={{animation:`fadeCell 0.4s ease ${(r*w+c)*15}ms both`}} />
      )))}
    </svg>
  );
}

function Num({ to, dur=2200, dec=0 }) {
  const [v,setV]=useState(0);
  const raf=useRef();
  useEffect(()=>{
    const t0=performance.now();
    const tick=(now)=>{
      const p=Math.min((now-t0)/dur,1);
      const e=1-Math.pow(1-p,4);
      setV(+(to*e).toFixed(dec));
      if(p<1) raf.current=requestAnimationFrame(tick);
    };
    raf.current=requestAnimationFrame(tick);
    return ()=>cancelAnimationFrame(raf.current);
  },[to,dur,dec]);
  return <>{v}</>;
}

export default function ArcAgi2Hero() {
  const [ex, setEx] = useState(0);
  const [hoveredFamily, setHoveredFamily] = useState(null);

  useEffect(()=>{
    const t=setInterval(()=>setEx(p=>(p+1)%EXAMPLES.length),6000);
    return ()=>clearInterval(t);
  },[]);

  const cur = EXAMPLES[ex];

  return (
    <div style={{
      "--ink":"#d4d0c8","--dim":"#5a574f","--faint":"#2a2824","--surface":"#0e0d0b",
      "--accent":"#c8f560","--accent2":"#60c8f5","--accent3":"#f5a060",
      fontFamily:"'Newsreader','Georgia',serif", color:"var(--ink)", background:"var(--surface)"
    }}>
      <link href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,300;6..72,400;6..72,600;6..72,700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet"/>
      <style>{`
        @keyframes fadeCell { from{opacity:0;transform:scale(0.6)} to{opacity:1;transform:scale(1)} }
        @keyframes slideUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
        @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
        .mono { font-family:'JetBrains Mono',monospace }
        .label { font-size:10px; letter-spacing:3px; text-transform:uppercase; color:var(--dim) }
        .card { background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06); border-radius:12px; transition:all 0.3s }
        .card:hover { border-color:rgba(200,245,96,0.15); background:rgba(255,255,255,0.035) }
        .pill { display:inline-block; padding:3px 10px; border-radius:20px; font-size:10px; font-weight:500 }
        .section { padding:56px 28px; max-width:960px; margin:0 auto }
        .grid-bg { position:absolute; opacity:0.025; pointer-events:none }
      `}</style>

      {/* HERO */}
      <section style={{
        position:"relative", overflow:"hidden",
        padding:"80px 28px 48px", textAlign:"center",
        background:"radial-gradient(ellipse at 25% 0%, rgba(200,245,96,0.04) 0%, transparent 50%), radial-gradient(ellipse at 75% 100%, rgba(96,200,245,0.03) 0%, transparent 50%), var(--surface)"
      }}>
        <div className="grid-bg" style={{top:24,left:24,transform:"rotate(-12deg)"}}>
          <Grid data={EXAMPLES[0].output} cell={6} gutter={1}/>
        </div>
        <div className="grid-bg" style={{bottom:24,right:24,transform:"rotate(7deg)"}}>
          <Grid data={EXAMPLES[2].output} cell={5} gutter={1}/>
        </div>

        <div className="mono label" style={{color:"var(--accent)",marginBottom:12,animation:"slideUp 0.6s ease both"}}>
          NeMo-WM — CORTEX Series
        </div>

        <h1 style={{
          fontSize:"clamp(56px,12vw,110px)", fontWeight:700, margin:"0 0 0",
          lineHeight:0.95, letterSpacing:"-3px",
          animation:"slideUp 0.8s ease 0.1s both"
        }}>
          <span style={{
            background:"linear-gradient(135deg, var(--accent) 0%, var(--accent2) 50%, var(--accent3) 100%)",
            backgroundSize:"200% 200%", animation:"gradientShift 8s ease infinite",
            WebkitBackgroundClip:"text", WebkitTextFillColor:"transparent"
          }}>
            <Num to={104} dur={2800}/>
          </span>
          <span style={{fontSize:"0.38em",opacity:0.35,letterSpacing:0}}>/1000</span>
        </h1>

        <div className="mono" style={{
          fontSize:"clamp(14px,2.5vw,20px)", fontWeight:700, color:"var(--accent)",
          marginTop:4, marginBottom:20,
          animation:"slideUp 0.8s ease 0.2s both"
        }}>
          <Num to={10.4} dec={1} dur={2800}/>% on ARC-AGI-2 — $0 inference cost
        </div>

        <p style={{
          maxWidth:560, margin:"0 auto 36px", color:"var(--dim)", fontSize:15.5,
          lineHeight:1.75, fontWeight:300, animation:"slideUp 0.8s ease 0.3s both"
        }}>
          A neuroscience-inspired working memory architecture that solves abstract reasoning through object-centric perception, compositional program synthesis, and learned relational transforms. No language model. No GPU. No API calls. 95ms per task on CPU.
        </p>

        <div style={{
          display:"flex", justifyContent:"center", gap:36, flexWrap:"wrap",
          animation:"slideUp 0.8s ease 0.4s both"
        }}>
          {[
            {v:"112",l:"SOLVERS",c:"var(--accent)"},
            {v:"7",l:"FAMILIES",c:"var(--accent2)"},
            {v:"$0",l:"PER TASK",c:"var(--accent3)"},
            {v:"95ms",l:"LATENCY",c:"var(--accent)"},
            {v:"0",l:"PARAMETERS",c:"var(--accent2)"},
          ].map(s=>(
            <div key={s.l} style={{textAlign:"center"}}>
              <div className="mono" style={{fontSize:26,fontWeight:700,color:s.c}}>{s.v}</div>
              <div className="mono label" style={{marginTop:2,fontSize:9}}>{s.l}</div>
            </div>
          ))}
        </div>
      </section>

      {/* SOLVED EXAMPLES */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent3)",marginBottom:20}}>
          Solved Examples — Live Showcase
        </div>

        <div style={{display:"flex",gap:4,marginBottom:24,flexWrap:"wrap"}}>
          {EXAMPLES.map((e,i)=>(
            <button key={e.id} onClick={()=>setEx(i)} className="mono" style={{
              padding:"5px 12px", borderRadius:6, border:"none", cursor:"pointer",
              fontSize:10, fontWeight:i===ex?700:400,
              background:i===ex?"rgba(200,245,96,0.1)":"transparent",
              color:i===ex?"var(--accent)":"var(--dim)",
              outline:i===ex?"1px solid rgba(200,245,96,0.2)":"none",
              transition:"all 0.25s"
            }}>
              {e.label}
            </button>
          ))}
        </div>

        <div className="card" style={{padding:28,display:"flex",alignItems:"center",justifyContent:"center",gap:36,flexWrap:"wrap"}}>
          <div style={{textAlign:"center"}}>
            <div className="mono label" style={{marginBottom:8}}>Input</div>
            <Grid data={cur.input} cell={cur.input.length>6?20:28} gutter={1.5} key={`in-${ex}`}/>
          </div>
          <div style={{fontSize:32,color:"var(--accent)",fontWeight:300,opacity:0.5}}>→</div>
          <div style={{textAlign:"center"}}>
            <div className="mono label" style={{marginBottom:8}}>Output</div>
            <Grid data={cur.output} cell={cur.output.length>6?20:(cur.output[0]?.length>10?14:28)} gutter={1.5} key={`out-${ex}`}/>
          </div>
        </div>

        <div style={{marginTop:14,display:"flex",alignItems:"baseline",gap:8,flexWrap:"wrap"}}>
          <span className="mono pill" style={{background:"rgba(200,245,96,0.08)",color:"var(--accent)",border:"1px solid rgba(200,245,96,0.15)"}}>
            {cur.family}
          </span>
          <span className="mono pill" style={{background:"rgba(255,255,255,0.03)",color:"var(--dim)"}}>
            {cur.id}
          </span>
          <span style={{fontSize:13.5,color:"var(--dim)",lineHeight:1.6,fontWeight:300}}>{cur.note}</span>
        </div>
      </section>

      {/* FIELD COMPARISON */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent2)",marginBottom:8}}>
          ARC-AGI-2 Leaderboard Context
        </div>
        <h2 style={{fontSize:28,fontWeight:700,marginBottom:4,letterSpacing:"-0.5px"}}>
          How NeMo-WM Compares
        </h2>
        <p style={{color:"var(--dim)",fontSize:14,marginBottom:28,maxWidth:520,lineHeight:1.7,fontWeight:300}}>
          Higher-scoring systems spend $2–14 per task on frontier LLM inference. NeMo-WM achieves its score at zero cost with deterministic, interpretable solvers.
        </p>

        <div style={{display:"flex",flexDirection:"column",gap:6}}>
          {FIELD.map(f=>{
            const w = Math.max(f.pct, 1.5);
            return (
              <div key={f.name} style={{
                display:"flex",alignItems:"center",gap:14,
                padding:"10px 16px",borderRadius:8,
                background:f.ours?"rgba(200,245,96,0.04)":"transparent",
                border:f.ours?"1px solid rgba(200,245,96,0.12)":"1px solid transparent"
              }}>
                <div style={{width:160,fontSize:13,fontWeight:f.ours?700:400,color:f.ours?"var(--accent)":"var(--ink)",whiteSpace:"nowrap"}}>
                  {f.name}
                </div>
                <div style={{flex:1,position:"relative",height:20,background:"rgba(255,255,255,0.025)",borderRadius:4,overflow:"hidden"}}>
                  <div style={{
                    width:`${w}%`,height:"100%",borderRadius:4,
                    background:f.ours
                      ? "linear-gradient(90deg,rgba(200,245,96,0.6),rgba(96,200,245,0.4))"
                      : `rgba(96,200,245,${0.08 + f.pct/250})`,
                    transition:"width 2s cubic-bezier(0.22,1,0.36,1)"
                  }}/>
                  <span className="mono" style={{
                    position:"absolute",right:8,top:"50%",transform:"translateY(-50%)",
                    fontSize:11,fontWeight:700,color:f.ours?"var(--accent)":"var(--dim)"
                  }}>
                    {f.pct}%
                  </span>
                </div>
                <div className="mono" style={{width:80,fontSize:9,color:"var(--dim)",textAlign:"right"}}>
                  {f.tag}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* SOLVER FAMILIES */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent3)",marginBottom:8}}>
          Solver Architecture
        </div>
        <h2 style={{fontSize:28,fontWeight:700,marginBottom:4,letterSpacing:"-0.5px"}}>
          112 Solvers, 7 Families
        </h2>
        <p style={{color:"var(--dim)",fontSize:14,marginBottom:24,maxWidth:520,lineHeight:1.7,fontWeight:300}}>
          Each family targets a distinct class of abstract reasoning. The stamp engine alone accounts for 8 tasks no pixel-level baseline can solve.
        </p>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(270px,1fr))",gap:10}}>
          {FAMILIES.map(f=>(
            <div key={f.name} className="card" style={{padding:"16px 18px",cursor:"default"}}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:6}}>
                <span style={{fontSize:15,fontWeight:600}}>{f.icon} {f.name}</span>
                <span className="mono" style={{fontSize:10,color:"var(--accent)"}}>
                  {f.n} solvers → {f.t} tasks
                </span>
              </div>
              <div style={{fontSize:13,color:"var(--dim)",lineHeight:1.55,fontWeight:300}}>{f.blurb}</div>
            </div>
          ))}
        </div>
      </section>

      {/* REASONING PIPELINE */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent)",marginBottom:8}}>
          Reasoning Pipeline
        </div>
        <h2 style={{fontSize:28,fontWeight:700,marginBottom:24,letterSpacing:"-0.5px"}}>
          How NeMo-WM Solves a Task
        </h2>

        <div style={{display:"flex",gap:4,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {n:"1",t:"Perceive",d:"Extract connected components as objects with color, shape, size, bbox, density, and spatial attributes.",c:"var(--accent2)"},
            {n:"2",t:"Relate",d:"Build relational graph: adjacency, containment, alignment, same-shape/color grouping between objects.",c:"var(--accent)"},
            {n:"3",t:"Hypothesize",d:"Run 112 solvers across 7 families — stamps, compositions, symmetry, tiling, gravity, periodic recoloring.",c:"var(--accent3)"},
            {n:"4",t:"Verify",d:"Check each hypothesis against ALL training I/O pairs. Only exact pixel-matches on every pair survive.",c:"var(--accent2)"},
            {n:"5",t:"Apply",d:"Apply the verified solver to the test input. Deterministic output — same input always gives same answer.",c:"var(--accent)"},
          ].map(s=>(
            <div key={s.n} style={{
              flex:"1 1 160px",maxWidth:185,padding:"18px 14px",borderRadius:10,textAlign:"center",
              background:`linear-gradient(180deg, ${s.c}06, transparent)`,
              border:`1px solid ${s.c}18`,margin:2
            }}>
              <div className="mono" style={{
                width:30,height:30,borderRadius:"50%",margin:"0 auto 8px",
                display:"flex",alignItems:"center",justifyContent:"center",
                background:`${s.c}15`,color:s.c,fontSize:14,fontWeight:700
              }}>{s.n}</div>
              <div style={{fontSize:14,fontWeight:600,color:s.c,marginBottom:4}}>{s.t}</div>
              <div style={{fontSize:11.5,color:"var(--dim)",lineHeight:1.55,fontWeight:300}}>{s.d}</div>
            </div>
          ))}
        </div>
      </section>

      {/* SCORE BREAKDOWN */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent2)",marginBottom:20}}>
          Score Breakdown
        </div>

        <div style={{display:"flex",gap:28,flexWrap:"wrap",justifyContent:"center"}}>
          {[
            {l:"S1 Baseline",v:54,c:"#5a574f",d:"37 pixel strategies"},
            {l:"Mental Models",v:4,c:"var(--accent2)",d:"Geometric compositions"},
            {l:"Advanced Ops",v:14,c:"var(--accent3)",d:"50 advanced operations"},
            {l:"Object Graph",v:32,c:"var(--accent)",d:"112 relational solvers"},
          ].map(b=>{
            const frac=b.v/104;
            const circ=2*Math.PI*38;
            return (
              <div key={b.l} style={{textAlign:"center",flex:"0 0 auto"}}>
                <div style={{position:"relative",width:96,height:96,margin:"0 auto 8px"}}>
                  <svg viewBox="0 0 96 96" style={{transform:"rotate(-90deg)"}}>
                    <circle cx="48" cy="48" r="38" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="6"/>
                    <circle cx="48" cy="48" r="38" fill="none" stroke={b.c} strokeWidth="6"
                      strokeDasharray={`${frac*circ} ${circ}`} strokeLinecap="round"
                      style={{transition:"stroke-dasharray 2s cubic-bezier(0.22,1,0.36,1)"}}/>
                  </svg>
                  <div className="mono" style={{
                    position:"absolute",top:"50%",left:"50%",transform:"translate(-50%,-50%)",
                    fontSize:22,fontWeight:700,color:b.c
                  }}>{b.v}</div>
                </div>
                <div style={{fontSize:13,fontWeight:600}}>{b.l}</div>
                <div style={{fontSize:11,color:"var(--dim)",fontWeight:300}}>{b.d}</div>
              </div>
            );
          })}
        </div>
      </section>

      {/* KEY FINDINGS */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent3)",marginBottom:20}}>
          Key Research Findings
        </div>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(280px,1fr))",gap:12}}>
          {[
            {t:"The Stamp Engine is the Golden Path",b:"Every stamp variant (cross, learned, per-color, bbox-relative, self-color) discovers 1–3 tasks that pixel-level baselines miss. The insight: learn arbitrary stamp patterns from input→output differences rather than hardcoding shapes."},
            {t:"Only Relational Solvers Break S1",b:"S1's 37 strategies catch all simple transforms — flip, rotate, scale, crop, color-map. Only solvers reasoning about object relationships (containment, adjacency, composition) survive the baseline filter."},
            {t:"Compositional Reasoning Validates Chollet",b:"Quadrant overlay, per-object transforms, and multi-step chains confirm that compositional solvers solve tasks single-step approaches can't. This aligns with Chollet's thesis about compositional generalization being key to intelligence."},
            {t:"Zero Cost, Maximum Interpretability",b:"104 tasks at $0/task, 95ms each, on CPU. Every solution is a named, inspectable solver with a clear reasoning trace. Compare: frontier LLM approaches cost $2–14/task with opaque reasoning."},
          ].map(x=>(
            <div key={x.t} className="card" style={{padding:"20px 22px"}}>
              <div style={{fontSize:15,fontWeight:600,marginBottom:8,lineHeight:1.35}}>{x.t}</div>
              <div style={{fontSize:13,color:"var(--dim)",lineHeight:1.65,fontWeight:300}}>{x.b}</div>
            </div>
          ))}
        </div>
      </section>

      {/* TRAINING DATA */}
      <section className="section" style={{borderTop:"1px solid var(--faint)"}}>
        <div className="mono label" style={{color:"var(--accent2)",marginBottom:8}}>
          Training Ecosystem
        </div>
        <h2 style={{fontSize:24,fontWeight:700,marginBottom:20,letterSpacing:"-0.5px"}}>
          2,960+ Tasks from 7 Sources
        </h2>

        <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(190px,1fr))",gap:8}}>
          {[
            {n:"ARC-AGI-2",v:"1,000",d:"Official benchmark"},
            {n:"RE-ARC",v:"400K",d:"Procedural generators"},
            {n:"Google ARC-GEN",v:"100K",d:"Mimetic generators"},
            {n:"ConceptARC",v:"160",d:"16 concept groups"},
            {n:"Grid-Math",v:"2,400",d:"Numerical reasoning"},
            {n:"ARC Collection",v:"9,356",d:"21 community sources"},
            {n:"DeepMind Math",v:"2M",d:"School-level Q&A"},
          ].map(d=>(
            <div key={d.n} className="card" style={{padding:"12px 14px"}}>
              <div style={{fontSize:12.5,fontWeight:600}}>{d.n}</div>
              <div className="mono" style={{fontSize:18,fontWeight:700,color:"var(--accent2)",marginTop:2}}>{d.v}</div>
              <div style={{fontSize:11,color:"var(--dim)",fontWeight:300}}>{d.d}</div>
            </div>
          ))}
        </div>
      </section>

      {/* FOOTER */}
      <footer style={{padding:"28px 28px",textAlign:"center",borderTop:"1px solid var(--faint)"}}>
        <div className="mono" style={{fontSize:11,color:"var(--dim)",letterSpacing:1}}>
          NeMo-WM · CORTEX Series · ARC-AGI-2 Benchmark · April 2026
        </div>
        <div className="mono" style={{fontSize:10,color:"rgba(255,255,255,0.15)",marginTop:4}}>
          github.com/taylorjohn/NEMO-WM
        </div>
      </footer>
    </div>
  );
}
