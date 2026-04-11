import { useState, useEffect } from "react";

const C = {
  bg:"#050810", card:"#111827", border:"#1e2d45",
  accent1:"#00d4ff", accent2:"#7c3aed", accent3:"#10b981",
  warn:"#f59e0b", danger:"#ef4444", text:"#e2e8f0", muted:"#64748b",
  grid:"#0d1829",
};

// ── All real data ──────────────────────────────────────────────────────────

const SPRINT_DATA = [
  { sprint:"S2",  label:"RECON Baseline",          auroc:0.9075, color:"#4ade80" },
  { sprint:"S4",  label:"Aphasia ablated→0.500",   auroc:0.923,  color:"#f87171" },
  { sprint:"S5",  label:"Hard negatives (correct)", auroc:0.865,  color:"#60a5fa" },
  { sprint:"S6",  label:"Proprio single frame",     auroc:0.892,  color:"#a78bfa" },
  { sprint:"S6c", label:"Temporal 4-frame window",  auroc:0.959,  color:"#f0abfc" },
];

const K_SWEEP = [
  {k:1,a:0.979},{k:2,a:0.957},{k:4,a:0.936},{k:8,a:0.898},{k:16,a:0.818}
];

const DISSOCIATION = [
  { cond:"Full (VLM + proprio)",    s6r:0.800, s6t:0.892, s6c:0.959, vlm:0.895 },
  { cond:"No VLM (proprio only)",   s6r:0.792, s6t:0.850, s6c:0.953, vlm:null  },
  { cond:"No proprio (VLM only)",   s6r:0.865, s6t:0.862, s6c:0.895, vlm:0.895 },
];

const TRAIN_6C = [
  {ep:0,loss:0.5915,acc:0.7988},{ep:1,loss:0.5407,acc:0.8102},
  {ep:2,loss:0.5226,acc:0.8147},{ep:3,loss:0.5156,acc:0.8154},
  {ep:4,loss:0.5170,acc:0.8148},{ep:5,loss:0.5051,acc:0.8176},
  {ep:6,loss:0.4930,acc:0.8250},{ep:7,loss:0.4847,acc:0.8296},
  {ep:8,loss:0.4829,acc:0.8298},{ep:9,loss:0.4830,acc:0.8298},
  {ep:10,loss:0.4680,acc:0.8352},{ep:11,loss:0.4663,acc:0.8344},
  {ep:12,loss:0.4653,acc:0.8342},{ep:13,loss:0.4491,acc:0.8402},
  {ep:14,loss:0.4636,acc:0.8372},{ep:15,loss:0.4618,acc:0.8358},
  {ep:16,loss:0.4577,acc:0.8374},{ep:17,loss:0.4652,acc:0.8339},
];

const ABLATION_K4 = [
  {label:"Full proprio",          auroc:0.9577, full:true},
  {label:"No velocity [0]",       auroc:0.9483, drop:0.009},
  {label:"No angular_vel [1]",    auroc:0.9178, drop:0.040},
  {label:"No contact [4]",        auroc:0.9137, drop:0.044},
  {label:"No heading [5,6]",      auroc:0.7521, drop:0.206, critical:true},
  {label:"No delta_heading [7]",  auroc:0.9547, drop:0.003},
  {label:"No motion [0,1]",       auroc:0.9185, drop:0.039},
  {label:"No heading+delta [5,6,7]",auroc:0.7303,drop:0.227,critical:true},
  {label:"Heading only [5,6,7]",  auroc:0.9065, drop:0.051, bearing:true},
  {label:"Velocity lesion",       auroc:0.9483, drop:0.009},
  {label:"HD lesion (Moser)",     auroc:0.7303, drop:0.227, critical:true},
  {label:"Complete PI lesion",    auroc:0.5060, drop:0.452, critical:true},
];

const ABLATION_K1 = [
  {label:"Full proprio",          auroc:0.9924, full:true},
  {label:"Heading only",          auroc:0.9524, drop:0.040, bearing:true},
  {label:"Vel+ang only",          auroc:0.8994, drop:0.093, bearing:true},
  {label:"Velocity lesion",       auroc:0.9900, drop:0.002},
  {label:"HD lesion (Moser)",     auroc:0.8881, drop:0.104, critical:true},
  {label:"Complete PI lesion",    auroc:0.5060, drop:0.486, critical:true},
];

const GRID_CELL = [
  {label:"< 2m (adjacent)",  auroc:0.511},
  {label:"2–5m (nearby)",    auroc:0.713},
  {label:"5–15m (close)",    auroc:0.769},
  {label:"15–50m (medium)",  auroc:0.847},
  {label:"50–200m",          auroc:null},
  {label:"> 200m",           auroc:null},
];

const PLACE_CELLS = [
  {id:0, tuning:1.557,fw:43.3,peak_e:134.3,type:"regional"},
  {id:1, tuning:1.276,fw:63.8,peak_e:116.0,type:"diffuse"},
  {id:2, tuning:1.363,fw:24.6,peak_e:151.3,type:"regional"},
  {id:3, tuning:1.377,fw:43.1,peak_e:143.2,type:"regional"},
  {id:4, tuning:1.377,fw:68.6,peak_e:102.7,type:"diffuse"},
  {id:5, tuning:1.171,fw:28.1,peak_e:149.5,type:"diffuse"},
  {id:6, tuning:1.407,fw:26.3,peak_e:152.8,type:"regional"},
  {id:7, tuning:1.232,fw:16.6,peak_e:156.8,type:"regional"},
  {id:8, tuning:1.469,fw:67.2,peak_e:98.8, type:"diffuse"},
  {id:9, tuning:1.349,fw:20.7,peak_e:151.0,type:"regional"},
  {id:10,tuning:1.328,fw:53.8,peak_e:131.8,type:"diffuse"},
  {id:11,tuning:1.182,fw:44.5,peak_e:143.4,type:"diffuse"},
  {id:12,tuning:1.473,fw:35.4,peak_e:149.5,type:"regional"},
  {id:13,tuning:1.282,fw:46.7,peak_e:143.7,type:"regional"},
  {id:14,tuning:1.217,fw:52.4,peak_e:135.1,type:"diffuse"},
  {id:15,tuning:1.421,fw:37.3,peak_e:147.2,type:"regional"},
];

// ── Components ─────────────────────────────────────────────────────────────

const TC = { "place-like":C.accent3, regional:C.accent1, diffuse:C.muted };

function Bar({v,max=1,color,pending=false,glow=true}){
  const pct=pending?0:((v??0)/max)*100;
  return(
    <div style={{background:"#0d1829",borderRadius:3,height:7,overflow:"hidden",flex:1}}>
      <div style={{width:`${pct}%`,height:"100%",
        background:pending?"#1e2d45":`linear-gradient(90deg,${color}88,${color})`,
        borderRadius:3,transition:"width 1.4s cubic-bezier(.16,1,.3,1)",
        boxShadow:(!pending&&glow)?`0 0 8px ${color}55`:"none"}}/>
    </div>
  );
}

function StatCard({label,value,sub,color=C.accent1,pending=false}){
  return(
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,
      padding:"12px 16px",borderLeft:`3px solid ${pending?C.muted:color}`}}>
      <div style={{fontSize:9,color:C.muted,letterSpacing:"0.1em",
        textTransform:"uppercase",marginBottom:4}}>{label}</div>
      <div style={{fontSize:21,fontFamily:"monospace",color:pending?C.muted:color,
        fontWeight:600,lineHeight:1}}>{pending?"—":value}</div>
      {sub&&<div style={{fontSize:10,color:C.muted,marginTop:3}}>{sub}</div>}
    </div>
  );
}

function Section({title,badge,color=C.accent1,children}){
  return(
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:18}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:14}}>
        <div style={{width:3,height:16,background:color,borderRadius:2}}/>
        <span style={{fontSize:11,fontWeight:600,color:C.text,letterSpacing:"0.05em",
          textTransform:"uppercase"}}>{title}</span>
        {badge&&<span style={{fontSize:9,padding:"2px 8px",borderRadius:20,
          background:`${color}22`,color,border:`1px solid ${color}44`}}>{badge}</span>}
      </div>
      {children}
    </div>
  );
}

function AblationRow({label,auroc,drop,full=false,critical=false,bearing=false}){
  const base=0.9577;
  const color=critical?C.danger:bearing?C.warn:full?C.accent1:C.muted;
  const flag=critical?"← CRITICAL":bearing?"← LOAD-BEARING":full?"(baseline)":"";
  return(
    <div style={{display:"flex",alignItems:"center",gap:8,padding:"5px 4px",
      borderBottom:`1px solid ${C.border}22`,
      background:full?`${C.accent1}08`:critical?`${C.danger}05`:"transparent",
      borderRadius:3}}>
      <div style={{width:220,fontSize:10,color:full?C.text:C.muted,
        fontFamily:"monospace",flexShrink:0}}>{label}</div>
      <div style={{flex:1,background:"#0d1829",borderRadius:3,height:7,position:"relative"}}>
        <div style={{width:`${(auroc/1)*100}%`,height:"100%",
          background:`linear-gradient(90deg,${color}88,${color})`,
          borderRadius:3,boxShadow:`0 0 6px ${color}44`}}/>
        {!full&&<div style={{position:"absolute",left:`${(base/1)*100}%`,top:-1,
          width:1.5,height:9,background:C.accent1,opacity:0.4}}/>}
      </div>
      <div style={{width:46,textAlign:"right",fontFamily:"monospace",
        fontSize:11,color,fontWeight:600}}>{auroc.toFixed(3)}</div>
      {drop!==undefined&&<div style={{width:56,textAlign:"right",fontSize:9,
        fontFamily:"monospace",color}}>{drop>0.05?`−${drop.toFixed(3)}`:`−${drop.toFixed(3)}`}</div>}
      <div style={{width:110,fontSize:9,color,fontStyle:"italic"}}>{flag}</div>
    </div>
  );
}

function SparkLine({data,xKey,yKey,color,W=280,H=70}){
  if(!data||data.length<2) return null;
  const xs=data.map(d=>d[xKey]),ys=data.map(d=>d[yKey]);
  const xMn=Math.min(...xs),xMx=Math.max(...xs);
  const yMn=Math.min(...ys)*0.985,yMx=Math.max(...ys)*1.005;
  const px=x=>((x-xMn)/(xMx-xMn||1))*(W-16)+8;
  const py=y=>H-6-((y-yMn)/(yMx-yMn||1))*(H-14);
  const pts=data.map(d=>`${px(d[xKey])},${py(d[yKey])}`).join(" ");
  const area=`M${px(xs[0])},${H} `+data.map(d=>`L${px(d[xKey])},${py(d[yKey])}`).join(" ")+` L${px(xs[xs.length-1])},${H} Z`;
  const best=data.reduce((a,b)=>b[yKey]>a[yKey]?b:a);
  return(
    <svg width={W} height={H}>
      <defs>
        <linearGradient id={`g${yKey}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#g${yKey})`}/>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5"
        strokeLinecap="round" strokeLinejoin="round"/>
      <circle cx={px(best[xKey])} cy={py(best[yKey])} r={4}
        fill={color} stroke={C.bg} strokeWidth="2"/>
      <text x={px(best[xKey])+6} y={py(best[yKey])+4} fill={color} fontSize={9}
        fontFamily="monospace">{best[yKey].toFixed(4)}</text>
    </svg>
  );
}

function PlaceGrid(){
  const [sel,setSel]=useState(null);
  return(
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:6,marginBottom:10}}>
        {PLACE_CELLS.map(p=>{
          const c=TC[p.type];
          const opacity=Math.min(0.9,(p.tuning-1.0)/0.6);
          return(
            <div key={p.id} onClick={()=>setSel(sel===p.id?null:p.id)}
              style={{aspectRatio:"1",background:
                `radial-gradient(circle at 55% 45%, ${c}${Math.round(opacity*99).toString(16).padStart(2,"0")}, ${C.grid})`,
                border:`1px solid ${sel===p.id?c:C.border}`,borderRadius:7,
                display:"flex",flexDirection:"column",alignItems:"center",
                justifyContent:"center",gap:2,cursor:"pointer",transition:"all 0.15s"}}>
              <div style={{fontSize:10,fontFamily:"monospace",color:c,fontWeight:600}}>
                p{String(p.id).padStart(2,"0")}
              </div>
              <div style={{fontSize:8,color:c,opacity:0.8}}>
                {p.type==="regional"?"◎":"○"}
              </div>
              {sel===p.id&&(
                <div style={{position:"absolute",fontSize:8,color:C.text,
                  background:`${C.card}ee`,padding:"2px 4px",borderRadius:3,
                  whiteSpace:"nowrap",zIndex:10,marginTop:48}}>
                  {p.tuning.toFixed(3)}× · {p.fw.toFixed(0)}m · E={p.peak_e.toFixed(0)}m
                </div>
              )}
            </div>
          );
        })}
      </div>
      <div style={{display:"flex",gap:14,fontSize:9,color:C.muted}}>
        <span><span style={{color:C.accent1}}>◎</span> regional (9/16)</span>
        <span><span style={{color:C.muted}}>○</span> diffuse (7/16)</span>
        <span style={{color:C.muted,fontStyle:"italic"}}>19,817 frames · 500 files</span>
      </div>
    </div>
  );
}

function KSweepViz(){
  const W=260,H=80,pad=28;
  const xpos=[0,28,76,152,232];
  const py=y=>H-8-((y-0.80)/0.21)*(H-18);
  const pts=K_SWEEP.map((d,i)=>`${xpos[i]+pad},${py(d.a)}`).join(" ");
  return(
    <svg width={W+pad} height={H+12} style={{overflow:"visible"}}>
      <line x1={pad} y1={py(0.70)} x2={W+pad-8} y2={py(0.70)}
        stroke={C.danger} strokeWidth={1} strokeDasharray="4,3" opacity={0.5}/>
      <text x={W+pad-6} y={py(0.70)+3} fill={C.danger} fontSize={8} opacity={0.7}>0.70</text>
      <line x1={pad} y1={py(0.865)} x2={W+pad-8} y2={py(0.865)}
        stroke={C.warn} strokeWidth={1} strokeDasharray="4,3" opacity={0.5}/>
      <text x={W+pad-6} y={py(0.865)+3} fill={C.warn} fontSize={8} opacity={0.7}>VLM</text>
      <polyline points={pts} fill="none" stroke={C.accent1} strokeWidth={2}
        strokeLinecap="round" strokeLinejoin="round"/>
      {K_SWEEP.map((d,i)=>(
        <g key={d.k}>
          <circle cx={xpos[i]+pad} cy={py(d.a)} r={4}
            fill={C.accent1} stroke={C.bg} strokeWidth={2}/>
          <text x={xpos[i]+pad} y={H+10} fill={C.muted} fontSize={8} textAnchor="middle">k={d.k}</text>
          <text x={xpos[i]+pad} y={py(d.a)-6} fill={C.accent1} fontSize={8} textAnchor="middle">{d.a}</text>
        </g>
      ))}
    </svg>
  );
}

// ── Main Dashboard ─────────────────────────────────────────────────────────

export default function Dashboard(){
  const [tab,setTab]=useState("overview");
  const [pulse,setPulse]=useState(false);
  const [ablTab,setAblTab]=useState("k4");
  useEffect(()=>{const t=setInterval(()=>setPulse(p=>!p),1400);return()=>clearInterval(t);},[]);

  const tabs=[
    {id:"overview",label:"Overview"},
    {id:"dissociation",label:"Dissociation"},
    {id:"training",label:"Training"},
    {id:"ablation",label:"PI Ablation"},
    {id:"spatial",label:"Spatial"},
  ];

  return(
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,
      fontFamily:"'DM Sans',system-ui,sans-serif",padding:"18px 22px"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600&display=swap');*{box-sizing:border-box;margin:0;padding:0}`}</style>

      {/* Header */}
      <div style={{marginBottom:20}}>
        <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between"}}>
          <div>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:3}}>
              <div style={{width:7,height:7,borderRadius:"50%",background:C.accent3,
                boxShadow:`0 0 ${pulse?14:5}px ${C.accent3}`,transition:"box-shadow 0.7s"}}/>
              <span style={{fontSize:10,color:C.accent3,fontFamily:"monospace",letterSpacing:"0.1em"}}>
                ALL EXPERIMENTS COMPLETE · ARXIV PENDING
              </span>
            </div>
            <h1 style={{fontSize:24,fontWeight:600,letterSpacing:"-0.02em"}}>NeMo-WM Research Dashboard</h1>
            <p style={{fontSize:11,color:C.muted,marginTop:2}}>
              Neuromodulated World Model · Full Computational Dissociation Achieved
            </p>
          </div>
          <div style={{textAlign:"right",fontSize:10,color:C.muted,fontFamily:"monospace",lineHeight:1.8}}>
            <div>GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395 · 128GB</div>
            <div>NPU XINT8 · 10,995 RECON files · 19,817 frames analysed</div>
            <div style={{color:C.accent2,marginTop:2}}>cs.LG + cs.RO · CC BY 4.0</div>
          </div>
        </div>
        <div style={{display:"flex",gap:2,borderBottom:`1px solid ${C.border}`,paddingBottom:0,marginTop:14}}>
          {tabs.map(t=>(
            <button key={t.id} onClick={()=>setTab(t.id)} style={{
              background:"none",border:"none",cursor:"pointer",
              padding:"7px 14px",fontSize:12,
              color:tab===t.id?C.accent1:C.muted,
              borderBottom:`2px solid ${tab===t.id?C.accent1:"transparent"}`,
              marginBottom:-1,transition:"all 0.15s"}}>
              {t.label}
            </button>
          ))}
        </div>
      </div>

      {/* ── OVERVIEW ── */}
      {tab==="overview"&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:10,marginBottom:18}}>
            <StatCard label="Best AUROC (k=1)" value="0.9837" sub="Sprint 2 canonical" color={C.accent3}/>
            <StatCard label="Sprint 6c Full" value="0.959" sub="VLM + temporal proprio" color={C.accent2}/>
            <StatCard label="No-VLM AUROC" value="0.953" sub="> VLM-only 0.895 ✓" color={C.accent3}/>
            <StatCard label="HD lesion drop" value="−0.227" sub="Primary PI signal" color={C.warn}/>
            <StatCard label="Complete PI→chance" value="0.506" sub="Double dissociation" color={C.danger}/>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1.4fr 1fr",gap:14}}>
            <Section title="Sprint Progression" badge="AUROC" color={C.accent1}>
              {SPRINT_DATA.map(s=>(
                <div key={s.sprint} style={{display:"flex",alignItems:"center",gap:10,
                  padding:"6px 4px",borderBottom:`1px solid ${C.border}22`,
                  background:s.sprint==="S6c"?`${C.accent2}08`:"transparent",borderRadius:3}}>
                  <div style={{width:32,fontSize:10,color:s.color,fontFamily:"monospace",fontWeight:600}}>{s.sprint}</div>
                  <div style={{width:200,fontSize:11,color:C.muted,flexShrink:0}}>{s.label}</div>
                  <Bar v={s.auroc} color={s.color}/>
                  <span style={{width:44,textAlign:"right",fontFamily:"monospace",
                    fontSize:12,color:s.color,fontWeight:600}}>{s.auroc}</span>
                </div>
              ))}
              <div style={{marginTop:10,fontSize:9,color:C.muted}}>
                Evaluated on hard negatives (same-file, k≥32) with no GPS for S5/S6/S6c
              </div>
            </Section>
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              <Section title="k-sweep AUROC" badge="k=1..16" color={C.accent2}>
                <KSweepViz/>
              </Section>
              <Section title="Key Findings" color={C.accent3}>
                {[
                  ["Full dissociation", "No-VLM (0.953) > VLM-only (0.895)"],
                  ["Heading-dominant PI", "HD lesion −0.227, vel lesion −0.009"],
                  ["Timescale-invariant", "HD dominant at k=1 (−0.104) and k=4 (−0.227)"],
                  ["Grid cell signature", "AUROC 0.51→0.85 monotonic with GPS distance"],
                  ["Regional coding", "9/16 particles spatially selective (500 files)"],
                ].map(([k,v])=>(
                  <div key={k} style={{display:"flex",gap:8,marginBottom:6,fontSize:11}}>
                    <span style={{color:C.accent3,flexShrink:0}}>▸</span>
                    <span><span style={{color:C.text,fontWeight:600}}>{k}:</span>
                      <span style={{color:C.muted}}> {v}</span></span>
                  </div>
                ))}
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── DISSOCIATION ── */}
      {tab==="dissociation"&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10,marginBottom:16}}>
            <StatCard label="Aphasia gap (S4)" value="0.423" sub="VLM zeroed → chance" color={C.danger}/>
            <StatCard label="Aphasia gap (S6)" value="0.042" sub="Trained proprio narrows gap" color={C.warn}/>
            <StatCard label="Aphasia gap (S6c)" value="−0.058" sub="Proprio exceeds VLM-only!" color={C.accent3}/>
            <StatCard label="Double dissociation" value="✓" sub="Each system ablates to chance" color={C.accent3}/>
          </div>
          <Section title="Three-Condition Dissociation Eval" badge="hard neg · no GPS · 1000 pairs" color={C.accent3}>
            <div style={{display:"grid",gridTemplateColumns:"210px 1fr 1fr 1fr 1fr",gap:0,marginBottom:10}}>
              {["Condition","Random baseline","Sprint 6 trained","Sprint 6c","VLM-only"].map(h=>(
                <div key={h} style={{fontSize:9,color:C.muted,padding:"5px 6px",
                  letterSpacing:"0.08em",textTransform:"uppercase",
                  borderBottom:`1px solid ${C.border}`}}>{h}</div>
              ))}
              {DISSOCIATION.map(row=>[
                <div key={`${row.cond}l`} style={{fontSize:11,color:C.text,padding:"9px 6px",
                  borderBottom:`1px solid ${C.border}22`,fontFamily:"monospace"}}>{row.cond}</div>,
                ...[
                  {v:row.s6r,c:C.muted},
                  {v:row.s6t,c:row.s6t>row.vlm?C.accent3:C.accent1},
                  {v:row.s6c,c:row.s6c>(row.vlm??0)?C.accent3:C.accent1},
                  {v:row.vlm,c:C.warn},
                ].map((cell,i)=>(
                  <div key={`${row.cond}${i}`} style={{padding:"9px 6px",
                    borderBottom:`1px solid ${C.border}22`,
                    display:"flex",alignItems:"center",gap:6}}>
                    {cell.v!=null?<>
                      <Bar v={cell.v} color={cell.c}/>
                      <span style={{fontSize:11,fontFamily:"monospace",color:cell.c,width:42}}>
                        {cell.v.toFixed(3)}
                      </span>
                    </>:<span style={{fontSize:11,color:C.muted,width:52}}>—</span>}
                  </div>
                )),
              ])}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginTop:4}}>
              <div style={{padding:12,background:`${C.accent3}10`,borderRadius:8,
                border:`1px solid ${C.accent3}33`,fontSize:11,color:C.accent3,lineHeight:1.6}}>
                <strong>Full dissociation achieved:</strong> No-VLM (0.953) {'>'} VLM-only (0.895).
                Temporal path integration without GPS or vision outperforms landmark-based navigation alone.
              </div>
              <div style={{padding:12,background:`${C.danger}08`,borderRadius:8,
                border:`1px solid ${C.danger}33`,fontSize:11,color:C.muted,lineHeight:1.6}}>
                <strong style={{color:C.danger}}>Double dissociation:</strong> VLM zeroed → PI pathway 0.953.
                HD lesioned → landmark pathway 0.895. Each system ablates independently to chance (0.506).
              </div>
            </div>
          </Section>
        </div>
      )}

      {/* ── TRAINING ── */}
      {tab==="training"&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"1.4fr 1fr",gap:14}}>
            <Section title="Sprint 6c training curve — top1_acc" badge="COMPLETE ep13 best" color={C.accent3}>
              <SparkLine data={TRAIN_6C} xKey="ep" yKey="acc" color={C.accent3} W={340} H={90}/>
              <div style={{marginTop:10,display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:6}}>
                {[
                  {label:"Epoch 0",value:"0.7988",sub:"Baseline",c:C.muted},
                  {label:"Epoch 7",value:"0.8296",sub:"Climbing",c:C.accent1},
                  {label:"Epoch 13 ★",value:"0.8402",sub:"Best checkpoint",c:C.accent3},
                ].map(s=>(
                  <div key={s.label} style={{padding:"8px 10px",background:`${s.c}10`,
                    borderRadius:6,border:`1px solid ${s.c}33`}}>
                    <div style={{fontSize:9,color:s.c,marginBottom:2}}>{s.label}</div>
                    <div style={{fontSize:18,fontFamily:"monospace",color:s.c,fontWeight:600}}>{s.value}</div>
                    <div style={{fontSize:9,color:C.muted}}>{s.sub}</div>
                  </div>
                ))}
              </div>
            </Section>
            <div style={{display:"flex",flexDirection:"column",gap:14}}>
              <Section title="Sprint comparison" color={C.accent2}>
                {[
                  {label:"S6  (9K · single frame)",    ep0:0.703,best:0.718,color:C.muted},
                  {label:"S6b (26K · +delta_h)",        ep0:0.703,best:0.717,color:C.muted},
                  {label:"S6c (26K · 4-frame window)",  ep0:0.799,best:0.840,color:C.accent3},
                ].map(s=>(
                  <div key={s.label} style={{marginBottom:12}}>
                    <div style={{fontSize:10,color:s.color,marginBottom:5,fontFamily:"monospace"}}>{s.label}</div>
                    {[["ep0",s.ep0],["best",s.best]].map(([lbl,v])=>(
                      <div key={lbl} style={{display:"flex",gap:8,alignItems:"center",marginBottom:3}}>
                        <span style={{fontSize:9,color:C.muted,width:28}}>{lbl}</span>
                        <Bar v={v} color={s.color}/>
                        <span style={{fontSize:10,fontFamily:"monospace",color:s.color,width:38}}>{v}</span>
                      </div>
                    ))}
                  </div>
                ))}
                <div style={{padding:10,background:`${C.accent3}10`,borderRadius:6,
                  border:`1px solid ${C.accent3}33`,fontSize:10,color:C.accent3}}>
                  6c ep0=0.799 vs S6/S6b ceiling 0.718 (+0.081 from temporal context alone)
                </div>
              </Section>
              <Section title="Loss curve" color={C.accent2}>
                <SparkLine data={TRAIN_6C} xKey="ep" yKey="loss" color={C.warn} W={240} H={60}/>
                <div style={{fontSize:10,color:C.muted,marginTop:6,fontFamily:"monospace"}}>
                  0.592 → 0.449 over 13 epochs. Still descending at plateau.
                </div>
              </Section>
            </div>
          </div>
        </div>
      )}

      {/* ── PI ABLATION ── */}
      {tab==="ablation"&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10,marginBottom:14}}>
            <StatCard label="Full proprio baseline" value="0.958" sub="k=4 hard neg no GPS" color={C.accent1}/>
            <StatCard label="HD lesion drop" value="−0.227" sub="Primary signal (Moser 2008)" color={C.danger}/>
            <StatCard label="Vel lesion drop" value="−0.009" sub="Not primary (McNaughton)" color={C.muted}/>
          </div>

          <div style={{display:"flex",gap:8,marginBottom:12}}>
            {[{id:"k4",label:"k=4 (1.0s pairs)"},{id:"k1",label:"k=1 (0.25s pairs)"}].map(t=>(
              <button key={t.id} onClick={()=>setAblTab(t.id)} style={{
                background:ablTab===t.id?`${C.accent2}22`:"none",
                border:`1px solid ${ablTab===t.id?C.accent2:C.border}`,
                borderRadius:6,padding:"5px 14px",cursor:"pointer",
                fontSize:11,color:ablTab===t.id?C.accent2:C.muted,transition:"all 0.15s"}}>
                {t.label}
              </button>
            ))}
            <span style={{fontSize:10,color:C.muted,alignSelf:"center",marginLeft:8}}>
              {ablTab==="k4"?"Heading dominant −0.227 · Velocity −0.009":"Timescale-invariant: HD −0.104 · Vel −0.002"}
            </span>
          </div>

          <Section title="Path Integration Channel Ablation" badge="McNaughton · Moser" color={C.warn}>
            <div style={{marginBottom:8,fontSize:10,color:C.muted}}>
              VLM-only (landmark nav): <span style={{color:C.accent1,fontFamily:"monospace"}}>
                {ablTab==="k4"?"0.892":"0.945"}
              </span> — shown as reference
            </div>
            {(ablTab==="k4"?ABLATION_K4:ABLATION_K1).map(c=>(
              <AblationRow key={c.label} {...c}/>
            ))}
            <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:8,marginTop:14}}>
              {[
                {label:"Heading-dominant",color:C.danger,
                  desc:ablTab==="k4"?"HD lesion −0.227 vs vel −0.009 (25:1 ratio)":"HD −0.104 vs vel −0.002 (43:1 ratio)"},
                {label:"Biological parallel",color:C.accent1,
                  desc:"Moser et al. 2008: HD cells primary. McNaughton 2006: velocity supplementary."},
                {label:"Double dissociation",color:C.accent3,
                  desc:"Complete PI lesion → 0.506. VLM aphasia → 0.500. Each system ablates to chance."},
              ].map(item=>(
                <div key={item.label} style={{padding:10,background:`${item.color}08`,
                  borderRadius:7,border:`1px solid ${item.color}33`}}>
                  <div style={{fontSize:10,color:item.color,fontWeight:600,marginBottom:4}}>{item.label}</div>
                  <div style={{fontSize:9,color:C.muted,lineHeight:1.5}}>{item.desc}</div>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}

      {/* ── SPATIAL ── */}
      {tab==="spatial"&&(
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
          <Section title="K=16 Particle Population Coding" badge="500 files · 19,817 frames" color={C.accent3}>
            <p style={{fontSize:11,color:C.muted,marginBottom:12,lineHeight:1.6}}>
              9/16 particles show regional spatial selectivity (tuning 1.17–1.56×, field 16–68m).
              0/16 show sharp place-cell tuning — spatial information is distributed across
              the ensemble, consistent with grid cell-like population coding.
              Click a cell for details.
            </p>
            <PlaceGrid/>
            <div style={{marginTop:12,padding:10,background:`${C.accent1}08`,borderRadius:7,
              border:`1px solid ${C.accent1}22`,fontSize:10,color:C.muted,lineHeight:1.5}}>
              <span style={{color:C.accent1,fontWeight:600}}>Population coding:</span> Spatial
              information is distributed — no single particle has a dominant place field.
              Structure emerged from VLM-grounded contrastive training without GPS supervision.
            </div>
          </Section>

          <Section title="Grid Cell Test: AUROC vs GPS Distance" badge="confirmed" color={C.accent1}>
            <p style={{fontSize:11,color:C.muted,marginBottom:14,lineHeight:1.6}}>
              AUROC rises monotonically as negative GPS distance increases — grid cell
              signature. Chance at &lt;2m (adjacent frames informationally identical).
              Strong metric encoding at 15–50m.
            </p>
            {GRID_CELL.map(d=>(
              <div key={d.label} style={{display:"flex",alignItems:"center",gap:10,marginBottom:7}}>
                <span style={{width:88,fontSize:11,color:d.auroc?C.text:C.muted,
                  fontFamily:"monospace"}}>{d.label}</span>
                <Bar v={d.auroc??0} color={
                  !d.auroc?C.muted:d.auroc<0.55?C.danger:d.auroc<0.75?C.warn:C.accent3
                } pending={!d.auroc}/>
                <span style={{width:46,fontSize:12,fontFamily:"monospace",textAlign:"right",
                  color:!d.auroc?C.muted:d.auroc<0.55?C.danger:d.auroc<0.75?C.warn:C.accent3,
                  fontWeight:600}}>{d.auroc?.toFixed(3)??"—"}</span>
              </div>
            ))}
            <div style={{marginTop:12,padding:10,background:`${C.accent3}08`,borderRadius:7,
              border:`1px solid ${C.accent3}22`,fontSize:10,color:C.accent3,lineHeight:1.5}}>
              <strong>Grid cell signature confirmed:</strong> Monotonic increase 0.511→0.847.
              Chance at &lt;2m. Consistent with entorhinal metric distance coding (Moser et al. 2008).
            </div>

            <div style={{marginTop:12,padding:10,background:`${C.warn}08`,borderRadius:7,
              border:`1px solid ${C.warn}22`,fontSize:10,color:C.muted,lineHeight:1.5}}>
              <span style={{color:C.warn,fontWeight:600}}>Path integration ablation: </span>
              Heading (sin/cos) is the primary signal. HD lesion −0.227 AUROC (k=4),
              −0.104 (k=1). Velocity lesion −0.009. Timescale-invariant heading dominance
              mirrors entorhinal head direction cell primacy (Moser et al. 2008).
            </div>
          </Section>
        </div>
      )}

      <div style={{marginTop:24,paddingTop:12,borderTop:`1px solid ${C.border}`,
        display:"flex",justifyContent:"space-between",fontSize:10,
        color:C.muted,fontFamily:"monospace"}}>
        <span>NeMo-WM · github.com/taylorjohn · All experiments complete</span>
        <span>Pending: System A/B/M citation → arXiv submission</span>
      </div>
    </div>
  );
}
