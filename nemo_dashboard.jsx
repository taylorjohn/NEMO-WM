import { useState, useEffect } from "react";

const C = {
  bg:"#050810",surface:"#0c1020",card:"#111827",border:"#1e2d45",
  accent1:"#00d4ff",accent2:"#7c3aed",accent3:"#10b981",warn:"#f59e0b",
  danger:"#ef4444",text:"#e2e8f0",muted:"#64748b",grid:"#0d1829",
};

const SPRINT_DATA=[
  {sprint:"S2",label:"RECON Baseline",auroc:0.9075,color:"#4ade80"},
  {sprint:"S4",label:"Aphasia ablated→0.500",auroc:0.923,vlm:0.923,novlm:0.500,color:"#f87171"},
  {sprint:"S5",label:"Hard negatives",auroc:0.865,vlm:0.865,color:"#60a5fa"},
  {sprint:"S6",label:"Proprio single frame",auroc:0.892,vlm:0.862,novlm:0.850,color:"#a78bfa"},
  {sprint:"S6b",label:"Proprio +rel_heading",auroc:null,vlm:0.862,color:"#c084fc"},
  {sprint:"S6c",label:"Temporal 4-frame window",auroc:null,vlm:0.862,color:"#f0abfc"},
];

const K_SWEEP=[{k:1,a:0.979},{k:2,a:0.957},{k:4,a:0.936},{k:8,a:0.898},{k:16,a:0.818}];

const DISSOCIATION=[
  {cond:"Full (VLM+proprio)",    s6r:0.800,s6t:0.892,s6c:null,vlm:0.865},
  {cond:"No VLM (proprio only)", s6r:0.792,s6t:0.850,s6c:null,vlm:null},
  {cond:"No proprio (VLM only)", s6r:0.865,s6t:0.862,s6c:null,vlm:0.865},
];

const TRAIN_6C=[
  {ep:0,loss:0.5915,acc:0.7988},
  {ep:1,loss:0.5407,acc:0.8102},
];

const ABLATION=[
  {label:"Full proprio",auroc:0.892,full:true},
  {label:"No velocity [0]",auroc:null},{label:"No angular_vel [1]",auroc:null},
  {label:"No heading [5,6]",auroc:null},{label:"No delta_h [7]",auroc:null},
  {label:"Velocity lesion (McNaughton)",auroc:null},
  {label:"HD lesion (Moser)",auroc:null},
  {label:"Complete PI lesion",auroc:null},
];

function Bar({v,max=1,color,pending=false}){
  const pct=pending?0:((v??0)/max)*100;
  return(
    <div style={{background:"#0d1829",borderRadius:3,height:7,overflow:"hidden",flex:1}}>
      <div style={{width:`${pct}%`,height:"100%",background:pending?"#1e2d45":color,
        borderRadius:3,transition:"width 1.2s cubic-bezier(.16,1,.3,1)",
        boxShadow:pending?"none":`0 0 8px ${color}66`}}/>
    </div>
  );
}

function StatCard({label,value,sub,color=C.accent1,pending=false}){
  return(
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:10,
      padding:"14px 18px",borderLeft:`3px solid ${pending?C.muted:color}`}}>
      <div style={{fontSize:10,color:C.muted,letterSpacing:"0.1em",textTransform:"uppercase",marginBottom:5}}>{label}</div>
      <div style={{fontSize:22,fontFamily:"monospace",color:pending?C.muted:color,fontWeight:600}}>{pending?"—":value}</div>
      {sub&&<div style={{fontSize:10,color:C.muted,marginTop:3}}>{sub}</div>}
    </div>
  );
}

function Section({title,badge,color=C.accent1,children}){
  return(
    <div style={{background:C.card,border:`1px solid ${C.border}`,borderRadius:12,padding:20}}>
      <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:16}}>
        <div style={{width:3,height:18,background:color,borderRadius:2}}/>
        <span style={{fontSize:12,fontWeight:600,color:C.text,letterSpacing:"0.05em",textTransform:"uppercase"}}>{title}</span>
        {badge&&<span style={{fontSize:9,padding:"2px 8px",borderRadius:20,background:`${color}22`,color,border:`1px solid ${color}44`}}>{badge}</span>}
      </div>
      {children}
    </div>
  );
}

function AurocRow({label,value,baseline,pending=false,hl=false}){
  const color=pending?C.muted:value>(baseline??0)?C.accent3:C.accent1;
  const drop=baseline&&value&&!pending?value-baseline:null;
  return(
    <div style={{display:"flex",alignItems:"center",gap:10,padding:"7px 6px",
      borderBottom:`1px solid ${C.border}22`,background:hl?`${C.accent1}08`:"transparent",
      borderRadius:4}}>
      <div style={{width:190,fontSize:11,color:C.muted,fontFamily:"monospace",flexShrink:0}}>{label}</div>
      <div style={{flex:1,position:"relative",height:18}}>
        <div style={{position:"absolute",top:5,left:0,right:0,background:"#0d1829",borderRadius:3,height:8}}>
          <div style={{width:pending?"0%":`${((value??0)/1)*100}%`,height:"100%",
            background:`linear-gradient(90deg,${color}88,${color})`,borderRadius:3,
            transition:"width 1.2s cubic-bezier(.16,1,.3,1)",boxShadow:pending?"none":`0 0 8px ${color}44`}}/>
          {baseline&&<div style={{position:"absolute",left:`${(baseline/1)*100}%`,top:-1,
            width:2,height:10,background:C.warn,borderRadius:1}}/>}
        </div>
      </div>
      <div style={{width:52,textAlign:"right",fontFamily:"monospace",fontSize:12,
        color:pending?C.muted:color,fontWeight:600}}>{pending?"pending":(value?.toFixed(4)??"—")}</div>
      {drop!==null&&<div style={{width:50,textAlign:"right",fontSize:10,fontFamily:"monospace",
        color:drop>0?C.accent3:C.danger}}>{drop>0?"+":""}{drop.toFixed(3)}</div>}
    </div>
  );
}

function KSweep(){
  const W=280,H=100,pad=32;
  const xs=[0,30,80,150,240];
  const py=y=>H-10-((y-0.80)/0.21)*(H-20);
  const pts=K_SWEEP.map((d,i)=>`${xs[i]+pad},${py(d.a)}`).join(" ");
  return(
    <svg width={W+pad} height={H+10} style={{overflow:"visible"}}>
      <line x1={pad} y1={py(0.70)} x2={W+pad-10} y2={py(0.70)} stroke={C.danger} strokeWidth={1} strokeDasharray="4,3" opacity={0.5}/>
      <text x={W+pad-8} y={py(0.70)+3} fill={C.danger} fontSize={9} opacity={0.7}>0.70</text>
      <line x1={pad} y1={py(0.862)} x2={W+pad-10} y2={py(0.862)} stroke={C.warn} strokeWidth={1} strokeDasharray="4,3" opacity={0.5}/>
      <text x={W+pad-8} y={py(0.862)+3} fill={C.warn} fontSize={9} opacity={0.7}>VLM</text>
      <polyline points={pts} fill="none" stroke={C.accent1} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"/>
      {K_SWEEP.map((d,i)=>(
        <g key={d.k}>
          <circle cx={xs[i]+pad} cy={py(d.a)} r={4} fill={C.accent1} stroke={C.bg} strokeWidth={2}/>
          <text x={xs[i]+pad} y={H+8} fill={C.muted} fontSize={9} textAnchor="middle">k={d.k}</text>
          <text x={xs[i]+pad} y={py(d.a)-7} fill={C.accent1} fontSize={9} textAnchor="middle">{d.a}</text>
        </g>
      ))}
    </svg>
  );
}

function SparkLine({data,xKey,yKey,color,W=260,H=70}){
  if(!data||data.length<2) return(
    <div style={{width:W,height:H,display:"flex",alignItems:"center",
      justifyContent:"center",color:C.muted,fontSize:11,fontStyle:"italic"}}>awaiting epochs…</div>
  );
  const xs=data.map(d=>d[xKey]),ys=data.map(d=>d[yKey]);
  const xMn=Math.min(...xs),xMx=Math.max(...xs);
  const yMn=Math.min(...ys)*0.985,yMx=Math.max(...ys)*1.01;
  const px=x=>((x-xMn)/(xMx-xMn||1))*(W-20)+10;
  const py=y=>H-8-((y-yMn)/(yMx-yMn||1))*(H-18);
  const pts=data.map(d=>`${px(d[xKey])},${py(d[yKey])}`).join(" ");
  const area=`M${px(xs[0])},${H} `+data.map(d=>`L${px(d[xKey])},${py(d[yKey])}`).join(" ")+` L${px(xs[xs.length-1])},${H} Z`;
  return(
    <svg width={W} height={H}>
      <defs>
        <linearGradient id={`sg${yKey}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.3"/>
          <stop offset="100%" stopColor={color} stopOpacity="0"/>
        </linearGradient>
      </defs>
      <path d={area} fill={`url(#sg${yKey})`}/>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      {data.map((d,i)=>(
        <circle key={i} cx={px(d[xKey])} cy={py(d[yKey])} r={i===data.length-1?5:3}
          fill={color} stroke={C.bg} strokeWidth="2"/>
      ))}
    </svg>
  );
}

function PlaceGrid(){
  const [hov,setHov]=useState(null);
  return(
    <div>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
        {Array.from({length:16},(_,k)=>(
          <div key={k} onMouseEnter={()=>setHov(k)} onMouseLeave={()=>setHov(null)}
            style={{aspectRatio:"1",background:C.grid,
              border:`1px solid ${hov===k?C.accent3:C.border}`,borderRadius:8,
              display:"flex",flexDirection:"column",alignItems:"center",
              justifyContent:"center",gap:3,cursor:"default",
              transition:"all 0.15s",position:"relative",overflow:"hidden"}}>
            <div style={{position:"absolute",inset:0,
              background:"repeating-linear-gradient(45deg,transparent,transparent 4px,#ffffff03 4px,#ffffff03 8px)"}}/>
            <div style={{fontSize:11,fontFamily:"monospace",color:C.muted,fontWeight:600,position:"relative"}}>
              p{String(k).padStart(2,"0")}
            </div>
            <div style={{fontSize:9,color:C.muted,position:"relative"}}>○ pending</div>
          </div>
        ))}
      </div>
      <div style={{marginTop:10,fontSize:10,color:C.muted}}>
        Run <code style={{color:C.accent1}}>eval_place_cell_receptive_fields.py</code> to populate
      </div>
    </div>
  );
}

export default function Dashboard(){
  const [tab,setTab]=useState("overview");
  const [pulse,setPulse]=useState(false);
  useEffect(()=>{const t=setInterval(()=>setPulse(p=>!p),1400);return()=>clearInterval(t);},[]);

  const tabs=["overview","dissociation","training","place fields","pi ablation"];

  return(
    <div style={{minHeight:"100vh",background:C.bg,color:C.text,
      fontFamily:"'DM Sans',system-ui,sans-serif",padding:"20px 24px"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-track{background:${C.bg}}
        ::-webkit-scrollbar-thumb{background:${C.border};border-radius:2px}
      `}</style>

      {/* Header */}
      <div style={{marginBottom:24}}>
        <div style={{display:"flex",alignItems:"flex-start",justifyContent:"space-between",marginBottom:4}}>
          <div>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
              <div style={{width:7,height:7,borderRadius:"50%",background:C.accent3,
                boxShadow:`0 0 ${pulse?14:6}px ${C.accent3}`,transition:"box-shadow 0.7s"}}/>
              <span style={{fontSize:10,color:C.accent3,fontFamily:"monospace",letterSpacing:"0.1em"}}>
                LIVE · SPRINT 6C TRAINING · ep1 acc=0.8102
              </span>
            </div>
            <h1 style={{fontSize:26,fontWeight:600,letterSpacing:"-0.02em"}}>NeMo-WM Research Dashboard</h1>
            <p style={{fontSize:12,color:C.muted,marginTop:2}}>Neuromodulated World Model · Computational Neuroscience Series</p>
          </div>
          <div style={{textAlign:"right",fontSize:10,color:C.muted,fontFamily:"monospace",lineHeight:1.7}}>
            <div>GMKtec EVO-X2 · AMD Ryzen AI MAX+ 395</div>
            <div>128GB RAM · NPU XINT8 · 10,995 RECON files</div>
            <div style={{color:C.accent2,marginTop:2}}>arXiv pending · cs.LG + cs.RO</div>
          </div>
        </div>

        <div style={{display:"flex",gap:2,borderBottom:`1px solid ${C.border}`,paddingBottom:0,marginTop:16}}>
          {tabs.map(t=>(
            <button key={t} onClick={()=>setTab(t)} style={{
              background:"none",border:"none",cursor:"pointer",
              padding:"7px 14px",fontSize:12,
              color:tab===t?C.accent1:C.muted,
              borderBottom:`2px solid ${tab===t?C.accent1:"transparent"}`,
              marginBottom:-1,transition:"all 0.15s",textTransform:"capitalize",
            }}>{t}</button>
          ))}
        </div>
      </div>

      {/* OVERVIEW */}
      {tab==="overview"&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:12,marginBottom:20}}>
            <StatCard label="Best RECON AUROC" value="0.9837" sub="Sprint 2, k=1" color={C.accent3}/>
            <StatCard label="VLM-Only Baseline" value="0.8624" sub="Hard neg, k≤4" color={C.accent1}/>
            <StatCard label="Sprint 6 Full" value="0.892" sub="VLM + trained proprio ✓" color={C.accent2}/>
            <StatCard label="Sprint 6c Epoch 1" value="0.8102" sub="top1_acc, training live" color={C.warn}/>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1.3fr 1fr",gap:16}}>
            <Section title="Sprint Progression" badge="AUROC" color={C.accent1}>
              {SPRINT_DATA.map(s=>(
                <AurocRow key={s.sprint}
                  label={`${s.sprint} — ${s.label}`}
                  value={s.auroc}
                  baseline={0.70}
                  pending={!s.auroc}
                  hl={s.sprint==="S6c"}/>
              ))}
              <div style={{marginTop:12,fontSize:10,color:C.muted}}>
                <span style={{color:C.warn}}>│</span> Sprint 2 threshold 0.70 &nbsp;·&nbsp; VLM-only baseline 0.862
              </div>
            </Section>
            <Section title="Temporal k-sweep" badge="k=1..16" color={C.accent2}>
              <KSweep/>
              <div style={{marginTop:12}}>
                {K_SWEEP.map(d=>(
                  <div key={d.k} style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
                    <span style={{width:28,fontSize:11,color:C.muted,fontFamily:"monospace"}}>k={d.k}</span>
                    <Bar v={d.a} color={C.accent2}/>
                    <span style={{width:40,fontSize:11,textAlign:"right",fontFamily:"monospace",
                      color:d.a>=0.70?C.accent3:C.danger}}>{d.a}</span>
                  </div>
                ))}
              </div>
            </Section>
          </div>
        </div>
      )}

      {/* DISSOCIATION */}
      {tab==="dissociation"&&(
        <div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:12,marginBottom:16}}>
            <StatCard label="Aphasia gap S4→S6" value="0.423→0.042" sub="VLM ablation delta" color={C.accent3}/>
            <StatCard label="Sprint 6 Full AUROC" value="0.892" sub="Beats VLM-only 0.862 ✓" color={C.accent2}/>
            <StatCard label="S6c dissociation target" value="≥ 0.862" sub="No-VLM must match VLM" color={C.warn}/>
          </div>
          <Section title="Three-Condition Dissociation Eval" badge="hard neg · no GPS" color={C.accent3}>
            <div style={{display:"grid",gridTemplateColumns:"200px 1fr 1fr 1fr",gap:0,marginBottom:8}}>
              {["Condition","Random baseline","Sprint 6 trained","Sprint 6c (pending)"].map(h=>(
                <div key={h} style={{fontSize:9,color:C.muted,padding:"5px 6px",
                  letterSpacing:"0.08em",textTransform:"uppercase",
                  borderBottom:`1px solid ${C.border}`}}>{h}</div>
              ))}
              {DISSOCIATION.map(row=>[
                <div key={`${row.cond}l`} style={{fontSize:11,color:C.text,padding:"9px 6px",
                  borderBottom:`1px solid ${C.border}22`,fontFamily:"monospace"}}>{row.cond}</div>,
                <div key={`${row.cond}r`} style={{padding:"9px 6px",borderBottom:`1px solid ${C.border}22`,
                  display:"flex",alignItems:"center",gap:6}}>
                  <Bar v={row.s6r} color={C.muted}/><span style={{fontSize:11,fontFamily:"monospace",
                  color:C.muted,width:42}}>{row.s6r?.toFixed(3)}</span>
                </div>,
                <div key={`${row.cond}t`} style={{padding:"9px 6px",borderBottom:`1px solid ${C.border}22`,
                  display:"flex",alignItems:"center",gap:6}}>
                  <Bar v={row.s6t} color={row.s6t>(row.vlm??0)?C.accent3:C.accent1}/>
                  <span style={{fontSize:11,fontFamily:"monospace",width:42,
                    color:row.s6t>(row.vlm??0)?C.accent3:C.accent1}}>{row.s6t?.toFixed(3)}</span>
                </div>,
                <div key={`${row.cond}c`} style={{padding:"9px 6px",borderBottom:`1px solid ${C.border}22`,
                  display:"flex",alignItems:"center",gap:6}}>
                  <Bar v={row.s6c??0} color={C.accent2} pending={!row.s6c}/>
                  <span style={{fontSize:11,fontFamily:"monospace",color:C.muted,width:42}}>
                    {row.s6c?.toFixed(3)??"—"}
                  </span>
                </div>,
              ])}
            </div>
            <div style={{padding:12,background:`${C.accent3}10`,borderRadius:8,
              border:`1px solid ${C.accent3}33`,fontSize:12,color:C.accent3,lineHeight:1.6}}>
              <strong>S6 PASSED:</strong> Full (0.892) &gt; VLM-only (0.862). Aphasia gap: 0.423 → 0.042.
              Path integration parallel: velocity+heading encodes navigational quasimetric without visual input.
            </div>
          </Section>
        </div>
      )}

      {/* TRAINING */}
      {tab==="training"&&(
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
          <Section title="Sprint 6c — top1_acc curve" badge="LIVE" color={C.accent3}>
            <SparkLine data={TRAIN_6C} xKey="ep" yKey="acc" color={C.accent3} W={280} H={80}/>
            <div style={{marginTop:12}}>
              {TRAIN_6C.map(d=>(
                <div key={d.ep} style={{display:"flex",gap:10,fontSize:11,color:C.muted,
                  fontFamily:"monospace",marginBottom:4,padding:"4px 0",
                  borderBottom:`1px solid ${C.border}22`}}>
                  <span style={{width:40}}>ep {d.ep}</span>
                  <span style={{color:C.accent3,width:90}}>acc={d.acc.toFixed(4)}</span>
                  <span style={{color:C.muted}}>loss={d.loss.toFixed(4)}</span>
                </div>
              ))}
              <div style={{marginTop:8,fontSize:11,color:C.muted,fontFamily:"monospace"}}>
                → awaiting epochs 2–40…
              </div>
            </div>
          </Section>
          <Section title="Sprint comparison" color={C.accent2}>
            {[
              {label:"S6  (9K · single frame)",ep0:0.703,best:0.718,color:C.muted},
              {label:"S6b (26K · +delta_h)",   ep0:0.703,best:0.717,color:C.muted},
              {label:"S6c (26K · 4-frame win)",ep0:0.799,best:0.810,color:C.accent3},
            ].map(s=>(
              <div key={s.label} style={{marginBottom:16}}>
                <div style={{fontSize:11,color:s.color,marginBottom:6,fontFamily:"monospace"}}>{s.label}</div>
                <div style={{display:"flex",gap:8,alignItems:"center",marginBottom:4}}>
                  <span style={{fontSize:10,color:C.muted,width:30}}>ep0</span>
                  <Bar v={s.ep0} color={s.color}/><span style={{fontSize:11,fontFamily:"monospace",
                  color:s.color,width:40}}>{s.ep0}</span>
                </div>
                <div style={{display:"flex",gap:8,alignItems:"center"}}>
                  <span style={{fontSize:10,color:C.muted,width:30}}>best</span>
                  <Bar v={s.best} color={s.color}/><span style={{fontSize:11,fontFamily:"monospace",
                  color:s.color,width:40}}>{s.best}</span>
                </div>
              </div>
            ))}
            <div style={{padding:12,background:`${C.accent3}10`,borderRadius:8,
              border:`1px solid ${C.accent3}33`,fontSize:11,color:C.accent3,marginTop:4}}>
              6c epoch 0 acc=0.799 vs S6/S6b ceiling 0.718 (+0.081 from temporal context alone)
            </div>
          </Section>
        </div>
      )}

      {/* PLACE FIELDS */}
      {tab==="place fields"&&(
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:16}}>
          <Section title="K=16 Particle Place Fields" badge="pending eval" color={C.accent3}>
            <p style={{fontSize:11,color:C.muted,marginBottom:14,lineHeight:1.6}}>
              Biological parallel: hippocampal place cells fire in localised regions of space.
              Each of the 16 particles should show a receptive field in GPS space.
              Colour = spatial tuning index. Run eval_place_cell_receptive_fields.py to populate.
            </p>
            <PlaceGrid/>
          </Section>
          <Section title="Grid Cell Test: AUROC vs GPS Distance" badge="pending" color={C.accent1}>
            <p style={{fontSize:11,color:C.muted,marginBottom:14,lineHeight:1.6}}>
              Prediction: if particles encode metric space (not temporal order), AUROC
              decays monotonically with GPS distance — grid cell signature.
            </p>
            {[["&lt; 2m (adjacent)",null],["2–5m (nearby)",null],["5–15m (close)",null],
              ["15–50m (medium)",null],["50–200m (far)",null],["&gt; 200m (very far)",null]
            ].map(([lbl,val])=>(
              <div key={lbl} style={{display:"flex",alignItems:"center",gap:10,marginBottom:7}}>
                <span style={{width:80,fontSize:11,color:C.muted,fontFamily:"monospace"}}
                  dangerouslySetInnerHTML={{__html:lbl}}/>
                <Bar v={val??0} color={C.accent1} pending={!val}/>
                <span style={{width:48,fontSize:11,fontFamily:"monospace",color:C.muted}}>
                  {val?.toFixed(3)??"—"}
                </span>
              </div>
            ))}
            <div style={{marginTop:10,fontSize:10,color:C.muted}}>
              Run with <code style={{color:C.accent1}}>--grid-test --no-place</code>
            </div>
          </Section>
        </div>
      )}

      {/* PI ABLATION */}
      {tab==="pi ablation"&&(
        <Section title="Path Integration Channel Ablation" badge="McNaughton · Moser" color={C.warn}>
          <p style={{fontSize:11,color:C.muted,marginBottom:14,lineHeight:1.6}}>
            Mirrors biological lesion studies. Zero each signal channel in the 6c temporal encoder
            and measure AUROC drop. Channels dropping &gt;0.05 are computationally load-bearing —
            analogous to velocity afferents (McNaughton 2006) and head direction cells (Moser 2008).
          </p>
          {ABLATION.map(c=>(
            <AurocRow key={c.label} label={c.label} value={c.auroc}
              baseline={c.full?null:0.892} pending={!c.auroc&&!c.full} hl={c.full}/>
          ))}
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr",gap:10,marginTop:16}}>
            {[
              {label:"Velocity lesion",desc:"McNaughton 2006 — muscle spindle afferents",color:C.accent3},
              {label:"HD lesion",desc:"Moser 2008 — anterior thalamus / entorhinal",color:C.accent1},
              {label:"Double dissociation",desc:"Gold standard — both load-bearing independently",color:C.warn},
            ].map(item=>(
              <div key={item.label} style={{padding:12,background:`${item.color}10`,
                borderRadius:8,border:`1px solid ${item.color}33`}}>
                <div style={{fontSize:11,color:item.color,fontWeight:600}}>{item.label}</div>
                <div style={{fontSize:9,color:C.muted,marginTop:2,lineHeight:1.5}}>{item.desc}</div>
                <div style={{fontSize:22,color:item.color,marginTop:8,fontFamily:"monospace"}}>—</div>
              </div>
            ))}
          </div>
          <div style={{marginTop:14,padding:12,background:`${C.warn}10`,borderRadius:8,
            border:`1px solid ${C.warn}33`,fontSize:11,color:C.warn,lineHeight:1.6}}>
            Run eval_path_integration_ablation.py after Sprint 6c completes.
            Paste AUROC values into ABLATION constant to populate this table.
          </div>
        </Section>
      )}

      <div style={{marginTop:28,paddingTop:14,borderTop:`1px solid ${C.border}`,
        display:"flex",justifyContent:"space-between",fontSize:10,
        color:C.muted,fontFamily:"monospace"}}>
        <span>NeMo-WM · Computational Neuroscience Series · github.com/taylorjohn</span>
        <span>Update data constants at top of file to populate charts with eval results</span>
      </div>
    </div>
  );
}
