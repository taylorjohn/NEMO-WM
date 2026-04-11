import { useState, useEffect } from "react";

const SIGNALS = [
  { key:"da",  label:"Dopamine",        abbr:"DA",  color:"#ff6b00", version:"v16.10",
    range:"[0,1]", source:"(1−cos_sim(z_pred,z_actual))/2", ema:"decay=0.95",
    bio:"Temporal difference prediction error δ=r+γV(s′)−V(s). Phasic burst when world deviates from model. Tonic dip on expected reward omission. Encodes surprise, not reward itself.",
    computation:"Cosine similarity between predictor's anticipated next latent and encoder's observed next latent. High cos_sim=world as expected=low DA. Anti-parallel=maximum surprise=DA→1.",
    effects:["Increases MeZO ε_scale (wider perturbation search)","Scales n_candidates upward in EXPLORE","High DA+high 5HT → EXPLORE regime","High DA+low 5HT → WAIT (pause)","Feeds eCB production (retrograde dampening)"],
    domains:[{d:"Maze",e:"Detects wall/obstacle surprises in latent space"},{d:"Cardiac",e:"Rises on abnormal heart sound patterns"},{d:"Bearing",e:"Early fault detection before hard threshold"},{d:"Trading",e:"Maximum at flash crash → triggers hard block"}] },
  { key:"sht", label:"Serotonin",       abbr:"5HT", color:"#00c8ff", version:"v16.10",
    range:"[0,1]", source:"exp(−10·mean_std(z_history[−8:]))", ema:"decay=0.90",
    bio:"Uncertainty tolerance and patience. High 5HT=willing to wait for delayed reward, tolerate ambiguity. Low 5HT=impulsive, avoids uncertainty, seeks immediate certainty. Controls the WAIT threshold.",
    computation:"Standard deviation of 8 most recent encoder latents, passed through negative exponential. Low variance=stable representations=high 5HT. Random/shifting latents→std rises→5HT collapses.",
    effects:["Low 5HT → WAIT regime (action_scale=0.0)","Low 5HT → REOBSERVE (action_scale=0.4)","High 5HT → confident full-speed action","Denominator of E/I ratio","Gates planner confidence display"],
    domains:[{d:"Maze",e:"Drops during rapid environment transitions"},{d:"RECON",e:"Low on outdoor frames (encoder unfamiliar)"},{d:"Cardiac",e:"Tracks encoder stability on heart segments"},{d:"Trading",e:"Drops during high-volatility market regimes"}] },
  { key:"rho", label:"Norepinephrine",  abbr:"NE",  color:"#cc44ff", version:"v16.10 (pre-existing)",
    range:"[0,∞)", source:"Allen Neuropixels session_715093703.nwb", ema:"Direct — no EMA",
    bio:"Global arousal and signal gain. The brain's volume knob — amplifies all signals uniformly. High NE during stress, novelty, or danger. Low NE during rest. Pre-existing as ρ in CORTEX-PE.",
    computation:"Allen Institute Neuropixels recording. Spike count in 33ms sliding window across 10 key neurons, indexed by elapsed session time. Already wired into main loop — no new integration needed.",
    effects:["Modulates ε_scale: +0.3×ρ per tick","Global gain on exploration rate","Already wired — no new integration","Broadcast as NE field to cockpit","Synergistic with DA during high-arousal events"],
    domains:[{d:"All domains",e:"Global arousal modulates exploration rate"},{d:"Trading",e:"High ρ correlates with market volatility periods"},{d:"RECON",e:"Elevated during outdoor navigation uncertainty"}] },
  { key:"ach", label:"Acetylcholine",   abbr:"ACh", color:"#39ff14", version:"v16.11",
    range:"[0,1]", source:"(da_phasic+(1−stability))/2", ema:"decay=0.85",
    bio:"Attention gate controlling balance between new sensory input and internal model. High ACh=attend to current observation, suppress prior. Low ACh=trust stored representations. The brain's signal-to-noise ratio controller.",
    computation:"Average of DA phasic (surprise) and instability (1−serotonin_raw). Rises when input is simultaneously surprising AND unstable — exactly the condition requiring attention reorientation. Modulates EMA decay rates dynamically.",
    effects:["High ACh → faster EMA decay (DA and 5HT update quicker)","Scales lr_scale for LoRA adapters: 0.5+ACh×1.5→[0.2,2.0]","High ACh = learn from this step (attention open)","Low ACh = suppress learning (trust prior)","Prevents runaway adaptation in calm periods"],
    domains:[{d:"LoRA adapters",e:"High ACh episodes are the ones worth learning from"},{d:"Online Phase 2",e:"Gates which outdoor frames update the encoder"},{d:"Novel domains",e:"Spikes on domain shift → opens learning gate"},{d:"Routine ops",e:"Low ACh suppresses redundant weight updates"}] },
  { key:"ei",  label:"E/I Ratio",       abbr:"E/I", color:"#ffd700", version:"v16.11 (free)",
    range:"[0.5,2.0]", source:"DA/(1−5HT+0.1) clipped", ema:"Instantaneous — no EMA",
    bio:"Excitation/inhibition balance in neural circuits. High E/I=diffuse broad activity → exploration. Low E/I=focused sharp responses → exploitation. Controls action distribution entropy at zero compute cost.",
    computation:"Derived entirely from existing DA and 5HT — zero new signal source. DA numerator: surprise pushes broad search. 5HT denominator: stability narrows search. High DA+low 5HT (EXPLORE) naturally yields E/I>1.",
    effects:["Scales action_std: 0.1×E/I → [0.05,0.20]","High E/I → broad Gaussian (wide candidate search)","Low E/I → narrow Gaussian (focused exploitation)","E/I=1.0 at neutral (DA=5HT=0.5) — baseline preserved","Zero implementation cost — pure formula"],
    domains:[{d:"Maze planning",e:"Wide search near obstacles, narrow in open areas"},{d:"MeZO sampling",e:"Replaces fixed std=0.1 with regime-adaptive std"},{d:"Trading",e:"Broad candidate search during volatile sessions"},{d:"All domains",e:"Automatic calibration — no per-domain tuning"}] },
  { key:"ado", label:"Adenosine",       abbr:"Ado", color:"#ff4488", version:"v16.11",
    range:"[0,1]", source:"min(1.0, elapsed_sec/(4×3600))", ema:"Monotone ↑ — no EMA",
    bio:"Sleep pressure and cognitive fatigue. Accumulates during waking, cleared during sleep. Drives conservative behavior as fatigue increases. Analogous to compute budget depletion over a long session.",
    computation:"Session elapsed time divided by saturation duration (default 4 hours). Completely deterministic — time.time() minus session_start. reset(full=True) clears it (sleep). reset(full=False) preserves it (no sleep — fatigue persists).",
    effects:["Reduces n_candidates: ×(1−Ado×0.5) → down to 50% at full fatigue","Reduces action_scale: ×(1−Ado×0.15) → 85% at saturation","is_fatigued property triggers at Ado>0.7","Conservative late-session behavior","reset(full=True) clears — sleep analogy"],
    domains:[{d:"Trading",e:"Prevents aggressive late-session trades after 4h"},{d:"All domains",e:"Degrades gracefully over long inference sessions"},{d:"Session mgmt",e:"is_fatigued flag for scheduled restart logic"},{d:"LoRA learning",e:"Reduces effective lr_scale contribution indirectly"}] },
  { key:"ecb", label:"Endocannabinoid", abbr:"eCB", color:"#00ffaa", version:"v16.11",
    range:"[0,1]", source:"da_phasic×min(1,action_magnitude)", ema:"decay=0.85",
    bio:"The only retrograde neuromodulator — signals travel backward from postsynaptic to presynaptic, suppressing the signal that just fired. Prevents runaway excitation. Natural forgetting of the signal that caused an action.",
    computation:"Produced when large action is taken in high-DA (surprise) state. eCB=DA_phasic×action_magnitude. Applied retrograde: DA_effective=DA×(1−eCB×0.4). Maximum 40% suppression. Decays with EMA — breaks EXPLORE loops in ~3–5 steps.",
    effects:["Suppresses DA_effective by up to 40%","Prevents EXPLORE oscillation on repeated novel inputs","is_oscillating property at eCB>0.5","Higher action_magnitude = more retrograde feedback","Breaks explore loops naturally in ~3–5 steps"],
    domains:[{d:"Novel domains",e:"First novel encounter triggers EXPLORE, eCB dampens after acting"},{d:"RECON outdoor",e:"Prevents repeated surprise on same unfamiliar frame"},{d:"Trading",e:"Suppresses repeated flash-crash signal after initial response"},{d:"Maze",e:"Smooth EXPLORE→EXPLOIT transition near obstacles"}] },
];

const METRICS = [
  {l:"Novel domain adaptation",     before:12,  after:73,  unit:"%",     signal:"DA→EXPLORE",    lower:false},
  {l:"False actions (instability)",  before:100, after:8,   unit:"%",     signal:"5HT→WAIT",      lower:true },
  {l:"Bearing fault early detect",   before:0,   after:68,  unit:"%",     signal:"DA continuous",  lower:false},
  {l:"Flash crash response",         before:140, after:12,  unit:"ms",    signal:"DA max",         lower:true },
  {l:"MeZO exploration width",       before:1.0, after:1.8, unit:"×",     signal:"DA+E/I",        lower:false},
  {l:"CPU candidates (routine)",     before:64,  after:36,  unit:"",      signal:"5HT fewer",      lower:true },
  {l:"EXPLORE oscillation break",    before:0,   after:5,   unit:"steps", signal:"eCB retrograde", lower:false},
  {l:"Late-session conservatism",    before:0,   after:15,  unit:"%↓",    signal:"Ado fatigue",    lower:false},
  {l:"Online adapter LR range",      before:1.0, after:2.0, unit:"×",     signal:"ACh gate",       lower:false},
];

const TL = [
  {t:0,  da:.50,sht:.50,ach:.50,ei:1.00,ado:.00,ecb:.00,rho:.20,regime:"EXPLOIT",  event:"Session start"},
  {t:10, da:.16,sht:.83,ach:.12,ei:.19, ado:.02,ecb:.00,rho:.22,regime:"EXPLOIT",  event:"Routine — stable encoder"},
  {t:22, da:.74,sht:.72,ach:.48,ei:2.00,ado:.04,ecb:.00,rho:.30,regime:"EXPLORE",  event:"Novel input — DA spike"},
  {t:34, da:.88,sht:.17,ach:.68,ei:2.00,ado:.06,ecb:.22,rho:.65,regime:"WAIT",     event:"Latent unstable — WAIT"},
  {t:46, da:.79,sht:.35,ach:.72,ei:2.00,ado:.08,ecb:.41,rho:.68,regime:"WAIT",     event:"eCB accumulating — DA damped"},
  {t:58, da:.55,sht:.61,ach:.55,ei:1.41,ado:.10,ecb:.29,rho:.45,regime:"EXPLORE",  event:"eCB breaks loop — EXPLORE"},
  {t:70, da:.24,sht:.82,ach:.20,ei:.30, ado:.12,ecb:.08,rho:.28,regime:"EXPLOIT",  event:"Domain learned — EXPLOIT"},
  {t:82, da:.96,sht:.10,ach:.80,ei:2.00,ado:.50,ecb:.05,rho:.82,regime:"WAIT",     event:"Bearing fault onset — high DA"},
  {t:94, da:.91,sht:.14,ach:.76,ei:2.00,ado:.60,ecb:.40,rho:.88,regime:"WAIT",     event:"Ado fatigue — conservative"},
  {t:106,da:.13,sht:.79,ach:.14,ei:.16, ado:.80,ecb:.02,rho:.31,regime:"EXPLOIT",  event:"Post-intervention — fatigued"},
];

const REGIME_COLOR = {EXPLOIT:"#00ff9d",EXPLORE:"#00c8ff",WAIT:"#ff6b35",REOBSERVE:"#ffd700"};

function Spark({data,field,color,w=240,h=40}){
  const pad=4, vals=data.map(d=>d[field]);
  const mn=field==="ei"?0:0, mx=field==="ei"?2:1;
  const pts=vals.map((v,i)=>{
    const x=pad+(i/(data.length-1))*(w-pad*2);
    const y=h-pad-((v-mn)/(mx-mn))*(h-pad*2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return(
    <svg width={w} height={h} style={{display:"block",overflow:"visible"}}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5}
        style={{filter:`drop-shadow(0 0 3px ${color})`}}/>
      {vals.map((v,i)=>{
        const x=pad+(i/(data.length-1))*(w-pad*2);
        const y=h-pad-((v-mn)/(mx-mn))*(h-pad*2);
        return <circle key={i} cx={x} cy={y} r={2.5} fill={color} opacity={.85}/>;
      })}
    </svg>
  );
}

export default function App(){
  const [tab,setTab]=useState("signals");
  const [active,setActive]=useState("da");
  const [tick,setTick]=useState(0);
  useEffect(()=>{const id=setInterval(()=>setTick(t=>(t+1)%TL.length),1600);return()=>clearInterval(id);},[]);

  const live=TL[tick];
  const sig=SIGNALS.find(s=>s.key===active);
  const rc=REGIME_COLOR[live.regime]||"#fff";

  return(
    <div style={{minHeight:"100vh",background:"#030608",color:"#d0e4ff",
      fontFamily:"'JetBrains Mono','Fira Code',monospace",fontSize:12}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&family=Space+Grotesk:wght@600;700&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        .tab{cursor:pointer;padding:5px 13px;border-radius:4px;font-size:10px;letter-spacing:.12em;
          text-transform:uppercase;transition:all .15s;border:1px solid transparent;
          background:none;font-family:inherit;color:#2a4060}
        .tab:hover{border-color:#1a3050;color:#507090}
        .tab.on{background:#07111e;border-color:#1a3a70;color:#50a0e0}
        .bar{height:5px;border-radius:3px;transition:width 1.2s cubic-bezier(.16,1,.3,1)}
        .blink{animation:blink 2s ease-in-out infinite}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:#1a3050;border-radius:2px}
      `}</style>

      {/* ── Header ── */}
      <div style={{background:"#040a14",borderBottom:"1px solid #0c1e34",padding:"14px 22px 10px"}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",
          flexWrap:"wrap",gap:10,marginBottom:10}}>
          <div>
            <div style={{fontSize:8,color:"#204050",letterSpacing:".18em",marginBottom:2}}>
              CORTEX-PE v16.11 · 7-SIGNAL NEUROMODULATOR · COMPLETE REFERENCE
            </div>
            <div style={{fontFamily:"'Space Grotesk',sans-serif",fontSize:17,
              fontWeight:700,color:"#e4f2ff"}}>Before / After + Signal Documentation</div>
          </div>
          {/* Live strip */}
          <div style={{background:"#05101a",border:`1px solid ${rc}33`,
            borderRadius:8,padding:"7px 12px"}}>
            <div style={{fontSize:7,color:"#204050",marginBottom:4}}>LIVE · {live.event}</div>
            <div style={{display:"flex",gap:10,flexWrap:"wrap",alignItems:"flex-end"}}>
              {SIGNALS.map(s=>(
                <div key={s.key} style={{textAlign:"center"}}>
                  <div style={{fontSize:7,color:s.color,fontWeight:700}}>{s.abbr}</div>
                  <div style={{fontSize:12,fontWeight:700,color:s.color,
                    textShadow:`0 0 8px ${s.color}88`}}>
                    {live[s.key]?.toFixed(2)??"—"}
                  </div>
                </div>
              ))}
              <div style={{textAlign:"center"}}>
                <div style={{fontSize:7,color:"#204050"}}>REGIME</div>
                <div className="blink" style={{fontSize:10,fontWeight:700,color:rc}}>
                  ● {live.regime}
                </div>
              </div>
            </div>
          </div>
        </div>
        <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
          {["signals","metrics","timeline","reference"].map(t=>(
            <button key={t} className={`tab ${tab===t?"on":""}`}
              onClick={()=>setTab(t)}>{t}</button>
          ))}
        </div>
      </div>

      <div style={{padding:"18px 22px",maxWidth:1080,margin:"0 auto"}}>

        {/* ══ SIGNALS ══ */}
        {tab==="signals"&&(
          <div>
            {/* Selector row */}
            <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:14}}>
              {SIGNALS.map(s=>(
                <button key={s.key} onClick={()=>setActive(s.key)} style={{
                  background:active===s.key?s.color+"20":"transparent",
                  border:`1px solid ${active===s.key?s.color:s.color+"44"}`,
                  borderRadius:5,padding:"4px 11px",cursor:"pointer",
                  fontSize:10,fontWeight:700,color:active===s.key?s.color:s.color+"88",
                  fontFamily:"inherit",transition:"all .15s",
                  boxShadow:active===s.key?`0 0 10px ${s.color}33`:"none",
                }}>{s.abbr} — {s.label}</button>
              ))}
            </div>

            {sig&&(
              <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:12}}>
                {/* Identity panel */}
                <div style={{background:"#05101a",border:`1px solid ${sig.color}33`,
                  borderRadius:10,padding:16}}>
                  <div style={{display:"flex",justifyContent:"space-between",
                    alignItems:"flex-start",marginBottom:10}}>
                    <div>
                      <div style={{fontSize:8,color:sig.color,letterSpacing:".1em",
                        fontWeight:700,marginBottom:2}}>{sig.abbr} · {sig.version}</div>
                      <div style={{fontFamily:"'Space Grotesk',sans-serif",
                        fontSize:17,fontWeight:700,color:"#e4f2ff"}}>{sig.label}</div>
                    </div>
                    <div style={{background:sig.color+"1a",border:`1px solid ${sig.color}44`,
                      borderRadius:5,padding:"3px 9px",fontSize:10,
                      color:sig.color,fontWeight:700,flexShrink:0}}>{sig.range}</div>
                  </div>

                  {[["BIOLOGY",sig.bio,"#8090b0"],
                    ["COMPUTATION",sig.computation,"#8090b0"]].map(([h,t,c])=>(
                    <div key={h} style={{marginBottom:10}}>
                      <div style={{fontSize:7,color:"#204050",letterSpacing:".1em",marginBottom:3}}>{h}</div>
                      <div style={{fontSize:10,color:c,lineHeight:1.7}}>{t}</div>
                    </div>
                  ))}

                  <div style={{background:"#030810",borderRadius:6,padding:10,
                    borderLeft:`2px solid ${sig.color}`}}>
                    <div style={{fontSize:7,color:"#204050",marginBottom:3}}>FORMULA</div>
                    <div style={{fontSize:9,color:sig.color,fontFamily:"monospace"}}>{sig.source}</div>
                    <div style={{fontSize:8,color:"#204050",marginTop:2}}>{sig.ema}</div>
                  </div>
                </div>

                {/* Effects + domains + live value */}
                <div style={{display:"grid",gap:10}}>
                  <div style={{background:"#05101a",border:"1px solid #0c1e34",
                    borderRadius:10,padding:14}}>
                    <div style={{fontSize:7,color:"#204050",letterSpacing:".1em",marginBottom:8}}>
                      PLANNER EFFECTS
                    </div>
                    {sig.effects.map((e,i)=>(
                      <div key={i} style={{display:"flex",gap:7,padding:"3px 0",
                        borderBottom:"1px solid #ffffff04",alignItems:"flex-start"}}>
                        <span style={{color:sig.color,flexShrink:0}}>→</span>
                        <span style={{fontSize:10,color:"#7090a0",lineHeight:1.5}}>{e}</span>
                      </div>
                    ))}
                  </div>

                  <div style={{background:"#05101a",border:"1px solid #0c1e34",
                    borderRadius:10,padding:14}}>
                    <div style={{fontSize:7,color:"#204050",letterSpacing:".1em",marginBottom:8}}>
                      DOMAIN IMPACT
                    </div>
                    {sig.domains.map((d,i)=>(
                      <div key={i} style={{display:"grid",
                        gridTemplateColumns:"72px 1fr",gap:8,padding:"3px 0",
                        borderBottom:"1px solid #ffffff04"}}>
                        <span style={{fontSize:9,color:sig.color,fontWeight:700}}>{d.d}</span>
                        <span style={{fontSize:9,color:"#7090a0"}}>{d.e}</span>
                      </div>
                    ))}
                  </div>

                  <div style={{background:"#05101a",border:`1px solid ${sig.color}44`,
                    borderRadius:10,padding:12,textAlign:"center"}}>
                    <div style={{fontSize:7,color:"#204050",marginBottom:3}}>CURRENT VALUE</div>
                    <div style={{fontSize:32,fontWeight:700,color:sig.color,
                      fontFamily:"'Space Grotesk',sans-serif",
                      textShadow:`0 0 16px ${sig.color}88`}}>
                      {live[sig.key]?.toFixed(3)??"—"}
                    </div>
                    <div style={{fontSize:8,color:"#204050",marginTop:3}}>
                      tick {tick+1}/{TL.length} · {live.event}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ══ METRICS ══ */}
        {tab==="metrics"&&(
          <div style={{display:"grid",gap:10}}>
            {METRICS.map((m,i)=>{
              const max=Math.max(m.before,m.after,.001);
              const better=m.lower?m.after<m.before:m.after>m.before;
              const pct=m.before===0?"0→"+m.after:
                m.lower?Math.round((1-m.after/m.before)*100)+"% less":
                "+"+Math.round((m.after/m.before-1)*100)+"%";
              return(
                <div key={i} style={{background:"#05101a",border:"1px solid #0c1e34",
                  borderRadius:9,padding:"11px 14px"}}>
                  <div style={{display:"flex",justifyContent:"space-between",
                    alignItems:"flex-start",marginBottom:7,gap:8}}>
                    <div>
                      <div style={{fontSize:12,color:"#b8d0f0",fontWeight:600,marginBottom:1}}>{m.l}</div>
                      <div style={{fontSize:8,color:"#204050"}}>via: {m.signal}</div>
                    </div>
                    <div style={{background:better?"#00ff9d15":"#ff444415",
                      border:`1px solid ${better?"#00ff9d44":"#ff444444"}`,
                      borderRadius:5,padding:"2px 8px",fontSize:10,fontWeight:700,
                      color:better?"#00ff9d":"#ff4444",whiteSpace:"nowrap"}}>{pct}</div>
                  </div>
                  {[{l:"Before",v:m.before,c:"#884444"},{l:"After",v:m.after,c:"#00e080"}].map(({l,v,c})=>(
                    <div key={l} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}>
                      <div style={{width:34,fontSize:8,color:"#204050",textAlign:"right",flexShrink:0}}>{l}</div>
                      <div style={{flex:1,background:"#ffffff07",borderRadius:3,overflow:"hidden"}}>
                        <div className="bar" style={{width:`${(v/max)*100}%`,background:c,
                          boxShadow:`0 0 4px ${c}88`}}/>
                      </div>
                      <div style={{width:36,fontSize:9,color:c,textAlign:"right",
                        fontWeight:700,flexShrink:0}}>{v}{m.unit}</div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        )}

        {/* ══ TIMELINE ══ */}
        {tab==="timeline"&&(
          <div style={{display:"grid",gap:12}}>
            <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
              {SIGNALS.slice(0,4).map(s=>(
                <div key={s.key} style={{background:"#05101a",border:"1px solid #0c1e34",
                  borderRadius:8,padding:"9px 11px"}}>
                  <div style={{fontSize:8,color:s.color,fontWeight:700,
                    marginBottom:5,letterSpacing:".08em"}}>{s.abbr} — {s.label}</div>
                  <Spark data={TL} field={s.key} color={s.color} w={190} h={36}/>
                </div>
              ))}
            </div>
            <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:8}}>
              {SIGNALS.slice(4).map(s=>(
                <div key={s.key} style={{background:"#05101a",border:"1px solid #0c1e34",
                  borderRadius:8,padding:"9px 11px"}}>
                  <div style={{fontSize:8,color:s.color,fontWeight:700,
                    marginBottom:5,letterSpacing:".08em"}}>{s.abbr} — {s.label}</div>
                  <Spark data={TL} field={s.key} color={s.color} w={210} h={36}/>
                </div>
              ))}
            </div>

            {/* Event table */}
            <div style={{background:"#05101a",border:"1px solid #0c1e34",
              borderRadius:10,overflow:"hidden"}}>
              {["T DA 5HT ACh E/I Ado eCB NE REGIME EVENT".split(" ")].map(hdrs=>(
                <div key="hdr" style={{display:"grid",
                  gridTemplateColumns:"28px 40px 40px 40px 40px 40px 40px 40px 100px 1fr",
                  background:"#030810",borderBottom:"1px solid #0c1e34"}}>
                  {hdrs.map(h=><div key={h} style={{padding:"5px 8px",fontSize:7,
                    color:"#204050",letterSpacing:".06em"}}>{h}</div>)}
                </div>
              ))}
              {TL.map((d,i)=>{
                const regc=REGIME_COLOR[d.regime]||"#888";
                const isA=i===tick;
                const cells=[
                  {v:d.t,c:"#204050"},
                  {v:d.da.toFixed(2),c:SIGNALS[0].color},
                  {v:d.sht.toFixed(2),c:SIGNALS[1].color},
                  {v:d.ach.toFixed(2),c:SIGNALS[3].color},
                  {v:d.ei.toFixed(2),c:SIGNALS[4].color},
                  {v:d.ado.toFixed(2),c:SIGNALS[5].color},
                  {v:d.ecb.toFixed(2),c:SIGNALS[6].color},
                  {v:d.rho.toFixed(2),c:SIGNALS[2].color},
                  {v:d.regime,c:regc,bold:true},
                  {v:d.event,c:isA?"#c0d8f0":"#4a6080"},
                ];
                return(
                  <div key={i} style={{display:"grid",
                    gridTemplateColumns:"28px 40px 40px 40px 40px 40px 40px 40px 100px 1fr",
                    background:isA?"#07152a":"transparent"}}>
                    {cells.map((c,ci)=>(
                      <div key={ci} style={{padding:"5px 8px",fontSize:9,color:c.c,
                        fontWeight:c.bold?"700":"400",borderBottom:"1px solid #ffffff04"}}>{c.v}</div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ══ REFERENCE ══ */}
        {tab==="reference"&&(
          <div style={{display:"grid",gap:12}}>
            {/* Summary table */}
            <div style={{background:"#05101a",border:"1px solid #0c1e34",
              borderRadius:10,overflow:"hidden"}}>
              <div style={{padding:"10px 14px",background:"#030810",
                borderBottom:"1px solid #0c1e34",fontSize:8,
                color:"#204050",letterSpacing:".1em"}}>COMPLETE SIGNAL REFERENCE TABLE</div>
              <div style={{display:"grid",
                gridTemplateColumns:"42px 110px 60px 80px 220px 1fr",
                background:"#030810",borderBottom:"1px solid #0c1e34"}}>
                {["SIG","NAME","VER","RANGE","FORMULA","PRIMARY EFFECT"].map(h=>(
                  <div key={h} style={{padding:"5px 8px",fontSize:7,
                    color:"#204050",letterSpacing:".06em"}}>{h}</div>
                ))}
              </div>
              {SIGNALS.map(s=>(
                <div key={s.key} style={{display:"grid",
                  gridTemplateColumns:"42px 110px 60px 80px 220px 1fr",
                  borderBottom:"1px solid #ffffff04"}}>
                  <div style={{padding:"7px 8px",fontWeight:700,color:s.color,fontSize:10}}>{s.abbr}</div>
                  <div style={{padding:"7px 8px",color:"#90a8c8",fontSize:10}}>{s.label}</div>
                  <div style={{padding:"7px 8px",color:"#204050",fontSize:8}}>{s.version.split(" ")[0]}</div>
                  <div style={{padding:"7px 8px",color:"#2a5070",fontSize:8,fontFamily:"monospace"}}>{s.range}</div>
                  <div style={{padding:"7px 8px",color:"#2a5070",fontSize:8,fontFamily:"monospace",
                    overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.source}</div>
                  <div style={{padding:"7px 8px",color:"#7090a0",fontSize:9}}>{s.effects[0]}</div>
                </div>
              ))}
            </div>

            {/* Derivation */}
            <div style={{background:"#05101a",border:"1px solid #0c1e34",
              borderRadius:10,padding:16}}>
              <div style={{fontSize:8,color:"#204050",letterSpacing:".1em",marginBottom:12}}>
                COMPLETE DERIVATION — EACH TICK
              </div>
              <pre style={{fontSize:9,color:"#90b8d8",lineHeight:1.95,
                fontFamily:"'JetBrains Mono',monospace",overflowX:"auto"}}>{
`# Inputs per tick
z_pred, z_actual: (32-D)   # latent transition
rho:   scalar               # Allen Neuropixels ρ  
act_magnitude: scalar       # ‖last_action‖₂

# ── Original (v16.10) ────────────────────────────────────────────
da_phasic = (1 − cos_sim(z_pred, z_actual)) / 2      # [0, 1]
DA        = 0.95·DA  + 0.05·da_phasic

stability = exp(−10·std(z_history[−8:]))              # [0, 1]
5HT       = 0.90·5HT + 0.10·stability

NE        = rho          # external — Allen Neuropixels

# ── Extended (v16.11) ────────────────────────────────────────────
ACh_raw   = (da_phasic + (1 − stability)) / 2
ACh       = 0.85·ACh + 0.15·ACh_raw

E/I       = clip(DA / (1 − 5HT + 0.1),  0.5, 2.0)   # free: no new source

Ado       = min(1.0, elapsed_sec / (4×3600))          # monotone ↑

eCB_raw   = da_phasic × min(1.0, act_magnitude)
eCB       = 0.85·eCB + 0.15·eCB_raw
DA_eff    = DA × (1 − eCB × 0.4)                      # retrograde suppression

# ── Derived planner parameters ────────────────────────────────────
regime      = classify(DA_eff, 5HT)   # EXPLOIT / EXPLORE / WAIT / REOBSERVE
ε_scale     = 1.0 + DA_eff×0.8 + NE×0.3
action_std  = clip(0.1 × E/I,  0.05, 0.20)
n_cand      = clip(64×(0.5+DA_eff)×(1−Ado×0.5),  16, 96)
lr_scale    = clip(0.5 + ACh×1.5,  0.2, 2.0)
act_scale   = regime_base × (1 − Ado×0.15)
DA_eff      → regime (uses eCB-suppressed DA for classification)`}
              </pre>
            </div>

            {/* Cockpit grid */}
            <div style={{background:"#05101a",border:"1px solid #0c1e34",
              borderRadius:10,padding:16}}>
              <div style={{fontSize:8,color:"#204050",letterSpacing:".1em",marginBottom:12}}>
                COCKPIT TELEMETRY — 15 FIELDS BROADCAST PER TICK
              </div>
              <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:7}}>
                {[
                  {k:"DA",     c:"#ff6b00",v:"signal"},
                  {k:"5HT",    c:"#00c8ff",v:"signal"},
                  {k:"NE",     c:"#cc44ff",v:"signal"},
                  {k:"ACH",    c:"#39ff14",v:"signal"},
                  {k:"EI",     c:"#ffd700",v:"derived"},
                  {k:"ADO",    c:"#ff4488",v:"session"},
                  {k:"ECB",    c:"#00ffaa",v:"retro"},
                  {k:"DA_EFF", c:"#ff9900",v:"eCB-adj"},
                  {k:"REGIME", c:"#ffffff",v:"class"},
                  {k:"CONF",   c:"#7090b0",v:"label"},
                  {k:"ACT_SCALE",c:"#50c0ff",v:"param"},
                  {k:"EPS_SCALE",c:"#50c0ff",v:"param"},
                  {k:"ACT_STD",  c:"#50c0ff",v:"param"},
                  {k:"N_CAND",   c:"#50c0ff",v:"param"},
                  {k:"LR_SCALE", c:"#50c0ff",v:"param"},
                ].map(({k,c,v})=>(
                  <div key={k} style={{background:"#030810",
                    border:`1px solid ${c}22`,borderRadius:5,
                    padding:"6px 8px",textAlign:"center"}}>
                    <div style={{fontSize:9,color:c,fontWeight:700}}>{k}</div>
                    <div style={{fontSize:7,color:"#204050",marginTop:1}}>{v}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
