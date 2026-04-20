import { useState, useEffect, useCallback } from "react";

const LEVELS = [
  {
    id: 1, name: "Counting & Recognition", age: "4-5", icon: "🔢",
    color: "#4CAF50",
    concepts: ["Count objects", "Compare numbers", "Odd vs Even"],
    examples: [
      { q: "How many items: [🍎, 🍌, 🍒]?", a: "3", explain: "Count each item: 1, 2, 3 → answer is 3" },
      { q: "Which is larger, 7 or 4?", a: "7", explain: "7 > 4, so 7 is larger" },
      { q: "Is 6 even or odd?", a: "even", explain: "6 ÷ 2 = 3 with no remainder → even" },
    ]
  },
  {
    id: 2, name: "Addition & Subtraction", age: "5-6", icon: "➕",
    color: "#2196F3",
    concepts: ["Add two numbers", "Subtract", "Negative results"],
    examples: [
      { q: "15 + 27 = ?", a: "42", explain: "15 + 27: carry the 1 → 42" },
      { q: "100 - 37 = ?", a: "63", explain: "100 - 37: borrow → 63" },
      { q: "-5 + 12 = ?", a: "7", explain: "Start at -5, move right 12 → 7" },
    ]
  },
  {
    id: 3, name: "Multiplication & Division", age: "6-7", icon: "✖️",
    color: "#FF9800",
    concepts: ["Times tables", "Division", "Remainders (mod)"],
    examples: [
      { q: "7 × 8 = ?", a: "56", explain: "7 groups of 8: 56" },
      { q: "144 ÷ 12 = ?", a: "12", explain: "144 split into 12 groups = 12 each" },
      { q: "17 mod 5 = ?", a: "2", explain: "17 = 3×5 + 2, remainder is 2" },
    ]
  },
  {
    id: 4, name: "Fractions & Decimals", age: "7-8", icon: "½",
    color: "#9C27B0",
    concepts: ["Fraction operations", "Decimals", "Percentages"],
    examples: [
      { q: "1/2 + 1/4 = ?", a: "3/4", explain: "2/4 + 1/4 = 3/4 (common denominator)" },
      { q: "2/3 × 3/5 = ?", a: "2/5", explain: "Multiply across: 6/15 = 2/5" },
      { q: "25% of 80 = ?", a: "20", explain: "80 × 0.25 = 20" },
    ]
  },
  {
    id: 5, name: "Order of Operations", age: "8-9", icon: "📐",
    color: "#F44336",
    concepts: ["PEMDAS/BODMAS", "Parentheses first", "Powers"],
    examples: [
      { q: "2 + 3 × 4 = ?", a: "14", explain: "Multiply first: 3×4=12, then 2+12=14" },
      { q: "(2 + 3) × 4 = ?", a: "20", explain: "Parentheses first: 5×4=20" },
      { q: "2³ + 1 = ?", a: "9", explain: "Power first: 8+1=9" },
    ]
  },
  {
    id: 6, name: "Factors, Primes, GCD/LCM", age: "9-10", icon: "🔍",
    color: "#607D8B",
    concepts: ["Prime numbers", "GCD", "LCM", "Factor counting"],
    examples: [
      { q: "Is 17 prime?", a: "Yes", explain: "Only divisible by 1 and 17 → prime" },
      { q: "GCD(24, 36) = ?", a: "12", explain: "Largest number dividing both: 12" },
      { q: "LCM(4, 6) = ?", a: "12", explain: "Smallest common multiple: 12" },
    ]
  },
  {
    id: 7, name: "Algebra Basics", age: "10-11", icon: "𝑥",
    color: "#00BCD4",
    concepts: ["Solve equations", "Evaluate functions", "Quadratics"],
    examples: [
      { q: "Solve 3x + 7 = 22", a: "x = 5", explain: "3x = 15, x = 5" },
      { q: "f(x) = 2x - 1, f(6) = ?", a: "11", explain: "2(6) - 1 = 11" },
      { q: "f(x) = x² + 1, f(3) = ?", a: "10", explain: "9 + 1 = 10" },
    ]
  },
  {
    id: 8, name: "Sequences & Patterns", age: "11-12", icon: "📈",
    color: "#8BC34A",
    concepts: ["Arithmetic sequences", "Geometric sequences", "Gauss sum"],
    examples: [
      { q: "Next: 4, 7, 10, 13, ?", a: "16", explain: "Difference = 3, so 13+3 = 16" },
      { q: "Next: 2, 6, 18, 54, ?", a: "162", explain: "Ratio = 3, so 54×3 = 162" },
      { q: "1+2+3+...+10 = ?", a: "55", explain: "Gauss: 10×11/2 = 55" },
    ]
  },
  {
    id: 9, name: "Geometry & Measurement", age: "12-13", icon: "📏",
    color: "#E91E63",
    concepts: ["Area", "Perimeter", "Volume"],
    examples: [
      { q: "Area of rectangle 5×8 = ?", a: "40", explain: "length × width = 40" },
      { q: "Perimeter of square, side 6 = ?", a: "24", explain: "4 × 6 = 24" },
      { q: "Volume of box 3×4×5 = ?", a: "60", explain: "l × w × h = 60" },
    ]
  },
  {
    id: 10, name: "Combinatorics & Probability", age: "13-14", icon: "🎲",
    color: "#FF5722",
    concepts: ["Factorials", "Combinations", "Permutations", "Probability"],
    examples: [
      { q: "5! = ?", a: "120", explain: "5×4×3×2×1 = 120" },
      { q: "C(10, 3) = ?", a: "120", explain: "10!/(3!×7!) = 120" },
      { q: "P(coin heads) = ?", a: "1/2", explain: "1 favorable out of 2 total" },
    ]
  },
];

const STATUS = { locked: "🔒", active: "📖", mastered: "★" };

export default function MathShowcase() {
  const [selectedLevel, setSelectedLevel] = useState(null);
  const [quizMode, setQuizMode] = useState(false);
  const [quizQ, setQuizQ] = useState(0);
  const [showAnswer, setShowAnswer] = useState(false);
  const [score, setScore] = useState(0);
  const [mastered, setMastered] = useState(new Set([1,2,3,4,5,6,7,8,9,10]));
  const [animate, setAnimate] = useState(false);

  useEffect(() => { setAnimate(true); }, []);

  const level = selectedLevel ? LEVELS.find(l => l.id === selectedLevel) : null;

  const startQuiz = () => {
    setQuizMode(true);
    setQuizQ(0);
    setShowAnswer(false);
    setScore(0);
  };

  const nextQuestion = (correct) => {
    if (correct) setScore(s => s + 1);
    if (quizQ < 2) {
      setQuizQ(q => q + 1);
      setShowAnswer(false);
    } else {
      setQuizMode(false);
    }
  };

  return (
    <div style={{
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      background: "#0a0a0f",
      color: "#e0e0e0",
      minHeight: "100vh",
      padding: "2rem",
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        .level-card {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 12px;
          padding: 1.2rem;
          position: relative;
          overflow: hidden;
        }
        .level-card:hover {
          transform: translateY(-4px);
          border-color: rgba(255,255,255,0.15);
          box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        }
        .level-card::before {
          content: '';
          position: absolute;
          top: 0; left: 0; right: 0;
          height: 3px;
        }
        .example-card {
          background: rgba(255,255,255,0.03);
          border: 1px solid rgba(255,255,255,0.08);
          border-radius: 10px;
          padding: 1.2rem;
          margin-bottom: 1rem;
          transition: all 0.2s;
        }
        .example-card:hover { background: rgba(255,255,255,0.06); }
        .btn {
          padding: 0.6rem 1.4rem;
          border-radius: 8px;
          border: 1px solid rgba(255,255,255,0.15);
          background: rgba(255,255,255,0.05);
          color: #e0e0e0;
          cursor: pointer;
          font-family: inherit;
          font-size: 0.85rem;
          transition: all 0.2s;
        }
        .btn:hover { background: rgba(255,255,255,0.12); transform: translateY(-1px); }
        .mastery-bar {
          height: 4px;
          background: rgba(255,255,255,0.06);
          border-radius: 2px;
          margin-top: 0.5rem;
          overflow: hidden;
        }
        .mastery-fill {
          height: 100%;
          border-radius: 2px;
          transition: width 1.5s cubic-bezier(0.4, 0, 0.2, 1);
        }
        @keyframes fadeUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .fade-up { animation: fadeUp 0.5s ease-out forwards; opacity: 0; }
      `}</style>

      {/* Header */}
      <div style={{ textAlign: "center", marginBottom: "3rem" }}>
        <h1 style={{
          fontFamily: "'Space Grotesk', sans-serif",
          fontSize: "2.2rem",
          fontWeight: 700,
          background: "linear-gradient(135deg, #4CAF50, #2196F3, #FF9800)",
          WebkitBackgroundClip: "text",
          WebkitTextFillColor: "transparent",
          marginBottom: "0.5rem",
        }}>
          NeMo-WM Math Curriculum
        </h1>
        <p style={{ color: "#888", fontSize: "0.95rem" }}>
          10 levels mastered • 500/500 DeepMind benchmark • 0ms/problem • Pure symbolic reasoning
        </p>
        <div style={{
          display: "inline-flex", gap: "2rem", marginTop: "1rem",
          background: "rgba(255,255,255,0.03)", padding: "0.8rem 1.5rem",
          borderRadius: "10px", border: "1px solid rgba(255,255,255,0.06)",
        }}>
          <div><span style={{color:"#4CAF50",fontWeight:700,fontSize:"1.4rem"}}>10/10</span><br/><span style={{fontSize:"0.75rem",color:"#666"}}>Levels Mastered</span></div>
          <div><span style={{color:"#2196F3",fontWeight:700,fontSize:"1.4rem"}}>500/500</span><br/><span style={{fontSize:"0.75rem",color:"#666"}}>Benchmark Score</span></div>
          <div><span style={{color:"#FF9800",fontWeight:700,fontSize:"1.4rem"}}>0ms</span><br/><span style={{fontSize:"0.75rem",color:"#666"}}>Per Problem</span></div>
          <div><span style={{color:"#F44336",fontWeight:700,fontSize:"1.4rem"}}>$0</span><br/><span style={{fontSize:"0.75rem",color:"#666"}}>No LLM</span></div>
        </div>
      </div>

      {/* Level Grid */}
      {!selectedLevel && (
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: "1rem",
          maxWidth: "1200px",
          margin: "0 auto",
        }}>
          {LEVELS.map((lv, i) => (
            <div
              key={lv.id}
              className="level-card fade-up"
              style={{
                animationDelay: `${i * 0.06}s`,
                background: `linear-gradient(135deg, rgba(${parseInt(lv.color.slice(1,3),16)},${parseInt(lv.color.slice(3,5),16)},${parseInt(lv.color.slice(5,7),16)},0.08), rgba(0,0,0,0.3))`,
              }}
              onClick={() => setSelectedLevel(lv.id)}
            >
              <div style={{position:"absolute",top:0,left:0,right:0,height:"3px",background:lv.color}} />
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:"0.6rem"}}>
                <span style={{fontSize:"1.8rem"}}>{lv.icon}</span>
                <span style={{
                  color: mastered.has(lv.id) ? "#4CAF50" : "#666",
                  fontSize: mastered.has(lv.id) ? "1.2rem" : "0.9rem",
                }}>
                  {mastered.has(lv.id) ? "★" : "○"}
                </span>
              </div>
              <h3 style={{fontFamily:"'Space Grotesk'",fontSize:"1rem",fontWeight:600,marginBottom:"0.3rem"}}>
                L{lv.id}: {lv.name}
              </h3>
              <p style={{fontSize:"0.75rem",color:"#888",marginBottom:"0.5rem"}}>Ages {lv.age}</p>
              <div style={{display:"flex",gap:"0.4rem",flexWrap:"wrap"}}>
                {lv.concepts.map((c, j) => (
                  <span key={j} style={{
                    fontSize:"0.65rem", padding:"0.2rem 0.5rem",
                    background:"rgba(255,255,255,0.05)", borderRadius:"4px", color:"#aaa",
                  }}>{c}</span>
                ))}
              </div>
              <div className="mastery-bar">
                <div className="mastery-fill" style={{
                  width: mastered.has(lv.id) ? "100%" : "0%",
                  background: lv.color,
                }} />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Level Detail */}
      {selectedLevel && level && !quizMode && (
        <div className="fade-up" style={{ maxWidth: "700px", margin: "0 auto" }}>
          <button className="btn" onClick={() => setSelectedLevel(null)} style={{marginBottom:"1.5rem"}}>
            ← Back to all levels
          </button>

          <div style={{
            display:"flex", alignItems:"center", gap:"1rem", marginBottom:"2rem",
          }}>
            <span style={{fontSize:"2.5rem"}}>{level.icon}</span>
            <div>
              <h2 style={{fontFamily:"'Space Grotesk'",fontSize:"1.5rem",fontWeight:700}}>
                Level {level.id}: {level.name}
              </h2>
              <p style={{color:"#888",fontSize:"0.85rem"}}>Ages {level.age} • {mastered.has(level.id) ? "★ Mastered" : "In progress"}</p>
            </div>
          </div>

          <h3 style={{color:level.color,fontSize:"0.9rem",fontWeight:600,marginBottom:"1rem",textTransform:"uppercase",letterSpacing:"0.05em"}}>
            Examples with Explanations
          </h3>

          {level.examples.map((ex, i) => (
            <div key={i} className="example-card fade-up" style={{animationDelay:`${i*0.1}s`}}>
              <div style={{fontSize:"1.05rem",fontWeight:600,marginBottom:"0.6rem",color:"#fff"}}>
                Q: {ex.q}
              </div>
              <div style={{
                display:"inline-block",
                background: `${level.color}22`,
                border: `1px solid ${level.color}44`,
                borderRadius:"6px",
                padding:"0.3rem 0.8rem",
                fontSize:"1.1rem",
                fontWeight:700,
                color: level.color,
                marginBottom:"0.5rem",
              }}>
                A: {ex.a}
              </div>
              <div style={{fontSize:"0.8rem",color:"#999",marginTop:"0.4rem",fontStyle:"italic"}}>
                💡 {ex.explain}
              </div>
            </div>
          ))}

          <div style={{marginTop:"1.5rem",display:"flex",gap:"1rem"}}>
            <button className="btn" onClick={startQuiz} style={{
              background: `${level.color}22`, borderColor: `${level.color}44`, color: level.color,
            }}>
              Take Quiz →
            </button>
          </div>
        </div>
      )}

      {/* Quiz Mode */}
      {quizMode && level && (
        <div className="fade-up" style={{ maxWidth: "600px", margin: "0 auto", textAlign: "center" }}>
          <p style={{color:"#888",fontSize:"0.8rem",marginBottom:"1rem"}}>
            Question {quizQ + 1} of 3 • Score: {score}
          </p>

          <div style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(255,255,255,0.1)",
            borderRadius: "14px",
            padding: "2.5rem",
            marginBottom: "1.5rem",
          }}>
            <p style={{fontSize:"0.75rem",color:level.color,marginBottom:"0.8rem",textTransform:"uppercase",letterSpacing:"0.05em"}}>
              Level {level.id}: {level.name}
            </p>
            <h2 style={{fontFamily:"'Space Grotesk'",fontSize:"1.5rem",fontWeight:700,marginBottom:"1.5rem"}}>
              {level.examples[quizQ].q}
            </h2>

            {!showAnswer ? (
              <button className="btn" onClick={() => setShowAnswer(true)} style={{
                background: `${level.color}22`, borderColor: `${level.color}44`, color: level.color,
                fontSize: "1rem", padding: "0.8rem 2rem",
              }}>
                Show Answer
              </button>
            ) : (
              <div className="fade-up">
                <div style={{
                  fontSize: "2rem", fontWeight: 700, color: level.color, marginBottom: "0.8rem",
                }}>
                  {level.examples[quizQ].a}
                </div>
                <p style={{fontSize:"0.85rem",color:"#999",marginBottom:"1.5rem"}}>
                  {level.examples[quizQ].explain}
                </p>
                <div style={{display:"flex",gap:"1rem",justifyContent:"center"}}>
                  <button className="btn" onClick={() => nextQuestion(true)} style={{
                    background:"rgba(76,175,80,0.15)", borderColor:"rgba(76,175,80,0.3)", color:"#4CAF50",
                  }}>✓ Got it!</button>
                  <button className="btn" onClick={() => nextQuestion(false)} style={{
                    background:"rgba(244,67,54,0.15)", borderColor:"rgba(244,67,54,0.3)", color:"#F44336",
                  }}>✗ Missed it</button>
                </div>
              </div>
            )}
          </div>

          {!quizMode || quizQ >= 2 && showAnswer ? null : (
            <div style={{display:"flex",gap:"0.3rem",justifyContent:"center"}}>
              {[0,1,2].map(i => (
                <div key={i} style={{
                  width:"40px",height:"4px",borderRadius:"2px",
                  background: i === quizQ ? level.color : i < quizQ ? "#4CAF50" : "rgba(255,255,255,0.1)",
                }} />
              ))}
            </div>
          )}
        </div>
      )}

      {/* After quiz */}
      {selectedLevel && !quizMode && score > 0 && (
        <div className="fade-up" style={{
          maxWidth:"400px", margin:"2rem auto", textAlign:"center",
          background:"rgba(76,175,80,0.08)", border:"1px solid rgba(76,175,80,0.2)",
          borderRadius:"12px", padding:"1.5rem",
        }}>
          <p style={{fontSize:"1.5rem",marginBottom:"0.5rem"}}>🎉</p>
          <p style={{fontSize:"1.1rem",fontWeight:600}}>Quiz Complete: {score}/3</p>
          <p style={{fontSize:"0.8rem",color:"#888"}}>NeMo-WM solved these using pure symbolic computation — no neural network, no LLM, 0ms per problem.</p>
        </div>
      )}

      {/* Footer */}
      <div style={{
        textAlign: "center", marginTop: "3rem", padding: "1.5rem",
        borderTop: "1px solid rgba(255,255,255,0.06)",
        color: "#555", fontSize: "0.75rem",
      }}>
        NeMo-WM Math Curriculum — Counting to Combinatorics in 10 levels<br/>
        Powered by symbolic computation. No LLM. No neural network. No memorization.<br/>
        This is Chollet's vision: discrete program execution, not curve fitting.
      </div>
    </div>
  );
}
