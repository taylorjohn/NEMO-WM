from pathlib import Path
lines = Path("run_trading.py").read_text(encoding="utf-8").splitlines()
for i, l in enumerate(lines):
    if "sendto" in l or ("hud" in l.lower() and "json" in l.lower()):
        start = max(0, i-1)
        end = min(len(lines), i+20)
        print(f"\n=== Found at line {i+1} ===")
        for j in range(start, end):
            print(f"  {j+1:4d}: {lines[j]}")
        break
