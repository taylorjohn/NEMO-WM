from pathlib import Path
lines = Path("run_trading.py").read_text(encoding="utf-8").splitlines()
print(f"Total lines: {len(lines)}")
print()
print("Lines 55-80:")
for i, l in enumerate(lines[54:80], start=55):
    marker = " <-- last top-level import" if i == 66 else ""
    print(f"  {i:4d}: {repr(l)}{marker}")
