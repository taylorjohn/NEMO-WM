"""
download_cwru.py — CORTEX-PE v16.15
Downloads CWRU Bearing Fault Dataset directly from Case Western Reserve University.

Downloads ~50MB of .mat files covering four fault conditions at 12kHz:
    Normal       — healthy bearing baseline
    Ball fault   — rolling element damage (0.007" diameter)
    Inner race   — inner raceway damage (0.007" diameter)
    Outer race   — outer raceway damage (0.007" diameter)

All files saved flat into --out directory (default: ./cwru_data)

Usage:
    python download_cwru.py
    python download_cwru.py --out ./cwru_data --48k
"""

import argparse
import urllib.request
from pathlib import Path


# ── File manifest ─────────────────────────────────────────────────────────────
# (filename, url, label)
# Source: https://engineering.case.edu/bearingdatacenter/download-data-file

BASE = "https://engineering.case.edu/sites/default/files/"

FILES_12K = [
    # Normal
    ("Normal_0.mat",     BASE + "Normal_0.mat",    "normal"),
    # Ball fault 0.007"
    ("B007_0.mat",       BASE + "B007_0.mat",      "ball"),
    ("B007_1.mat",       BASE + "B007_1.mat",      "ball"),
    ("B007_2.mat",       BASE + "B007_2.mat",      "ball"),
    ("B007_3.mat",       BASE + "B007_3.mat",      "ball"),
    # Inner race 0.007"
    ("IR007_0.mat",      BASE + "IR007_0.mat",     "inner"),
    ("IR007_1.mat",      BASE + "IR007_1.mat",     "inner"),
    ("IR007_2.mat",      BASE + "IR007_2.mat",     "inner"),
    ("IR007_3.mat",      BASE + "IR007_3.mat",     "inner"),
    # Outer race 0.007" @ 6 o'clock
    ("OR007@6_0.mat",    BASE + "OR007@6_0.mat",   "outer"),
    ("OR007@6_1.mat",    BASE + "OR007@6_1.mat",   "outer"),
    ("OR007@6_2.mat",    BASE + "OR007@6_2.mat",   "outer"),
    ("OR007@6_3.mat",    BASE + "OR007@6_3.mat",   "outer"),
]

# 48kHz drive end — larger files, richer frequency content
FILES_48K = [
    ("48k_Normal_0.mat", BASE + "Normal_0.mat",    "normal"),
    ("48k_B007_0.mat",   BASE + "48k_B007_0.mat",  "ball"),
    ("48k_IR007_0.mat",  BASE + "48k_IR007_0.mat", "inner"),
    ("48k_OR007@6_0.mat",BASE + "48k_OR007@6_0.mat","outer"),
]


def download_file(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  ✓ Already exists: {dest.name}")
        return True
    try:
        print(f"  ↓ {dest.name} ... ", end="", flush=True)
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"{size_kb}KB")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download CWRU bearing dataset")
    parser.add_argument("--out", default="./cwru_data", help="Output directory")
    parser.add_argument("--48k", dest="use_48k", action="store_true",
                        help="Download 48kHz files instead of 12kHz")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files   = FILES_48K if args.use_48k else FILES_12K
    rate    = "48kHz" if args.use_48k else "12kHz"

    print(f"\n  CWRU Bearing Dataset Downloader — CORTEX-PE v16.15")
    print(f"  Rate  : {rate}")
    print(f"  Output: {out_dir.resolve()}")
    print(f"  Files : {len(files)}\n")

    ok, failed = 0, []
    for fname, url, label in files:
        success = download_file(url, out_dir / fname)
        if success:
            ok += 1
        else:
            failed.append(fname)

    print(f"\n  Downloaded: {ok}/{len(files)}")
    if failed:
        print(f"  Failed    : {failed}")
        print("\n  For failed files, download manually from:")
        print("  https://engineering.case.edu/bearingdatacenter/download-data-file")
    else:
        print(f"\n  ✅ All files downloaded to {out_dir.resolve()}")
        print("\n  Run anomaly detection:")
        print(f"  python subspace_ad.py \\")
        print(f"      --encoder ./checkpoints/maze_weak_sigreg_straight/cortex_student_phase2_final.pt \\")
        print(f"      --data {args.out}")


if __name__ == "__main__":
    main()
