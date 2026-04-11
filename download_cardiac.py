"""
download_cardiac.py — CORTEX-PE v16.15
Downloads PhysioNet 2016 Heart Sound Challenge Dataset

Source: https://physionet.org/content/challenge-2016/1.0.0/
~120MB total — WAV files of heart sounds labeled normal/abnormal

Structure after download:
    cardiac_data/
        training-a/   (409 recordings, labeled)
        training-b/   (490 recordings, labeled)
        training-c/   (31 recordings, labeled)
        training-d/   (55 recordings, labeled)
        training-e/   (2141 recordings, labeled)
        training-f/   (114 recordings, labeled)
        RECORDS       (master file list)

Labels: -1 = abnormal, 1 = normal  (in each subset's REFERENCE.csv)

Usage:
    python download_cardiac.py               # downloads all subsets
    python download_cardiac.py --subset a    # download only subset a (~10MB)
    python download_cardiac.py --subset a b  # download subsets a and b
"""

import argparse
import urllib.request
import zipfile
import io
from pathlib import Path


PHYSIONET_BASE = "https://physionet.org/files/challenge-2016/1.0.0/"

SUBSETS = {
    "a": "training-a.zip",
    "b": "training-b.zip",
    "c": "training-c.zip",
    "d": "training-d.zip",
    "e": "training-e.zip",
    "f": "training-f.zip",
}

# Approx sizes in MB
SUBSET_SIZES = {"a": 9, "b": 11, "c": 1, "d": 2, "e": 50, "f": 4}


def download_and_extract(subset: str, out_dir: Path) -> bool:
    fname  = SUBSETS[subset]
    url    = PHYSIONET_BASE + fname
    dest   = out_dir / f"training-{subset}"

    if dest.exists() and any(dest.glob("*.wav")):
        n = len(list(dest.glob("*.wav")))
        print(f"  ✓ training-{subset}  already present ({n} wav files)")
        return True

    size_mb = SUBSET_SIZES.get(subset, "?")
    print(f"  ↓ training-{subset}  (~{size_mb}MB) ... ", end="", flush=True)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CORTEX-PE/16.15"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()

        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(out_dir)

        n_wav = len(list(dest.glob("*.wav")))
        print(f"OK ({n_wav} wav files)")
        return True

    except Exception as e:
        print(f"FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download PhysioNet 2016 cardiac data")
    parser.add_argument("--out",    default="./cardiac_data",
                        help="Output directory (default: ./cardiac_data)")
    parser.add_argument("--subset", nargs="+", choices=list(SUBSETS.keys()),
                        default=list(SUBSETS.keys()),
                        help="Subsets to download (default: all)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  PhysioNet 2016 Heart Sound Downloader — CORTEX-PE v16.15")
    print(f"  Output  : {out_dir.resolve()}")
    print(f"  Subsets : {args.subset}")
    total_mb = sum(SUBSET_SIZES.get(s, 0) for s in args.subset)
    print(f"  Est. size: ~{total_mb}MB\n")

    ok, failed = [], []
    for subset in args.subset:
        if download_and_extract(subset, out_dir):
            ok.append(subset)
        else:
            failed.append(subset)

    # Count totals
    total_wav  = len(list(out_dir.rglob("*.wav")))
    total_norm = 0
    total_abn  = 0
    for ref in out_dir.rglob("REFERENCE.csv"):
        for line in ref.read_text().strip().splitlines():
            parts = line.strip().split(",")
            if len(parts) >= 2:
                label = parts[1].strip()
                if label == "1":
                    total_norm += 1
                elif label == "-1":
                    total_abn += 1

    print(f"\n  Downloaded : {ok}")
    if failed:
        print(f"  Failed     : {failed}")
    print(f"  Total WAV  : {total_wav:,}")
    print(f"  Normal     : {total_norm:,}")
    print(f"  Abnormal   : {total_abn:,}")

    if ok:
        print(f"\n  ✅ Ready — run training:")
        print(f"  python train_cardiac_distillation.py --data {args.out}")
    print()


if __name__ == "__main__":
    main()
