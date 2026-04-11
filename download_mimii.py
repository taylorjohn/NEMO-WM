"""
download_mimii.py — CORTEX-PE v16.15
Downloads MIMII Dataset from Zenodo (record 3384388)

Strategy: download one machine type at one SNR level to start.
Full dataset is 100GB — start with 0dB_fan (~10GB), the canonical benchmark.

Structure after extraction:
    mimii_data/
        0dB_fan/
            id_00/  normal/*.wav  abnormal/*.wav
            id_02/  normal/*.wav  abnormal/*.wav
            id_04/  normal/*.wav  abnormal/*.wav
            id_06/  normal/*.wav  abnormal/*.wav

Usage:
    python download_mimii.py                    # downloads 0dB_fan (~10GB)
    python download_mimii.py --machine pump     # downloads 0dB_pump (~8GB)
    python download_mimii.py --snr 6            # downloads 6dB_fan (~10GB)
"""

import argparse
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://zenodo.org/records/3384388/files/"

# All available files: {snr}_{machine}.zip
MACHINES = ["fan", "pump", "slider", "valve"]
SNRS     = ["-6_dB", "0_dB", "6_dB"]

SIZES_GB = {
    ("-6_dB", "fan"): 10.9, ("-6_dB", "pump"): 8.2,
    ("-6_dB", "slider"): 8.0, ("-6_dB", "valve"): 8.0,
    ("0_dB",  "fan"): 10.4, ("0_dB",  "pump"): 7.9,
    ("0_dB",  "slider"): 7.5, ("0_dB",  "valve"): 7.5,
    ("6_dB",  "fan"): 10.2, ("6_dB",  "pump"): 7.7,
    ("6_dB",  "slider"): 7.1, ("6_dB",  "valve"): 6.9,
}


def download_mimii(snr: str, machine: str, out_dir: Path) -> bool:
    snr_str  = f"{snr}_dB" if not snr.endswith("dB") else snr
    fname    = f"{snr_str}_{machine}.zip"
    url      = BASE + fname + "?download=1"
    dest_zip = out_dir / fname
    dest_dir = out_dir / f"{snr_str}_{machine}"

    if dest_dir.exists() and any(dest_dir.rglob("*.wav")):
        n = len(list(dest_dir.rglob("*.wav")))
        print(f"  ✓ {fname} already extracted ({n:,} wav files)")
        return True

    size_gb = SIZES_GB.get((snr_str, machine), "?")
    print(f"\n  Downloading {fname} (~{size_gb}GB)...")
    print(f"  URL: {url}")
    print(f"  This will take ~{int(float(str(size_gb).replace('?','10')) * 60 / 100)} min on a 100Mbps connection")
    print(f"  Press Ctrl+C to cancel\n")

    try:
        t0 = time.time()

        def progress(count, block_size, total_size):
            if total_size > 0:
                pct  = min(count * block_size / total_size * 100, 100)
                mb   = count * block_size / 1024 / 1024
                elapsed = time.time() - t0
                speed = mb / elapsed if elapsed > 0 else 0
                print(f"\r  {pct:5.1f}%  {mb:6.0f}MB  {speed:5.1f}MB/s", end="", flush=True)

        req = urllib.request.Request(url, headers={"User-Agent": "CORTEX-PE/16.15"})
        urllib.request.urlretrieve(url, dest_zip, reporthook=progress)
        print(f"\n  Download complete: {dest_zip.stat().st_size / 1e9:.2f}GB")

        print(f"  Extracting {fname}...")
        with zipfile.ZipFile(dest_zip, "r") as zf:
            zf.extractall(out_dir)

        # Cleanup zip to save space
        dest_zip.unlink()
        n = len(list(dest_dir.rglob("*.wav")))
        print(f"  ✅ Extracted: {n:,} wav files in {dest_dir}")
        return True

    except KeyboardInterrupt:
        print("\n  Cancelled")
        if dest_zip.exists():
            dest_zip.unlink()
        return False
    except Exception as e:
        print(f"\n  FAILED: {e}")
        if dest_zip.exists():
            dest_zip.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download MIMII dataset from Zenodo")
    parser.add_argument("--out",     default="./mimii_data")
    parser.add_argument("--machine", default="fan",
                        choices=MACHINES, help="Machine type (default: fan)")
    parser.add_argument("--snr",     default="0",
                        choices=["min6", "-6", "0", "6"],
                        help="SNR level: -6, 0, 6 dB (default: 0)")
    args = parser.parse_args()

    # Normalise SNR arg
    snr_map = {"min6": "-6_dB", "-6": "-6_dB", "0": "0_dB", "6": "6_dB"}
    snr_str = snr_map[args.snr]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  MIMII Downloader — CORTEX-PE v16.15")
    print(f"  Machine : {args.machine}")
    print(f"  SNR     : {snr_str}")
    print(f"  Output  : {out_dir.resolve()}")

    success = download_mimii(snr_str, args.machine, out_dir)

    if success:
        dest = out_dir / f"{snr_str}_{args.machine}"
        normal_count   = len(list(dest.rglob("normal/*.wav")))
        abnormal_count = len(list(dest.rglob("abnormal/*.wav")))
        print(f"\n  Normal   : {normal_count:,} clips")
        print(f"  Abnormal : {abnormal_count:,} clips")
        print(f"\n  Run eval:")
        print(f"  python eval_mimii.py --data {args.out} --machine {args.machine} --snr {args.snr}")


if __name__ == "__main__":
    main()
