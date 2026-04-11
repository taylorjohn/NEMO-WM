"""
neuropixels_reader.py — Allen Brain Observatory Neuropixels arousal scalar

Reads spike times from the cached NWB session file and produces a normalised
arousal scalar rho in [0, 1] for biological modulation of the MPC planner.

Session 715093703 diagnostics (confirmed on disk):
    Total neurons:  2779
    Session range:  t=[27, 9578]s  (~2.6 hours)
    Neuron 0:       31302 spikes,  3.28 Hz mean
    Neuron 2:       57564 spikes,  6.03 Hz mean
    Neuron 4:       74302 spikes,  7.78 Hz mean

Configuration:
    session_window = 9000s  (full session — cycles through all data)
    start_offset   = 500s   (skip sparse early period t=[0, 27]s)
    bin_ms         = 500ms  (matches ~3-7 Hz firing rate per neuron)
    n_neurons      = 20

Expected rho behaviour:
    History accumulates over first 10 samples (~10s at 1Hz loop)
    Once warm: rho varies 0.1-0.9 with typical std ~0.15
    High-activity epochs (population burst): rho -> 0.85+
    Low-activity epochs (quiet):             rho -> 0.15

Usage:
    from neuropixels_reader import get_neuropixels_rho
    rho = get_neuropixels_rho()   # float in [0, 1]
"""

import time
import threading
import numpy as np


# =============================================================================
# NWB session path
# =============================================================================
NWB_PATH = (
    r"C:\Users\MeteorAI\Desktop\cortex-12v15\allen_cache"
    r"\session_715093703\session_715093703.nwb"
)


# =============================================================================
# NeuropixelsReader
# =============================================================================
class NeuropixelsReader:
    """
    Thread-safe rolling Neuropixels arousal scalar.

    Loads the NWB session file once at init. All subsequent calls to
    get_rho() are O(N_neurons) numpy slicing — no file I/O.

    Falls back to rho=0.0 if pynwb is unavailable or the NWB file
    is not found. The trading loop never crashes on this.

    Parameters
    ----------
    nwb_path : str
        Path to the .nwb session file.
    n_neurons : int
        Number of neurons to monitor. Default 20.
    bin_ms : float
        Spike count window in milliseconds. Default 500ms.
        At 3-7 Hz per neuron x 20 neurons: ~30-70 spikes per 500ms bin.
    history_len : int
        Rolling z-score history length. Default 300 samples.
    session_window : float
        Duration of biological time window to cycle through in seconds.
        Default 9000s (full session). Wall-clock time maps as:
            t_bio = start_offset + (t_wall - t_init) % session_window
    start_offset : float
        Offset into session to begin reading. Default 500s.
        Skips the sparse early period (t=[0,27]s has no spikes).
    """

    def __init__(
        self,
        nwb_path:       str   = NWB_PATH,
        n_neurons:      int   = 20,
        bin_ms:         float = 500.0,
        history_len:    int   = 300,
        session_window: float = 9000.0,
        start_offset:   float = 500.0,
    ):
        self.n_neurons      = n_neurons
        self.bin_s          = bin_ms / 1000.0
        self.history_len    = history_len
        self.session_window = session_window
        self.start_offset   = start_offset
        self.start_time     = time.time()
        self._lock          = threading.Lock()
        self._history       = []
        self._spike_times   = None
        self._available     = False

        self._load(nwb_path)

    def _load(self, path: str) -> None:
        """Load spike times from NWB file into memory. Called once at init."""
        try:
            import pynwb
            io      = pynwb.NWBHDF5IO(path, "r")
            nwbfile = io.read()
            units   = nwbfile.units.to_dataframe()
            n       = min(self.n_neurons, len(units))

            self._spike_times = [
                np.array(units.spike_times.iloc[i], dtype=np.float64)
                for i in range(n)
            ]
            self._available = True
            print(
                f"✅ Neuropixels: {n} neurons loaded "
                f"(window={self.session_window:.0f}s, "
                f"offset={self.start_offset:.0f}s, "
                f"bin={self.bin_s*1000:.0f}ms)"
            )

        except FileNotFoundError:
            print(f"⚠️  Neuropixels NWB not found: {path}")
            print("   rho=0.0 — trading loop unaffected")
        except ImportError:
            print("⚠️  pynwb not installed — run: pip install pynwb")
            print("   rho=0.0 — trading loop unaffected")
        except Exception as e:
            print(f"⚠️  Neuropixels load error: {e}")
            print("   rho=0.0 — trading loop unaffected")

    def get_rho(self) -> float:
        """
        Returns normalised arousal scalar rho in [0, 1].

        Thread-safe. Returns 0.0 if NWB unavailable or history < 10 samples.

        Computation:
            1. Map wall-clock time to biological session time
            2. Count spikes from N_neurons in the current 500ms bin
            3. z-score against rolling 300-sample history
            4. sigmoid(z) -> rho in [0, 1]

        Biological time mapping:
            t_bio = start_offset + (t_wall - t_init) % session_window
            At 1Hz loop: advances 1s per tick through 9000s of data
        """
        if not self._available or not self._spike_times:
            return 0.0

        # Map wall-clock to biological time (cycles through full session)
        wall_elapsed = time.time() - self.start_time
        t_bio        = self.start_offset + (wall_elapsed % self.session_window)
        t0           = t_bio
        t1           = t_bio + self.bin_s

        # Count spikes across all monitored neurons in bin [t0, t1]
        count = 0
        for spikes in self._spike_times:
            count += int(np.sum((spikes >= t0) & (spikes < t1)))

        # Rolling z-score then sigmoid
        with self._lock:
            self._history.append(float(count))
            if len(self._history) > self.history_len:
                self._history.pop(0)

            n_hist = len(self._history)
            if n_hist < 10:
                return 0.0   # not enough history yet

            arr = np.array(self._history, dtype=np.float64)
            mu  = arr.mean()
            sig = arr.std() + 1e-6
            z   = (count - mu) / sig

        # sigmoid: z=+3 -> 0.95, z=0 -> 0.50, z=-3 -> 0.05
        rho = float(1.0 / (1.0 + np.exp(-z)))
        return rho

    @property
    def available(self) -> bool:
        return self._available

    def diagnostics(self) -> dict:
        """
        Return spike count statistics for the current session position.
        Useful for threshold tuning.
        """
        if not self._available:
            return {"available": False}

        wall_elapsed = time.time() - self.start_time
        t_bio        = self.start_offset + (wall_elapsed % self.session_window)

        counts = []
        for spikes in self._spike_times:
            counts.append(int(np.sum(
                (spikes >= t_bio) & (spikes < t_bio + self.bin_s)
            )))

        with self._lock:
            hist = list(self._history)

        return {
            "available":      True,
            "t_bio":          round(t_bio, 2),
            "total_count":    sum(counts),
            "per_neuron":     counts,
            "history_len":    len(hist),
            "history_mean":   round(float(np.mean(hist)), 3) if hist else 0,
            "history_std":    round(float(np.std(hist)), 3) if hist else 0,
            "rho":            self.get_rho(),
        }


# =============================================================================
# Module-level singleton and accessor
# =============================================================================
_reader: NeuropixelsReader = None


def get_neuropixels_rho(nwb_path: str = NWB_PATH) -> float:
    """
    Module-level accessor. Lazy-initialises the reader on first call.

    Returns rho in [0, 1]. Falls back to 0.0 silently on any error.
    Safe to call every tick — no file I/O after first call.
    """
    global _reader
    if _reader is None:
        _reader = NeuropixelsReader(nwb_path=nwb_path)
    return _reader.get_rho()


def reset_reader() -> None:
    """Force re-initialisation of the singleton (useful for testing)."""
    global _reader
    _reader = None


# =============================================================================
# Smoke test
# =============================================================================
if __name__ == "__main__":
    print("Testing NeuropixelsReader...")
    reader = NeuropixelsReader()

    if reader.available:
        print("\nWarming history (10 samples needed)...")
        print("Collecting 25 rho samples (1s apart):\n")

        for i in range(25):
            rho   = reader.get_rho()
            diag  = reader.diagnostics()
            count = diag["total_count"]
            bar   = "=" * int(rho * 40)
            print(
                f"  Sample {i+1:>2}: rho={rho:.4f}  "
                f"count={count:>3}  "
                f"t_bio={diag['t_bio']:>8.1f}s  "
                f"|{bar:<40}|"
            )
            time.sleep(1.0)

        print()
        diag = reader.diagnostics()
        print(f"History: n={diag['history_len']}  "
              f"mean={diag['history_mean']}  "
              f"std={diag['history_std']}")
    else:
        print("NWB file unavailable — rho=0.0 fallback confirmed")

    print("\nDone.")
