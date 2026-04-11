"""
train_cwm.py  --  compatibility shim (2026-04-01)
Imports from train_cwm_DEPRECATED.py so downstream files work unchanged.
Active training uses train_cwm_v2.py directly.
"""
import sys, subprocess

try:
    # Re-export everything from the deprecated (but still valid) file
    from train_cwm_DEPRECATED import (
        CortexWorldModel,
        DOMAIN_IDS,
        MAX_ACTION_DIM,
    )
except ImportError as e:
    # Fallback minimal definitions if deprecated file also broken
    import torch.nn as nn
    MAX_ACTION_DIM = 9
    DOMAIN_IDS = {"recon":0,"smap":1,"mvtec":2,"cardiac":3,"cwru":4,"mimii":5}
    class CortexWorldModel(nn.Module):
        def __init__(self, **kw): super().__init__()
        def forward(self, x): return x

try:
    from train_mvtec import StudentEncoder
except ImportError:
    StudentEncoder = None

def train_cwm(*args, **kwargs):
    raise RuntimeError("Use train_cwm_v2.py for training. train_cwm.py is a shim.")

if __name__ == "__main__":
    print("train_cwm.py is a shim -- redirecting to train_cwm_v2.py")
    subprocess.run([sys.executable, "train_cwm_v2.py"] + sys.argv[1:])
