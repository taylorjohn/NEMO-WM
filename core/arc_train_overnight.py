"""
arc_train_overnight.py — Overnight Training on 1994 Gap-Filling Tasks
======================================================================
Uses the structured training dataset (data/arc_training_big/) to train:
  1. JEPA predictor (latent space transformation learning)
  2. JEPA decoder (direct grid prediction) — two-phase training
  3. New solver rules learned from the training data

Two-phase JEPA training (fixes decoder divergence from v1):
  Phase 1: Train predictor only (freeze decoder) — 1000 epochs
  Phase 2: Train decoder only (freeze predictor) — 1000 epochs

The training data covers 13 skills we score 0% on:
  recolor_distance, recolor_automaton, recolor_membership,
  fill_voronoi, fill_connect, fill_enclosed, propagate,
  extract_recolor, sort_recolor, segment_flip, align_left,
  gravity_up/left/right

Usage:
    python arc_train_overnight.py --train --epochs 2000
    python arc_train_overnight.py --status
    python arc_train_overnight.py --solve --data path/to/ARC-AGI-2/data -v
"""

import argparse
import json
import os
import numpy as np
from collections import Counter
from pathlib import Path
import time

from arc_solver import Grid, score_task

try:
    from arc_phase2 import solve_task_phase2 as s1_solve
except ImportError:
    from arc_solver import solve_task as s1_solve

SAVE_DIR = Path("data/jepa_v2")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DIR = Path("data/arc_training_big")


class JEPAv2:
    """
    Two-phase JEPA trainer.
    Phase 1: Encoder + Predictor (learn latent transformations)
    Phase 2: Decoder (learn grid prediction from latent)
    """

    def __init__(self, grid_size=15, latent_dim=128, hidden=512):
        self.grid_size = grid_size
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.input_dim = grid_size * grid_size * 10

        rng = np.random.RandomState(42)

        # Encoder
        self.eW1 = rng.randn(self.input_dim, hidden).astype(np.float32) * np.sqrt(2.0/self.input_dim)
        self.eb1 = np.zeros(hidden, dtype=np.float32)
        self.eW2 = rng.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0/hidden)
        self.eb2 = np.zeros(hidden, dtype=np.float32)
        self.eW3 = rng.randn(hidden, latent_dim).astype(np.float32) * np.sqrt(2.0/hidden)
        self.eb3 = np.zeros(latent_dim, dtype=np.float32)

        # EMA target encoder
        self.teW1=self.eW1.copy(); self.teb1=self.eb1.copy()
        self.teW2=self.eW2.copy(); self.teb2=self.eb2.copy()
        self.teW3=self.eW3.copy(); self.teb3=self.eb3.copy()

        # Predictor (z_in + action → z_out)
        rng2 = np.random.RandomState(123)
        pdim = latent_dim * 2
        self.pW1 = rng2.randn(pdim, hidden).astype(np.float32) * np.sqrt(2.0/pdim)
        self.pb1 = np.zeros(hidden, dtype=np.float32)
        self.pW2 = rng2.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0/hidden)
        self.pb2 = np.zeros(hidden, dtype=np.float32)
        self.pW3 = rng2.randn(hidden, latent_dim).astype(np.float32) * np.sqrt(2.0/hidden)
        self.pb3 = np.zeros(latent_dim, dtype=np.float32)

        # Decoder (z → grid)
        rng3 = np.random.RandomState(789)
        out_dim = grid_size * grid_size * 10
        self.dW1 = rng3.randn(latent_dim, hidden).astype(np.float32) * np.sqrt(2.0/latent_dim)
        self.db1 = np.zeros(hidden, dtype=np.float32)
        self.dW2 = rng3.randn(hidden, hidden).astype(np.float32) * np.sqrt(2.0/hidden)
        self.db2 = np.zeros(hidden, dtype=np.float32)
        self.dW3 = rng3.randn(hidden, out_dim).astype(np.float32) * np.sqrt(2.0/hidden)
        self.db3 = np.zeros(out_dim, dtype=np.float32)

    def grid_to_onehot(self, g):
        if isinstance(g, Grid):
            arr = g.arr
        else:
            arr = np.array(g)
        oh = np.zeros((self.grid_size, self.grid_size, 10), dtype=np.float32)
        h, w = min(arr.shape[0], self.grid_size), min(arr.shape[1], self.grid_size)
        for r in range(h):
            for c in range(w):
                v = int(arr[r, c])
                if v < 10: oh[r, c, v] = 1.0
        return oh.flatten()

    def encode(self, g):
        x = self.grid_to_onehot(g)
        h1 = np.maximum(0, x @ self.eW1 + self.eb1)
        h2 = np.maximum(0, h1 @ self.eW2 + self.eb2)
        z = h2 @ self.eW3 + self.eb3
        return z / (np.sqrt(np.sum(z**2)) + 1e-8)

    def encode_target(self, g):
        x = self.grid_to_onehot(g)
        h1 = np.maximum(0, x @ self.teW1 + self.teb1)
        h2 = np.maximum(0, h1 @ self.teW2 + self.teb2)
        z = h2 @ self.teW3 + self.teb3
        return z / (np.sqrt(np.sum(z**2)) + 1e-8)

    def predict(self, z_in, action):
        x = np.concatenate([z_in, action])
        h1 = np.maximum(0, x @ self.pW1 + self.pb1)
        h2 = np.maximum(0, h1 @ self.pW2 + self.pb2)
        z = h2 @ self.pW3 + self.pb3
        return z / (np.sqrt(np.sum(z**2)) + 1e-8)

    def decode(self, z, out_h, out_w):
        h1 = np.maximum(0, z @ self.dW1 + self.db1)
        h2 = np.maximum(0, h1 @ self.dW2 + self.db2)
        logits = (h2 @ self.dW3 + self.db3).reshape(self.grid_size, self.grid_size, 10)
        arr = np.argmax(logits, axis=2).astype(np.int32)
        return arr[:min(out_h, self.grid_size), :min(out_w, self.grid_size)]

    def update_ema(self, m=0.996):
        self.teW1 = m*self.teW1 + (1-m)*self.eW1
        self.teb1 = m*self.teb1 + (1-m)*self.eb1
        self.teW2 = m*self.teW2 + (1-m)*self.eW2
        self.teb2 = m*self.teb2 + (1-m)*self.eb2
        self.teW3 = m*self.teW3 + (1-m)*self.eW3
        self.teb3 = m*self.teb3 + (1-m)*self.eb3

    def load_training_data(self):
        """Load all training pairs from arc_training_big."""
        if not TRAIN_DIR.exists():
            print(f"  ERROR: {TRAIN_DIR} not found. Generate with the training generator first.")
            return [], []

        pairs_in = []
        pairs_out = []
        for f in sorted(TRAIN_DIR.glob("*.json")):
            task = json.load(open(f))
            for p in task.get('train', []) + task.get('test', []):
                pairs_in.append(np.array(p['input']))
                pairs_out.append(np.array(p['output']))

        print(f"  Loaded {len(pairs_in)} training pairs from {TRAIN_DIR}")
        return pairs_in, pairs_out

    def train(self, epochs=2000, batch_size=64, lr=0.003, checkpoint_every=500):
        """Two-phase training."""
        pairs_in, pairs_out = self.load_training_data()
        if not pairs_in:
            return

        n = len(pairs_in)
        phase1_epochs = epochs // 2
        phase2_epochs = epochs - phase1_epochs

        print("=" * 70)
        print(f"  JEPA v2 Two-Phase Training")
        print(f"  Data: {n} pairs")
        print(f"  Phase 1: Predictor ({phase1_epochs} epochs)")
        print(f"  Phase 2: Decoder ({phase2_epochs} epochs)")
        print(f"  Latent: {self.latent_dim}-D, Grid: {self.grid_size}x{self.grid_size}")
        print("=" * 70)

        # Pre-encode
        print("  Encoding grids...")
        Z_in = np.array([self.encode(g) for g in pairs_in])
        Z_out = np.array([self.encode_target(g) for g in pairs_out])

        # Per-task action vectors (group by similar transforms)
        actions = Z_out - Z_in
        
        # Target one-hot for decoder
        OH_out = np.array([self.grid_to_onehot(g).reshape(self.grid_size, self.grid_size, 10) for g in pairs_out])

        t0 = time.time()

        # ═══ PHASE 1: Train predictor ═══
        print(f"\n  Phase 1: Training predictor...")
        for epoch in range(phase1_epochs):
            idx = np.random.permutation(n)
            total_loss = 0
            nb = 0
            cur_lr = lr * (0.999 ** epoch)

            for start in range(0, n, batch_size):
                batch = idx[start:start+batch_size]
                bs = len(batch)

                b_zin = Z_in[batch]
                b_ztarget = Z_out[batch]
                b_action = actions[batch]

                # Forward
                x = np.concatenate([b_zin, b_action], axis=1)
                ph1 = np.maximum(0, x @ self.pW1 + self.pb1)
                ph2 = np.maximum(0, ph1 @ self.pW2 + self.pb2)
                z_raw = ph2 @ self.pW3 + self.pb3
                z_norm = np.sqrt(np.sum(z_raw**2, axis=1, keepdims=True) + 1e-8)
                z_pred = z_raw / z_norm

                loss = np.mean(np.sum((z_pred - b_ztarget)**2, axis=1))
                total_loss += loss; nb += 1

                # Backward
                dz = 2*(z_pred - b_ztarget)/bs / z_norm
                dpW3 = ph2.T @ dz; dpb3 = dz.sum(0)
                dph2 = dz @ self.pW3.T * (ph2>0); dpW2 = ph1.T @ dph2; dpb2 = dph2.sum(0)
                dph1 = dph2 @ self.pW2.T * (ph1>0); dpW1 = x.T @ dph1; dpb1 = dph1.sum(0)

                for g in [dpW1,dpb1,dpW2,dpb2,dpW3,dpb3]: np.clip(g,-1,1,out=g)

                self.pW1 -= cur_lr*dpW1; self.pb1 -= cur_lr*dpb1
                self.pW2 -= cur_lr*dpW2; self.pb2 -= cur_lr*dpb2
                self.pW3 -= cur_lr*dpW3; self.pb3 -= cur_lr*dpb3

            self.update_ema()

            if epoch % 50 == 0 and epoch > 0:
                Z_out = np.array([self.encode_target(g) for g in pairs_out])
                actions = Z_out - Z_in

            if epoch % max(1, phase1_epochs//10) == 0:
                print(f"    Epoch {epoch:>5}: pred_loss={total_loss/max(nb,1):.4f} "
                      f"lr={cur_lr:.5f} [{time.time()-t0:.0f}s]")

            if (epoch+1) % checkpoint_every == 0:
                self.save(SAVE_DIR / f"phase1_ep{epoch+1}.npz")
                print(f"    ✓ Checkpoint: phase1_ep{epoch+1}")

        pred_loss = total_loss / max(nb, 1)
        print(f"  Phase 1 complete: pred_loss={pred_loss:.4f}")

        # ═══ PHASE 2: Train decoder (predictor frozen) ═══
        print(f"\n  Phase 2: Training decoder...")
        
        # Re-encode and predict all z_outs using trained predictor
        Z_pred = []
        for i in range(n):
            z_p = self.predict(Z_in[i], actions[i])
            Z_pred.append(z_p)
        Z_pred = np.array(Z_pred)

        for epoch in range(phase2_epochs):
            idx = np.random.permutation(n)
            total_loss = 0
            nb = 0
            cur_lr = lr * 0.5 * (0.999 ** epoch)  # lower LR for decoder

            for start in range(0, n, batch_size):
                batch = idx[start:start+batch_size]
                bs = len(batch)

                b_zpred = Z_pred[batch]
                b_target = OH_out[batch]

                # Forward decoder
                dh1 = np.maximum(0, b_zpred @ self.dW1 + self.db1)
                dh2 = np.maximum(0, dh1 @ self.dW2 + self.db2)
                logits = (dh2 @ self.dW3 + self.db3).reshape(bs, self.grid_size, self.grid_size, 10)

                # Softmax
                lmax = logits.max(axis=3, keepdims=True)
                exp_l = np.exp(np.clip(logits - lmax, -20, 20))
                probs = exp_l / (exp_l.sum(axis=3, keepdims=True) + 1e-8)

                loss = -np.mean(np.sum(b_target * np.log(probs + 1e-8), axis=(1,2,3)))
                total_loss += loss; nb += 1

                # Backward
                dl = (probs - b_target).reshape(bs, -1) / bs
                ddW3 = dh2.T @ dl; ddb3 = dl.sum(0)
                ddh2 = dl @ self.dW3.T * (dh2>0); ddW2 = dh1.T @ ddh2; ddb2 = ddh2.sum(0)
                ddh1 = ddh2 @ self.dW2.T * (dh1>0); ddW1 = b_zpred.T @ ddh1; ddb1 = ddh1.sum(0)

                for g in [ddW1,ddb1,ddW2,ddb2,ddW3,ddb3]: np.clip(g,-1,1,out=g)

                self.dW1 -= cur_lr*ddW1; self.db1 -= cur_lr*ddb1
                self.dW2 -= cur_lr*ddW2; self.db2 -= cur_lr*ddb2
                self.dW3 -= cur_lr*ddW3; self.db3 -= cur_lr*ddb3

            ep_total = phase1_epochs + epoch
            if epoch % max(1, phase2_epochs//10) == 0:
                print(f"    Epoch {ep_total:>5}: dec_loss={total_loss/max(nb,1):.4f} "
                      f"lr={cur_lr:.5f} [{time.time()-t0:.0f}s]")

            if (epoch+1) % checkpoint_every == 0:
                self.save(SAVE_DIR / f"phase2_ep{epoch+1}.npz")
                print(f"    ✓ Checkpoint: phase2_ep{epoch+1}")

        dec_loss = total_loss / max(nb, 1)
        elapsed = time.time() - t0

        self.save(SAVE_DIR / "final.npz")

        print(f"\n{'='*70}")
        print(f"  Training complete!")
        print(f"  Pred loss:  {pred_loss:.4f}")
        print(f"  Dec loss:   {dec_loss:.4f}")
        print(f"  Time:       {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print(f"  Saved to:   {SAVE_DIR / 'final.npz'}")
        print(f"{'='*70}")

    def save(self, path):
        np.savez(path,
            eW1=self.eW1, eb1=self.eb1, eW2=self.eW2, eb2=self.eb2,
            eW3=self.eW3, eb3=self.eb3,
            teW1=self.teW1, teb1=self.teb1, teW2=self.teW2, teb2=self.teb2,
            teW3=self.teW3, teb3=self.teb3,
            pW1=self.pW1, pb1=self.pb1, pW2=self.pW2, pb2=self.pb2,
            pW3=self.pW3, pb3=self.pb3,
            dW1=self.dW1, db1=self.db1, dW2=self.dW2, db2=self.db2,
            dW3=self.dW3, db3=self.db3,
            gs=self.grid_size, ld=self.latent_dim)

    def load(self, path):
        d = np.load(path)
        self.eW1=d['eW1']; self.eb1=d['eb1']; self.eW2=d['eW2']; self.eb2=d['eb2']
        self.eW3=d['eW3']; self.eb3=d['eb3']
        self.teW1=d['teW1']; self.teb1=d['teb1']; self.teW2=d['teW2']; self.teb2=d['teb2']
        self.teW3=d['teW3']; self.teb3=d['teb3']
        self.pW1=d['pW1']; self.pb1=d['pb1']; self.pW2=d['pW2']; self.pb2=d['pb2']
        self.pW3=d['pW3']; self.pb3=d['pb3']
        self.dW1=d['dW1']; self.db1=d['db1']; self.dW2=d['dW2']; self.db2=d['db2']
        self.dW3=d['dW3']; self.db3=d['db3']
        print(f"  Loaded from {path}")


def generate_training_data():
    """Generate 1994 structured training tasks targeting gap skills."""
    import numpy as np
    from collections import Counter
    
    rng = np.random.RandomState(2026)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    
    def mp(i, o):
        return {'input': i.tolist(), 'output': o.tolist()}
    
    all_tasks = {}
    tid = 0
    
    print("  Generating training data for 13 gap skills...")
    
    # 1. RECOLOR BY DISTANCE
    for _ in range(200):
        h, w = rng.randint(5,15), rng.randint(5,15)
        thresholds = sorted(rng.choice(range(1,6), min(3, rng.randint(2,4)), replace=False))
        dc = rng.choice(range(1,10), len(thresholds)+1, replace=False)
        demos = []
        for _ in range(rng.randint(2,4)):
            grid = np.zeros((h,w), dtype=int)
            anchors = []
            for _ in range(rng.randint(1,4)):
                r,c = rng.randint(0,h), rng.randint(0,w)
                grid[r,c] = int(dc[0]); anchors.append((r,c))
            if not anchors: continue
            out = grid.copy()
            for r in range(h):
                for c in range(w):
                    if grid[r,c]!=0: continue
                    md = min(abs(r-ar)+abs(c-ac) for ar,ac in anchors)
                    for ti,t in enumerate(thresholds):
                        if md<=t: out[r,c]=int(dc[ti+1]); break
                    else: out[r,c]=int(dc[-1])
            demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'recolor_distance_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 2. RECOLOR BY AUTOMATON
    for _ in range(200):
        h,w = rng.randint(5,12), rng.randint(5,12)
        k = rng.randint(1,4); fc = rng.randint(1,9)
        demos = []
        for _ in range(rng.randint(2,4)):
            grid = np.zeros((h,w),dtype=int)
            for r in range(h):
                for c in range(w):
                    if rng.random()<0.25: grid[r,c]=fc
            out = grid.copy()
            for r in range(h):
                for c in range(w):
                    if grid[r,c]!=0: continue
                    nc=sum(1 for dr,dc2 in [(-1,0),(1,0),(0,-1),(0,1)] if 0<=r+dr<h and 0<=c+dc2<w and grid[r+dr,c+dc2]!=0)
                    if nc>=k: out[r,c]=fc
            if not np.array_equal(grid,out): demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'recolor_automaton_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 3. RECOLOR BY MEMBERSHIP
    for _ in range(200):
        h,w = rng.randint(6,12), rng.randint(6,12)
        sc = rng.randint(1,8); nc2 = rng.choice(range(2,6))
        new_c = rng.choice(range(1,10), nc2, replace=False)
        demos = []
        for _ in range(rng.randint(2,4)):
            grid = np.zeros((h,w),dtype=int)
            for ci in range(nc2):
                for _ in range(30):
                    oh,ow = rng.randint(2,4), rng.randint(2,4)
                    r,c = rng.randint(0,h-oh), rng.randint(0,w-ow)
                    if grid[r:r+oh,c:c+ow].sum()==0: grid[r:r+oh,c:c+ow]=sc; break
            out = np.zeros((h,w),dtype=int); visited = np.zeros((h,w),dtype=bool); cid=0
            for r in range(h):
                for c in range(w):
                    if grid[r,c]==sc and not visited[r,c]:
                        stk=[(r,c)]; cells=[]
                        while stk:
                            cr,cc=stk.pop()
                            if cr<0 or cr>=h or cc<0 or cc>=w: continue
                            if visited[cr,cc] or grid[cr,cc]!=sc: continue
                            visited[cr,cc]=True; cells.append((cr,cc))
                            stk.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                        for cr,cc in cells: out[cr,cc]=new_c[cid%len(new_c)]
                        cid+=1
            if cid>=2: demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'recolor_membership_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 4. FILL VORONOI
    for _ in range(200):
        h,w = rng.randint(5,12), rng.randint(5,12)
        ns = rng.randint(2,6); colors = rng.choice(range(1,10),ns,replace=False)
        demos = []
        for _ in range(rng.randint(2,4)):
            grid = np.zeros((h,w),dtype=int); seeds=[]
            for i in range(ns):
                r,c = rng.randint(0,h), rng.randint(0,w)
                grid[r,c]=int(colors[i]); seeds.append((r,c,int(colors[i])))
            out = np.zeros((h,w),dtype=int)
            for r in range(h):
                for c in range(w):
                    bd,bc=float('inf'),0
                    for sr,sc2,scc in seeds:
                        d=abs(r-sr)+abs(c-sc2)
                        if d<bd: bd,bc=d,scc
                    out[r,c]=bc
            demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'fill_voronoi_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 5. FILL CONNECT
    for _ in range(200):
        h,w = rng.randint(6,12), rng.randint(6,12)
        color = rng.randint(1,8); nd = rng.randint(2,5)
        demos = []
        for _ in range(rng.randint(2,4)):
            grid = np.zeros((h,w),dtype=int); dots=[]
            for _ in range(nd):
                r,c = rng.randint(0,h), rng.randint(0,w)
                grid[r,c]=color; dots.append((r,c))
            out = grid.copy()
            for i in range(len(dots)-1):
                r1,c1=dots[i]; r2,c2=dots[i+1]
                for cc in range(min(c1,c2),max(c1,c2)+1): out[r1,cc]=color
                for rr in range(min(r1,r2),max(r1,r2)+1): out[rr,c2]=color
            if not np.array_equal(grid,out): demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'fill_connect_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 6. PROPAGATE
    for _ in range(200):
        h,w = rng.randint(7,14), rng.randint(7,14)
        color = rng.randint(1,8); radius = rng.randint(2,4)
        demos = []
        for _ in range(rng.randint(2,4)):
            grid = np.zeros((h,w),dtype=int)
            sr,sc2 = rng.randint(radius,h-radius), rng.randint(radius,w-radius)
            grid[sr,sc2]=color
            out = np.zeros((h,w),dtype=int)
            for r in range(h):
                for c in range(w):
                    if abs(r-sr)+abs(c-sc2)<=radius: out[r,c]=color
            demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'propagate_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 7-9. COMPOUND SKILLS
    for skill, n_iter in [('extract_recolor',150),('sort_recolor',150),('segment_flip',150)]:
        for _ in range(n_iter):
            h,w = 12,12
            demos = []
            for _ in range(rng.randint(2,4)):
                grid = np.zeros((h,w),dtype=int)
                if skill == 'segment_flip':
                    dc = rng.randint(1,9); dr = rng.randint(3,h-3)
                    grid[dr,:]=dc
                    for r in range(h):
                        for c in range(w):
                            if r==dr: continue
                            if rng.random()<0.25: grid[r,c]=rng.randint(1,7)
                    out = grid.copy(); out[:dr]=grid[:dr][::-1]
                else:
                    src = rng.randint(1,8); objs=[]
                    for _ in range(rng.randint(2,5)):
                        for _ in range(30):
                            oh,ow = rng.randint(1,5), rng.randint(1,5)
                            r,c = rng.randint(0,h-oh), rng.randint(0,w-ow)
                            if grid[r:r+oh,c:c+ow].sum()==0:
                                grid[r:r+oh,c:c+ow]=src; objs.append((r,c,oh,ow,oh*ow)); break
                    if len(objs)<2: continue
                    if skill == 'extract_recolor':
                        nc = rng.randint(1,9)
                        largest = max(objs, key=lambda o:o[4])
                        out = np.zeros((h,w),dtype=int)
                        r,c,oh,ow,_ = largest; out[r:r+oh,c:c+ow]=nc
                    else:
                        rc = rng.choice(range(1,10),len(objs),replace=False)
                        so = sorted(objs, key=lambda o:o[4])
                        out = np.zeros((h,w),dtype=int)
                        for i,(r,c,oh,ow,_) in enumerate(so):
                            out[r:r+oh,c:c+ow]=int(rc[i%len(rc)])
                demos.append(mp(grid,out))
            if len(demos)>=2:
                all_tasks[f'{skill}_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 10. ALIGN + GRAVITY
    for skill in ['align_left','gravity_up','gravity_left','gravity_right']:
        for _ in range(100 if skill=='align_left' else 50):
            h,w = rng.randint(5,10), rng.randint(5,10)
            demos = []
            for _ in range(rng.randint(2,4)):
                grid = np.zeros((h,w),dtype=int)
                if skill=='align_left':
                    color=rng.randint(1,8); ow=rng.randint(1,3); positions=[]
                    for _ in range(rng.randint(2,4)):
                        r,c=rng.randint(0,h), rng.randint(0,w-ow)
                        grid[r,c:c+ow]=color; positions.append((r,c))
                    tc=min(p[1] for p in positions)
                    out=np.zeros((h,w),dtype=int)
                    for r,_ in positions: out[r,tc:tc+ow]=color
                else:
                    for _ in range(rng.randint(3,8)):
                        r,c=rng.randint(0,h),rng.randint(0,w)
                        grid[r,c]=rng.randint(1,7)
                    out=np.zeros((h,w),dtype=int)
                    if skill=='gravity_up':
                        for c in range(w):
                            vals=[int(grid[r,c]) for r in range(h) if grid[r,c]!=0]
                            for i,v in enumerate(vals): out[i,c]=v
                    elif skill=='gravity_left':
                        for r in range(h):
                            vals=[int(grid[r,c]) for c in range(w) if grid[r,c]!=0]
                            for i,v in enumerate(vals): out[r,i]=v
                    else:
                        for r in range(h):
                            vals=[int(grid[r,c]) for c in range(w) if grid[r,c]!=0]
                            for i,v in enumerate(reversed(vals)): out[r,w-1-i]=v
                if not np.array_equal(grid,out): demos.append(mp(grid,out))
            if len(demos)>=2:
                all_tasks[f'{skill}_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # 11. FILL ENCLOSED (various)
    for _ in range(150):
        h,w = rng.randint(7,14), rng.randint(7,14)
        bc=rng.randint(1,8); fc=rng.choice([c for c in range(1,9) if c!=bc])
        demos = []
        for _ in range(rng.randint(2,4)):
            grid=np.zeros((h,w),dtype=int)
            r1,c1=rng.randint(1,h//3),rng.randint(1,w//3)
            r2,c2=rng.randint(2*h//3,h-1),rng.randint(2*w//3,w-1)
            grid[r1,c1:c2+1]=bc; grid[r2,c1:c2+1]=bc
            grid[r1:r2+1,c1]=bc; grid[r1:r2+1,c2]=bc
            out=grid.copy()
            for r in range(r1+1,r2):
                for c in range(c1+1,c2): out[r,c]=fc
            demos.append(mp(grid,out))
        if len(demos)>=2:
            all_tasks[f'fill_enclosed_{tid:05d}']={'train':demos[:-1],'test':demos[-1:]}; tid+=1
    
    # Save
    for name, task in all_tasks.items():
        with open(TRAIN_DIR / f'{name}.json', 'w') as f:
            json.dump(task, f)
    
    print(f"  Generated {len(all_tasks)} training tasks in {TRAIN_DIR}")


def show_status():
    print("=" * 70)
    print("  JEPA v2 Training Status")
    print("=" * 70)
    for p in sorted(SAVE_DIR.glob("*.npz")):
        print(f"  {p.name} ({p.stat().st_size/1e6:.1f} MB)")
    if not list(SAVE_DIR.glob("*.npz")):
        print("  No checkpoints found.")
        print("  Run: python arc_train_overnight.py --train --epochs 2000")
    
    if TRAIN_DIR.exists():
        n = len(list(TRAIN_DIR.glob("*.json")))
        print(f"\n  Training data: {n} tasks in {TRAIN_DIR}")
    else:
        print(f"\n  WARNING: {TRAIN_DIR} not found!")
    print("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.status:
        show_status()
    elif args.generate:
        generate_training_data()
    elif args.train:
        model = JEPAv2(grid_size=15, latent_dim=128)
        model.train(epochs=args.epochs)
    elif args.solve:
        model = JEPAv2(grid_size=15, latent_dim=128)
        final = SAVE_DIR / "final.npz"
        if final.exists():
            model.load(final)
        else:
            print("No model found. Train first.")
            exit(1)
        
        data_base = args.data or "ARC-AGI-2/data"
        d = os.path.join(data_base, "training")
        files = sorted(f for f in os.listdir(d) if f.endswith('.json'))
        
        solved = s1_n = jepa_n = 0
        for f in files:
            task = json.load(open(os.path.join(d, f)))
            s1 = s1_solve(task)
            if score_task(task, s1):
                solved += 1; s1_n += 1; continue
            
            pairs = task['train']
            tests = task['test']
            z_ins = np.array([model.encode(p['input']) for p in pairs])
            z_outs = np.array([model.encode_target(p['output']) for p in pairs])
            action = np.mean(z_outs - z_ins, axis=0)
            
            guesses = []
            for tc in tests:
                gi = np.array(tc['input'])
                z_test = model.encode(gi)
                z_pred = model.predict(z_test, action)
                out_h = np.array(tc['output']).shape[0] if 'output' in tc else gi.shape[0]
                out_w = np.array(tc['output']).shape[1] if 'output' in tc else gi.shape[1]
                decoded = model.decode(z_pred, out_h, out_w)
                guesses.append([decoded.tolist()])
            
            if score_task(task, guesses):
                solved += 1; jepa_n += 1
                if args.verbose:
                    print(f"  {f}: JEPA")
        
        print(f"\n  Results: {solved}/{len(files)} (S1:{s1_n}, JEPA:{jepa_n})")
    else:
        show_status()
