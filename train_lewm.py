"""
train_lewm.py — LeWM-style joint encoder+predictor training
No DINOv2 teacher. Encoder and predictor trained jointly.
Loss: L_pred (next-embedding MSE) + lambda_sigreg * L_sigreg

Based on: Maes, Le Lidec, Scieur, LeCun, Balestriero (2026)
"LeWorldModel: Stable End-to-End JEPA from Pixels"
arXiv:2603.19312
"""
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from student_encoder import StudentEncoder, CortexCNNBackbone, ShatteredLatentHead
from latent_predictor import sigreg_loss, temporal_straightening_loss
from train_maze_flow import FlowFrameDataset  # reuse frame loading

def train_lewm(
    data_dir,           # trajectory frames directory
    steps     = 12000,
    batch_size = 32,
    lr         = 1e-3,
    lambda_sigreg = 5.0,  # LeWM uses similar weight
    lambda_curv   = 0.1,  # temporal straightening
    out_dir    = "./checkpoints/lewm",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Encoder + small predictor — trained jointly (LeWM style)
    encoder  = StudentEncoder()
    # Simple 1-step predictor (action-conditioned MLP)
    predictor = nn.Sequential(
        nn.Linear(128 + 2, 256), nn.ReLU(),
        nn.Linear(256, 128)
    )
    
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    
    dataset = FlowFrameDataset(data_dir)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=0, drop_last=True)
    data_iter = iter(loader)
    
    for step in range(1, steps+1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        imgs, imgs_next, actions = batch  # (B,3,224,224), (B,3,224,224), (B,2)
        
        # Joint encode — no stop-gradient, no EMA, no teacher
        z_t    = encoder(imgs)       # (B, 128)
        z_next = encoder(imgs_next)  # (B, 128)  ← encoder sees both frames
        
        # Predict next latent from current + action
        z_pred = predictor(torch.cat([z_t, actions], dim=-1))  # (B, 128)
        
        # LeWM loss 1: next-embedding prediction
        l_pred = F.mse_loss(z_pred, z_next.detach())
        
        # LeWM loss 2: SIGReg (Gaussian regularizer — anti-collapse)
        l_sigreg = sigreg_loss(z_t, z_next)
        
        # Temporal straightening (optional — Appendix H of LeWM)
        # Requires triplet — skip for now or add with 3-frame dataset
        
        loss = l_pred + lambda_sigreg * l_sigreg
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()), 1.0
        )
        optimizer.step()
        scheduler.step()
        
        if step % 100 == 0:
            print(f"Step {step:>6}/{steps} | "
                  f"pred={l_pred.item():.4f}  sigreg={l_sigreg.item():.4f}")
        
        if step % 1000 == 0:
            torch.save({'model': encoder.state_dict(), 'step': step},
                       f"{out_dir}/lewm_encoder_step{step:06d}.pt")
    
    torch.save({'model': encoder.state_dict()},
               f"{out_dir}/lewm_encoder_final.pt")
    print(f"Done → {out_dir}/lewm_encoder_final.pt")