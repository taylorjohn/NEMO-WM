lines = open('train_distillation.py', encoding='utf-8').readlines()
sigreg_block = [
'    # -- SIGReg variant selection ---------------------------------------\n',
'    _sig = getattr(args, "sigreg", "vicreg")\n',
'    if _sig == "weak":\n',
'        import torch as _t\n',
'        def sigreg_loss(z, K=32, **kw):\n',
'            D=z.shape[1]; _t.manual_seed(42)\n',
'            S=_t.randn(D,K,device=z.device)/(K**0.5); S=S/S.norm(dim=0,keepdim=True)\n',
'            sk=z@S; sk_c=sk-sk.mean(0); cov=(sk_c.T@sk_c)/(z.shape[0]-1)\n',
'            return (cov-_t.eye(K,device=z.device)).pow(2).sum()/K\n',
'        print("  SIGReg: Weak-SIGReg K=32")\n',
'    elif _sig == "strong":\n',
'        import torch as _t, torch.nn.functional as _F\n',
'        def sigreg_loss(z, M=16, T=17, **kw):\n',
'            z_n=(z-z.mean(0))/(z.std(0)+1e-6); d=_F.normalize(_t.randn(M,z.shape[1],device=z.device),dim=1)\n',
'            p2=z_n@d.T; t=_t.linspace(-4,4,T,device=z.device); loss=_t.tensor(0.,device=z.device)\n',
'            for m in range(M): loss=loss+(_t.cos(t.unsqueeze(0)*p2[:,m].unsqueeze(1)).mean(0)-_t.exp(-0.5*t**2)).pow(2).mean()\n',
'            return loss/M\n',
'        print("  SIGReg: Strong Epps-Pulley (LeWM)")\n',
'    else: print("  SIGReg: VICReg-style (current)")\n',
'    # -------------------------------------------------------------------\n',
]
# Insert after GC wrapper (after line 281 = index 280)
for i,l in enumerate(lines):
    if 'Optimizer: AdamW baseline' in l:
        insert_at = i + 1
        break
lines = lines[:insert_at] + sigreg_block + lines[insert_at:]
open('train_distillation.py', 'w', encoding='utf-8').writelines(lines)
print(f'Injected SIGReg block after line {insert_at}')
