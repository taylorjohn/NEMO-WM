src = open('train_distillation.py', encoding='utf-8').read()
wrapper = '''
    # -- SIGReg variant selection ------------------------------------------
    _sigreg_mode = getattr(args, 'sigreg', 'vicreg')
    if _sigreg_mode == 'weak':
        import torch
        def sigreg_loss(z, K=32, seed=42, eps=1e-4):
            D = z.shape[1]
            torch.manual_seed(seed)
            S = torch.randn(D, K, device=z.device) / (K**0.5)
            S = S / S.norm(dim=0, keepdim=True)
            sk = z @ S
            sk_c = sk - sk.mean(dim=0)
            cov = (sk_c.T @ sk_c) / (z.shape[0]-1)
            import torch.nn.functional as _F
            return (cov - torch.eye(K, device=z.device)).pow(2).sum() / K
        print('  SIGReg: Weak-SIGReg K=32 (Akbar 2026)')
    elif _sigreg_mode == 'strong':
        import torch, torch.nn.functional as _F
        def sigreg_loss(z, M=16, T=17, eps=1e-6):
            z_n = (z - z.mean(0)) / (z.std(0) + eps)
            dirs = _F.normalize(torch.randn(M, z.shape[1], device=z.device), dim=1)
            proj = z_n @ dirs.T
            t = torch.linspace(-4, 4, T, device=z.device)
            loss = torch.tensor(0.0, device=z.device)
            for m in range(M):
                ecf = torch.cos(t.unsqueeze(0) * proj[:,m].unsqueeze(1)).mean(0)
                gcf = torch.exp(-0.5 * t**2)
                loss = loss + (ecf - gcf).pow(2).mean()
            return loss / M
        print('  SIGReg: Strong Epps-Pulley (LeWM/LeJEPA)')
    else:
        print('  SIGReg: VICReg-style (current CORTEX-PE)')
    # ------------------------------------------------------------------------
'''
anchor = '    # -- GC ablation wrapper'
if anchor in src:
    src = src.replace(anchor, wrapper + anchor, 1)
    print('Injected SIGReg dispatcher before GC wrapper')
else:
    anchor2 = '    if gc != chr(39)none chr(39):'
    src = src.replace(anchor2, wrapper + anchor2, 1)
    print('Injected SIGReg dispatcher (alt anchor)')
open('train_distillation.py', 'w', encoding='utf-8').write(src)
