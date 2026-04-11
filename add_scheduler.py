src = open('train_beats_cardiac.py', encoding='utf-8').read()
old = 'optimizer = torch.optim.Adam(model.parameters(), lr=LR)'
new = '''optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)'''
src = src.replace(old, new)
old2 = "    print(f'Epoch {epoch+1:>3}/{EPOCHS} | AUROC={auroc:.4f}')"
new2 = "    scheduler.step()\n    print(f'Epoch {epoch+1:>3}/{EPOCHS} | AUROC={auroc:.4f} | lr={scheduler.get_last_lr()[0]:.2e}')"
src = src.replace(old2, new2)
open('train_beats_cardiac.py', 'w', encoding='utf-8').write(src)
print('cosine scheduler added')
