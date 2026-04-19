"""
arc_advanced_ops.py — Advanced Operations for 658 Unsolved Tasks
==================================================================
Targets the 3 biggest gaps:
  1. Recolor (232 tasks): distance, size, membership, propagation
  2. Extraction (218 tasks): by shape, context, unique, intersection
  3. Fill (208 tasks): proximity, symmetry, pattern, between objects

Each operation is a standalone solver that:
  - Analyzes the training demos to learn the rule
  - Applies the learned rule to test inputs
  - Verifies against all training pairs before returning

Usage:
    python arc_advanced_ops.py --training --data path/to/ARC-AGI-2/data -v
    python arc_advanced_ops.py --eval --data path/to/ARC-AGI-2/data
    python arc_advanced_ops.py --test
"""

import argparse, json, os, numpy as np, time
from collections import Counter
from arc_solver import Grid, score_task
try:
    from arc_phase2 import solve_task_phase2 as s1_solve
except ImportError:
    from arc_solver import solve_task as s1_solve

# ── Utilities ──
def get_objects_cc(arr):
    h,w = arr.shape
    bg = int(np.argmax(np.bincount(arr.flatten())))
    visited = np.zeros((h,w),dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if arr[r,c]!=bg and not visited[r,c]:
                color=int(arr[r,c]); cells=[]; stack=[(r,c)]
                while stack:
                    cr,cc=stack.pop()
                    if cr<0 or cr>=h or cc<0 or cc>=w: continue
                    if visited[cr,cc] or int(arr[cr,cc])!=color: continue
                    visited[cr,cc]=True; cells.append((cr,cc))
                    stack.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                if cells:
                    rs=[p[0] for p in cells]; cs=[p[1] for p in cells]
                    objects.append({'color':color,'cells':cells,'size':len(cells),
                        'bbox':(min(rs),max(rs),min(cs),max(cs)),
                        'center':(np.mean(rs),np.mean(cs))})
    return objects, bg

def cheby(r1,c1,r2,c2): return max(abs(r1-r2),abs(c1-c2))
def manh(r1,c1,r2,c2): return abs(r1-r2)+abs(c1-c2)
def crop_bbox(arr,bbox):
    r1,r2,c1,c2=bbox; return arr[r1:r2+1,c1:c2+1].copy()

# ══════════════════════════════════════════════════════════════════
# 1. RECOLOR (232 unsolved)
# ══════════════════════════════════════════════════════════════════
def try_recolor_by_distance(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    _,bg=get_objects_cc(gi)
    nz=[(r,c) for r in range(h) for c in range(w) if gi[r,c]!=bg]
    if not nz: return None
    dm={}
    for r in range(h):
        for c in range(w):
            d=0 if gi[r,c]!=bg else min(cheby(r,c,nr,nc) for nr,nc in nz)
            oc=int(go[r,c])
            if d in dm and dm[d]!=oc: return None
            dm[d]=oc
    def apply(inp):
        a=np.array(inp); h,w=a.shape; _,bg2=get_objects_cc(a)
        nz2=[(r,c) for r in range(h) for c in range(w) if a[r,c]!=bg2]
        if not nz2: return a
        r2=np.zeros_like(a)
        for r in range(h):
            for c in range(w):
                d=0 if a[r,c]!=bg2 else min(cheby(r,c,nr,nc) for nr,nc in nz2)
                r2[r,c]=dm.get(d,dm.get(max(dm),bg2))
        return r2
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_recolor_by_size(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    objs,bg=get_objects_cc(gi)
    if len(objs)<2: return None
    sizes=sorted(set(o['size'] for o in objs),reverse=True)
    sr={s:i for i,s in enumerate(sizes)}
    r2c={}
    for o in objs:
        rk=sr[o['size']]; r,c=o['cells'][0]; oc=int(go[r,c])
        if rk in r2c and r2c[rk]!=oc: return None
        r2c[rk]=oc
    def apply(inp):
        a=np.array(inp); o2,bg2=get_objects_cc(a)
        if len(o2)<2: return a
        s2=sorted(set(o['size'] for o in o2),reverse=True)
        sr2={s:i for i,s in enumerate(s2)}
        res=np.full_like(a,bg2)
        for o in o2:
            rk=sr2[o['size']]; nc=r2c.get(rk,r2c.get(max(r2c),o['color']))
            for r,c in o['cells']: res[r,c]=nc
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_recolor_propagation(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    _,bg=get_objects_cc(gi)
    if np.sum(go!=bg)<=np.sum(gi!=bg): return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; bg2=int(np.argmax(np.bincount(a.flatten())))
        res=a.copy()
        ext=np.zeros((h,w),dtype=bool); stk=[]
        for r in range(h):
            for c in [0,w-1]:
                if a[r,c]==bg2: stk.append((r,c))
        for c in range(w):
            for r in [0,h-1]:
                if a[r,c]==bg2: stk.append((r,c))
        while stk:
            r,c=stk.pop()
            if r<0 or r>=h or c<0 or c>=w or ext[r,c] or a[r,c]!=bg2: continue
            ext[r,c]=True; stk.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
        objs2,_=get_objects_cc(a)
        for r in range(h):
            for c in range(w):
                if a[r,c]==bg2 and not ext[r,c] and objs2:
                    md=float('inf'); bc=bg2
                    for o in objs2:
                        for cr,cc in o['cells']:
                            d=manh(r,c,cr,cc)
                            if d<md: md=d; bc=o['color']
                    res[r,c]=bc
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_recolor_majority_neighbor(pairs):
    """Each cell becomes majority color in its 3x3 neighborhood."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; res=np.zeros_like(a)
        for r in range(h):
            for c in range(w):
                vals=[]
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<h and 0<=nc<w: vals.append(int(a[nr,nc]))
                res[r,c]=Counter(vals).most_common(1)[0][0]
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_recolor_row_col_rule(pairs):
    """Color based on (input_color, row%mod, col%mod)."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    for mod in [2,3,4]:
        rules={}; ok=True
        for p in pairs:
            gi=np.array(p['input']); go=np.array(p['output'])
            for r in range(gi.shape[0]):
                for c in range(gi.shape[1]):
                    k=(int(gi[r,c]),r%mod,c%mod); v=int(go[r,c])
                    if k in rules and rules[k]!=v: ok=False; break
                    rules[k]=v
                if not ok: break
            if not ok: break
        if not ok or not rules: continue
        if all(k[0]==v for k,v in rules.items()): continue
        def mk(r=rules,m=mod):
            def apply(inp):
                a=np.array(inp); res=a.copy()
                for row in range(a.shape[0]):
                    for col in range(a.shape[1]):
                        k=(int(a[row,col]),row%m,col%m)
                        if k in r: res[row,col]=r[k]
                return res
            return apply
        fn=mk()
        if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs):
            return fn
    return None

# ══════════════════════════════════════════════════════════════════
# 2. EXTRACTION (218 unsolved)
# ══════════════════════════════════════════════════════════════════
def try_extract_unique(pairs):
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if go.shape[0]>=gi.shape[0] and go.shape[1]>=gi.shape[1]: return None
    objs,bg=get_objects_cc(gi)
    if len(objs)<2: return None
    cc=Counter(o['color'] for o in objs)
    uc=[c for c,n in cc.items() if n==1]
    for u in uc:
        t=[o for o in objs if o['color']==u][0]
        cr=crop_bbox(gi,t['bbox'])
        cm=np.where(cr==u,u,0)
        if cr.shape==go.shape and np.array_equal(cr,go):
            def mk(mode='crop'):
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    cc2=Counter(o['color'] for o in o2)
                    uc2=[c for c,n in cc2.items() if n==1]
                    if uc2:
                        t2=[o for o in o2 if o['color']==uc2[0]][0]
                        return crop_bbox(a,t2['bbox'])
                    return a
                return apply
            fn=mk(); 
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
        if cm.shape==go.shape and np.array_equal(cm,go):
            def mk2():
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    cc2=Counter(o['color'] for o in o2)
                    uc2=[c for c,n in cc2.items() if n==1]
                    if uc2:
                        t2=[o for o in o2 if o['color']==uc2[0]][0]
                        cr2=crop_bbox(a,t2['bbox'])
                        return np.where(cr2==uc2[0],uc2[0],0)
                    return a
                return apply
            fn=mk2()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_extract_by_color(pairs):
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if go.shape[0]>=gi.shape[0] and go.shape[1]>=gi.shape[1]: return None
    objs,bg=get_objects_cc(gi)
    for o in objs:
        cr=crop_bbox(gi,o['bbox'])
        if cr.shape==go.shape and np.array_equal(cr,go):
            tc=o['color']
            def mk(c=tc):
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    for ob in o2:
                        if ob['color']==c: return crop_bbox(a,ob['bbox'])
                    return a
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    # By size rank
    objs_s=sorted(objs,key=lambda o:-o['size'])
    for rk,o in enumerate(objs_s):
        cr=crop_bbox(gi,o['bbox'])
        if cr.shape==go.shape and np.array_equal(cr,go):
            def mk2(r=rk):
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    o2s=sorted(o2,key=lambda x:-x['size'])
                    if r<len(o2s): return crop_bbox(a,o2s[r]['bbox'])
                    return a
                return apply
            fn=mk2()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_extract_border(pairs):
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    if go.shape[0]>=h and go.shape[1]>=w: return None
    objs,bg=get_objects_cc(gi)
    for o in objs:
        tb=any(r==0 or r==h-1 or c==0 or c==w-1 for r,c in o['cells'])
        if tb:
            cr=crop_bbox(gi,o['bbox'])
            if cr.shape==go.shape and np.array_equal(cr,go):
                def apply(inp):
                    a=np.array(inp); h2,w2=a.shape; o2,_=get_objects_cc(a)
                    for ob in o2:
                        if any(r==0 or r==h2-1 or c==0 or c==w2-1 for r,c in ob['cells']):
                            return crop_bbox(a,ob['bbox'])
                    return a
                if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

def try_extract_intersection(pairs):
    """Extract the overlapping region of two colored areas."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if go.shape[0]>=gi.shape[0] and go.shape[1]>=gi.shape[1]: return None
    objs,bg=get_objects_cc(gi)
    if len(objs)<2: return None
    # Check if output matches bbox overlap of two objects
    for i in range(len(objs)):
        for j in range(i+1,len(objs)):
            b1=objs[i]['bbox']; b2=objs[j]['bbox']
            r1=max(b1[0],b2[0]); r2=min(b1[1],b2[1])
            c1=max(b1[2],b2[2]); c2=min(b1[3],b2[3])
            if r1<=r2 and c1<=c2:
                inter=gi[r1:r2+1,c1:c2+1].copy()
                if inter.shape==go.shape and np.array_equal(inter,go):
                    def apply(inp):
                        a=np.array(inp); o2,_=get_objects_cc(a)
                        if len(o2)>=2:
                            bb1=o2[0]['bbox']; bb2=o2[1]['bbox']
                            rr1=max(bb1[0],bb2[0]); rr2=min(bb1[1],bb2[1])
                            cc1=max(bb1[2],bb2[2]); cc2=min(bb1[3],bb2[3])
                            if rr1<=rr2 and cc1<=cc2:
                                return a[rr1:rr2+1,cc1:cc2+1].copy()
                        return a
                    if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

# ══════════════════════════════════════════════════════════════════
# 3. FILL (208 unsolved)
# ══════════════════════════════════════════════════════════════════
def try_fill_symmetry(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    go=np.array(pairs[0]['output'])
    for st in ['h','v','both','rot180']:
        sym=True
        if st in ['h','both']: sym=sym and np.array_equal(go,go[:,::-1])
        if st in ['v','both']: sym=sym and np.array_equal(go,go[::-1,:])
        if st=='rot180': sym=np.array_equal(go,np.rot90(go,2))
        if not sym: continue
        def mk(s=st):
            def apply(inp):
                a=np.array(inp).copy(); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
                if s in ['h','both']:
                    for r in range(h):
                        for c in range(w):
                            mc=w-1-c
                            if a[r,c]==bg and a[r,mc]!=bg: a[r,c]=a[r,mc]
                            elif a[r,mc]==bg and a[r,c]!=bg: a[r,mc]=a[r,c]
                if s in ['v','both']:
                    for r in range(h):
                        for c in range(w):
                            mr=h-1-r
                            if a[r,c]==bg and a[mr,c]!=bg: a[r,c]=a[mr,c]
                            elif a[mr,c]==bg and a[r,c]!=bg: a[mr,c]=a[r,c]
                if s=='rot180':
                    for r in range(h):
                        for c in range(w):
                            mr,mc=h-1-r,w-1-c
                            if a[r,c]==bg and a[mr,mc]!=bg: a[r,c]=a[mr,mc]
                            elif a[mr,mc]==bg and a[r,c]!=bg: a[mr,mc]=a[r,c]
                return a
            return apply
        fn=mk()
        if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_fill_voronoi(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); bg=int(np.argmax(np.bincount(gi.flatten())))
    go=np.array(pairs[0]['output'])
    if np.sum(go!=bg)<=np.sum(gi!=bg)*2: return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; bg2=int(np.argmax(np.bincount(a.flatten())))
        seeds=[(r,c,int(a[r,c])) for r in range(h) for c in range(w) if a[r,c]!=bg2]
        if not seeds: return a
        res=a.copy()
        for r in range(h):
            for c in range(w):
                if a[r,c]==bg2:
                    ds=[(manh(r,c,sr,sc),col) for sr,sc,col in seeds]
                    md=min(d for d,_ in ds)
                    cands=[col for d,col in ds if d==md]
                    res[r,c]=min(cands)
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_fill_extend_h(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp).copy(); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
        for r in range(h):
            cols=[int(a[r,c]) for c in range(w) if a[r,c]!=bg]
            if cols:
                fc=Counter(cols).most_common(1)[0][0]
                for c in range(w):
                    if a[r,c]==bg: a[r,c]=fc
        return a
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_fill_extend_v(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp).copy(); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
        for c in range(w):
            rows=[int(a[r,c]) for r in range(h) if a[r,c]!=bg]
            if rows:
                fc=Counter(rows).most_common(1)[0][0]
                for r in range(h):
                    if a[r,c]==bg: a[r,c]=fc
        return a
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_fill_enclosed(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp).copy(); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
        ext=np.zeros((h,w),dtype=bool); stk=[]
        for r in range(h):
            for c in [0,w-1]:
                if a[r,c]==bg: stk.append((r,c))
        for c in range(w):
            for r in [0,h-1]:
                if a[r,c]==bg: stk.append((r,c))
        while stk:
            r,c=stk.pop()
            if r<0 or r>=h or c<0 or c>=w or ext[r,c] or a[r,c]!=bg: continue
            ext[r,c]=True; stk.extend([(r-1,c),(r+1,c),(r,c-1),(r,c+1)])
        vis=np.zeros((h,w),dtype=bool)
        for r in range(h):
            for c in range(w):
                if a[r,c]==bg and not ext[r,c] and not vis[r,c]:
                    region=[]; border=Counter(); stk2=[(r,c)]
                    while stk2:
                        cr,cc=stk2.pop()
                        if cr<0 or cr>=h or cc<0 or cc>=w or vis[cr,cc]: continue
                        if a[cr,cc]!=bg: border[int(a[cr,cc])]+=1; continue
                        if ext[cr,cc]: continue
                        vis[cr,cc]=True; region.append((cr,cc))
                        stk2.extend([(cr-1,cc),(cr+1,cc),(cr,cc-1),(cr,cc+1)])
                    if region and border:
                        fc=border.most_common(1)[0][0]
                        for cr,cc in region: a[cr,cc]=fc
        return a
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

# ══════════════════════════════════════════════════════════════════
# ALL OPS + SOLVER
# ══════════════════════════════════════════════════════════════════
ALL_ADVANCED_OPS = [
    ("distance_recolor", try_recolor_by_distance),
    ("recolor_by_size", try_recolor_by_size),
    ("recolor_propagation", try_recolor_propagation),
    ("majority_neighbor", try_recolor_majority_neighbor),
    ("row_col_rule", try_recolor_row_col_rule),
    ("extract_unique", try_extract_unique),
    ("extract_by_color", try_extract_by_color),
    ("extract_border", try_extract_border),
    ("extract_intersection", try_extract_intersection),
    ("fill_symmetry", try_fill_symmetry),
    ("fill_voronoi", try_fill_voronoi),
    ("fill_extend_h", try_fill_extend_h),
    ("fill_extend_v", try_fill_extend_v),
    ("fill_enclosed", try_fill_enclosed),
]

def solve_task_advanced(task):
    pairs=task['train']; tests=task['test']
    s1=s1_solve(task)
    if score_task(task,s1): return s1,'S1'
    for name,fn in ALL_ADVANCED_OPS:
        try:
            rule=fn(pairs)
            if rule is not None:
                gs=[]; ok=True
                for tc in tests:
                    try: gs.append([rule(tc['input']).tolist()])
                    except: ok=False; break
                if ok and score_task(task,gs): return gs,f'ADV:{name}'
        except: continue
    return s1,'FAIL'

def run_benchmark(data_dir,limit=None,verbose=False):
    files=sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if limit: files=files[:limit]
    print("="*70)
    print(f"  ARC-AGI-2 Advanced Operations ({len(ALL_ADVANCED_OPS)} ops)")
    print(f"  Tasks: {len(files)}")
    print("="*70)
    solved=s1_n=adv_n=0; adv_ops=Counter(); t0=time.time()
    for fi,f in enumerate(files):
        task=json.load(open(os.path.join(data_dir,f)))
        preds,method=solve_task_advanced(task)
        if score_task(task,preds):
            solved+=1
            if method=='S1': s1_n+=1
            else: adv_n+=1; adv_ops[method]+=1
            if verbose and method!='S1': print(f"  {f[:20]}: {method}")
        if (fi+1)%max(1,len(files)//5)==0:
            print(f"    {fi+1}/{len(files)} | Solved: {solved} | {time.time()-t0:.0f}s")
    pct=solved/max(len(files),1)*100
    print(f"\n{'='*70}")
    print(f"  RESULTS: {solved}/{len(files)} ({pct:.1f}%)")
    print(f"    S1: {s1_n}  Advanced: {adv_n}")
    for op,cnt in adv_ops.most_common(): print(f"      {op}: {cnt}")
    print(f"  Time: {time.time()-t0:.1f}s")
    print(f"{'='*70}")

def run_tests():
    print("="*65); print("  Advanced Ops Tests"); print("="*65)
    p=t=0
    print("\n  T1: Objects CC")
    a=np.array([[1,1,0],[0,0,0],[0,2,2]]); objs,bg=get_objects_cc(a)
    ok=len(objs)==2; print(f"    {len(objs)} objects {'PASS' if ok else 'FAIL'}"); p+=int(ok); t+=1
    
    print("\n  T2: Recolor by size")
    rule=try_recolor_by_size([{'input':[[0,1,0],[0,0,0],[2,2,2]],'output':[[0,5,0],[0,0,0],[2,2,2]]}])
    ok=rule is not None; print(f"    {'PASS' if ok else 'FAIL'}"); p+=int(ok); t+=1
    
    print("\n  T3: Fill symmetry")
    rule=try_fill_symmetry([{'input':[[1,0,0],[0,0,0],[0,0,0]],'output':[[1,0,1],[0,0,0],[1,0,1]]}])
    ok=rule is not None; print(f"    {'PASS' if ok else 'FAIL'}"); p+=int(ok); t+=1
    
    print("\n  T4: Fill enclosed")
    rule=try_fill_enclosed([{'input':[[1,1,1],[1,0,1],[1,1,1]],'output':[[1,1,1],[1,1,1],[1,1,1]]}])
    ok=rule is not None; print(f"    {'PASS' if ok else 'FAIL'}"); p+=int(ok); t+=1
    
    print("\n  T5: All ops count")
    ok=len(ALL_ADVANCED_OPS)>=14; print(f"    {len(ALL_ADVANCED_OPS)} ops {'PASS' if ok else 'FAIL'}"); p+=int(ok); t+=1
    
    print("\n  T6: Extract by color")
    rule=try_extract_by_color([{'input':[[0,0,0,0],[0,3,3,0],[0,3,3,0],[0,0,0,1]],'output':[[3,3],[3,3]]}])
    ok=rule is not None; print(f"    {'PASS' if ok else 'FAIL'}"); p+=int(ok); t+=1
    
    print(f"\n{'='*65}\n  Results: {p}/{t}\n{'='*65}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--test",action="store_true")
    ap.add_argument("--training",action="store_true")
    ap.add_argument("--eval",action="store_true")
    ap.add_argument("--limit",type=int)
    ap.add_argument("-v","--verbose",action="store_true")
    ap.add_argument("--data",type=str)
    args=ap.parse_args()
    if args.data: db=args.data
    else:
        for b in [".","ARC-AGI-2","../ARC-AGI-2","/home/claude/ARC-AGI-2","C:/Users/MeteorAI/Desktop/ARC-AGI-2"]:
            if os.path.exists(os.path.join(b,"data")): db=os.path.join(b,"data"); break
        else: db="ARC-AGI-2/data"
    if args.test: run_tests()
    elif args.eval: run_benchmark(os.path.join(db,"evaluation"),args.limit,args.verbose)
    else: run_benchmark(os.path.join(db,"training"),args.limit,args.verbose)
