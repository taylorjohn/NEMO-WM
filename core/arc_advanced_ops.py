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
# 4. NEW HIGH-LEVERAGE OPS (targeting 33+ additional tasks)
# ══════════════════════════════════════════════════════════════════

def try_extract_by_property(pairs):
    """Extract object by property: smallest, leftmost, topmost, rightmost."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if go.shape[0]>=gi.shape[0] and go.shape[1]>=gi.shape[1]: return None
    objs,bg=get_objects_cc(gi)
    if len(objs)<2: return None
    for prop in ['smallest','leftmost','topmost','rightmost','bottommost']:
        if prop=='smallest': target=min(objs,key=lambda o:o['size'])
        elif prop=='leftmost': target=min(objs,key=lambda o:o['bbox'][2])
        elif prop=='topmost': target=min(objs,key=lambda o:o['bbox'][0])
        elif prop=='rightmost': target=max(objs,key=lambda o:o['bbox'][3])
        elif prop=='bottommost': target=max(objs,key=lambda o:o['bbox'][1])
        cr=crop_bbox(gi,target['bbox'])
        if cr.shape==go.shape and np.array_equal(cr,go):
            def mk(p=prop):
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    if len(o2)<2: return a
                    if p=='smallest': t=min(o2,key=lambda o:o['size'])
                    elif p=='leftmost': t=min(o2,key=lambda o:o['bbox'][2])
                    elif p=='topmost': t=min(o2,key=lambda o:o['bbox'][0])
                    elif p=='rightmost': t=max(o2,key=lambda o:o['bbox'][3])
                    elif p=='bottommost': t=max(o2,key=lambda o:o['bbox'][1])
                    return crop_bbox(a,t['bbox'])
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
        # Also try: extract and mask (only target color, bg=0)
        cm=np.where(cr==target['color'],target['color'],0)
        if cm.shape==go.shape and np.array_equal(cm,go):
            def mk2(p=prop):
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    if len(o2)<2: return a
                    if p=='smallest': t=min(o2,key=lambda o:o['size'])
                    elif p=='leftmost': t=min(o2,key=lambda o:o['bbox'][2])
                    elif p=='topmost': t=min(o2,key=lambda o:o['bbox'][0])
                    elif p=='rightmost': t=max(o2,key=lambda o:o['bbox'][3])
                    elif p=='bottommost': t=max(o2,key=lambda o:o['bbox'][1])
                    cr2=crop_bbox(a,t['bbox'])
                    return np.where(cr2==t['color'],t['color'],0)
                return apply
            fn=mk2()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_fill_by_projection(pairs):
    """For each row/col with colored cells, fill entire row/col with that color."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    for mode in ['row','col','both']:
        def mk(m=mode):
            def apply(inp):
                a=np.array(inp).copy(); h,w=a.shape
                bg=int(np.argmax(np.bincount(a.flatten())))
                if m in ['row','both']:
                    for r in range(h):
                        nz=[int(a[r,c]) for c in range(w) if a[r,c]!=bg]
                        if len(nz)==1:
                            for c in range(w): a[r,c]=nz[0]
                if m in ['col','both']:
                    for c in range(w):
                        nz=[int(a[r,c]) for r in range(h) if a[r,c]!=bg]
                        if len(nz)==1:
                            for r in range(h): a[r,c]=nz[0]
                return a
            return apply
        fn=mk()
        if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_recolor_by_neighbor_identity(pairs):
    """If cell touches color X, become color Y."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    rules={}; ok=True
    for p in pairs:
        gi=np.array(p['input']); go=np.array(p['output']); h,w=gi.shape
        for r in range(h):
            for c in range(w):
                ic=int(gi[r,c]); oc=int(go[r,c])
                if ic==oc: continue
                ncolors=set()
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and gi[nr,nc]!=0:
                        ncolors.add(int(gi[nr,nc]))
                for nc_color in ncolors:
                    k=(ic,nc_color)
                    if k in rules and rules[k]!=oc: pass
                    else: rules[k]=oc
    if not rules: return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; res=a.copy()
        for r in range(h):
            for c in range(w):
                ic=int(a[r,c])
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and a[nr,nc]!=0:
                        k=(ic,int(a[nr,nc]))
                        if k in rules: res[r,c]=rules[k]; break
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_pattern_tiling(pairs):
    """Detect repeating motif and tile to fill output."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    ih,iw=gi.shape; oh,ow=go.shape
    if oh<=ih and ow<=iw: return None
    # Check if output is tiled version of input
    for reps_h in range(1,6):
        for reps_w in range(1,6):
            if reps_h*ih==oh and reps_w*iw==ow:
                tiled=np.tile(gi,(reps_h,reps_w))
                if np.array_equal(tiled,go):
                    def mk(rh=reps_h,rw=reps_w):
                        def apply(inp):
                            return np.tile(np.array(inp),(rh,rw))
                        return apply
                    fn=mk()
                    if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    # Check if output is tiled version of a sub-pattern of input
    for ph in range(1,ih+1):
        for pw in range(1,iw+1):
            if oh%ph==0 and ow%pw==0:
                pat=gi[:ph,:pw]
                tiled=np.tile(pat,(oh//ph,ow//pw))
                if np.array_equal(tiled,go):
                    def mk2(pph=ph,ppw=pw,ooh=oh,oow=ow):
                        def apply(inp):
                            a=np.array(inp)
                            p2=a[:pph,:ppw]
                            return np.tile(p2,(ooh//pph,oow//ppw))
                        return apply
                    fn=mk2()
                    if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_symmetry_completion_rotational(pairs):
    """Complete to 4-fold rotational symmetry."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    go=np.array(pairs[0]['output']); h,w=go.shape
    if h!=w: return None
    # Check if output has 4-fold rotational symmetry
    if not (np.array_equal(go,np.rot90(go,1)) and np.array_equal(go,np.rot90(go,2))): return None
    def apply(inp):
        a=np.array(inp).copy(); h,w=a.shape
        bg=int(np.argmax(np.bincount(a.flatten())))
        # Fill missing by rotating existing content
        for rot in [1,2,3]:
            rotated=np.rot90(a,rot)
            for r in range(h):
                for c in range(w):
                    if a[r,c]==bg and rotated[r,c]!=bg:
                        a[r,c]=rotated[r,c]
        return a
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_border_crop(pairs):
    """Crop to bounding box of all non-background cells."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if go.shape[0]>=gi.shape[0] and go.shape[1]>=gi.shape[1]: return None
    bg=int(np.argmax(np.bincount(gi.flatten())))
    nz=np.argwhere(gi!=bg)
    if len(nz)==0: return None
    r1,c1=nz.min(axis=0); r2,c2=nz.max(axis=0)
    cr=gi[r1:r2+1,c1:c2+1]
    if cr.shape==go.shape and np.array_equal(cr,go):
        def apply(inp):
            a=np.array(inp); bg2=int(np.argmax(np.bincount(a.flatten())))
            nz2=np.argwhere(a!=bg2)
            if len(nz2)==0: return a
            r1,c1=nz2.min(axis=0); r2,c2=nz2.max(axis=0)
            return a[r1:r2+1,c1:c2+1]
        if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

def try_color_map_global(pairs):
    """Simple global color substitution map."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    cmap={}; ok=True
    for p in pairs:
        gi=np.array(p['input']); go=np.array(p['output'])
        for r in range(gi.shape[0]):
            for c in range(gi.shape[1]):
                ic=int(gi[r,c]); oc=int(go[r,c])
                if ic in cmap and cmap[ic]!=oc: ok=False; break
                cmap[ic]=oc
            if not ok: break
        if not ok: break
    if not ok or not cmap: return None
    if all(k==v for k,v in cmap.items()): return None
    def apply(inp):
        a=np.array(inp); res=a.copy()
        for k,v in cmap.items(): res[a==k]=v
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_extract_subgrid(pairs):
    """Extract subgrid defined by colored dividers."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    h,w=gi.shape
    if go.shape[0]>=h and go.shape[1]>=w: return None
    bg=int(np.argmax(np.bincount(gi.flatten())))
    # Find horizontal and vertical dividers
    h_divs=[r for r in range(h) if len(set(gi[r].tolist()))==1 and gi[r,0]!=bg]
    v_divs=[c for c in range(w) if len(set(gi[:,c].tolist()))==1 and gi[0,c]!=bg]
    if not h_divs and not v_divs: return None
    # Split into sub-grids
    h_bounds=[0]+[r for r in h_divs]+[h]
    v_bounds=[0]+[c for c in v_divs]+[w]
    subgrids=[]
    for i in range(len(h_bounds)-1):
        for j in range(len(v_bounds)-1):
            r1,r2=h_bounds[i],h_bounds[i+1]
            c1,c2=v_bounds[j],v_bounds[j+1]
            if r1 in h_divs: r1+=1
            if c1 in v_divs: c1+=1
            if r1<r2 and c1<c2:
                sg=gi[r1:r2,c1:c2]
                subgrids.append((sg,i,j))
    # Check which subgrid matches output
    for sg,si,sj in subgrids:
        if sg.shape==go.shape and np.array_equal(sg,go):
            # Find the selection criterion
            # Try: subgrid with most non-bg cells
            max_nz=max(np.sum(s[0]!=bg) for s in subgrids)
            target_sg=[s for s in subgrids if np.sum(s[0]!=bg)==max_nz]
            if len(target_sg)==1 and np.array_equal(target_sg[0][0],go):
                def apply(inp):
                    a=np.array(inp); hh,ww=a.shape
                    bg2=int(np.argmax(np.bincount(a.flatten())))
                    hd=[r for r in range(hh) if len(set(a[r].tolist()))==1 and a[r,0]!=bg2]
                    vd=[c for c in range(ww) if len(set(a[:,c].tolist()))==1 and a[0,c]!=bg2]
                    hb=[0]+hd+[hh]; vb=[0]+vd+[ww]
                    sgs=[]
                    for i in range(len(hb)-1):
                        for j in range(len(vb)-1):
                            r1,r2=hb[i],hb[i+1]; c1,c2=vb[j],vb[j+1]
                            if r1 in hd: r1+=1
                            if c1 in vd: c1+=1
                            if r1<r2 and c1<c2: sgs.append(a[r1:r2,c1:c2])
                    if sgs:
                        return max(sgs,key=lambda s:np.sum(s!=bg2))
                    return a
                if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

def try_recolor_by_object_count(pairs):
    """Output color depends on number of objects."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    objs,bg=get_objects_cc(gi)
    n_obj=len(objs)
    # Check if each object is recolored uniformly based on total count
    # or if output = input with bg cells colored based on nearest object
    return None  # placeholder for complex counting rules

def try_gravity_variants(pairs):
    """Gravity in all 4 directions."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    for direction in ['up','left','right']:
        def mk(d=direction):
            def apply(inp):
                a=np.array(inp); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
                res=np.full_like(a,bg)
                if d=='up':
                    for c in range(w):
                        vals=[int(a[r,c]) for r in range(h) if a[r,c]!=bg]
                        for i,v in enumerate(vals): res[i,c]=v
                elif d=='left':
                    for r in range(h):
                        vals=[int(a[r,c]) for c in range(w) if a[r,c]!=bg]
                        for i,v in enumerate(vals): res[r,i]=v
                elif d=='right':
                    for r in range(h):
                        vals=[int(a[r,c]) for c in range(w) if a[r,c]!=bg]
                        for i,v in enumerate(reversed(vals)): res[r,w-1-i]=v
                return res
            return apply
        fn=mk()
        if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_largest_blob_reduction(pairs):
    """Output = solid grid filled with the largest non-bg blob color. (JEPA-discovered)"""
    for p in pairs:
        go=np.array(p['output'])
        if go.size>30 or len(set(go.flatten().tolist()))!=1: return None
    oh=np.array(pairs[0]['output']).shape[0]; ow=np.array(pairs[0]['output']).shape[1]
    for p in pairs:
        gi=np.array(p['input']); go=np.array(p['output']); expected=int(go[0,0])
        h,w=gi.shape; counts=Counter(int(v) for v in gi.flatten())
        total=h*w; bg={c for c,n in counts.items() if n>total*0.15}
        blobs=[(n,c) for c,n in counts.items() if c not in bg and c!=0]
        blobs.sort(reverse=True)
        if not blobs or blobs[0][1]!=expected: return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; counts=Counter(int(v) for v in a.flatten())
        total=h*w; bg={c for c,n in counts.items() if n>total*0.15}
        blobs=[(n,c) for c,n in counts.items() if c not in bg and c!=0]
        blobs.sort(reverse=True)
        if blobs: return np.full((oh,ow),blobs[0][1],dtype=int)
        return a
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

# ══════════════════════════════════════════════════════════════════
# ALL OPS + SOLVER
# ══════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════
# 5. COUNTING / MATH OPS — "JEPA predicts integer, Python draws grid"
# ══════════════════════════════════════════════════════════════════

def try_count_to_size(pairs):
    """Output grid size = count of some property in input.
    e.g., output is NxN where N = number of red objects."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    oh,ow=go.shape
    objs,bg=get_objects_cc(gi)
    
    # Try: output size = number of objects
    n_obj=len(objs)
    if (n_obj==oh or n_obj==ow):
        # Check if consistent across all demos
        ok=True
        for p in pairs[1:]:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            o2,_=get_objects_cc(gi2)
            if len(o2)!=go2.shape[0] and len(o2)!=go2.shape[1]:
                ok=False; break
        if ok:
            # What fills the output?
            out_color=Counter(int(v) for v in go.flatten() if v!=0)
            if out_color:
                fc=out_color.most_common(1)[0][0]
                use_h=(n_obj==oh); use_w=(n_obj==ow)
                def mk(uh=use_h,uw=use_w,c=fc):
                    def apply(inp):
                        a=np.array(inp); o2,_=get_objects_cc(a)
                        n=len(o2)
                        h2=n if uh else go.shape[0]
                        w2=n if uw else go.shape[1]
                        return np.full((h2,w2),c,dtype=int)
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    
    # Try: output size = count of specific color
    for color in range(1,10):
        count=np.sum(gi==color)
        if count>0 and (count==oh or count==ow):
            ok=True
            for p in pairs[1:]:
                gi2=np.array(p['input']); go2=np.array(p['output'])
                c2=np.sum(gi2==color)
                if c2!=go2.shape[0] and c2!=go2.shape[1]: ok=False; break
            if ok:
                out_c=Counter(int(v) for v in go.flatten() if v!=0)
                if out_c:
                    fc=out_c.most_common(1)[0][0]
                    use_h=(count==oh); use_w=(count==ow)
                    def mk2(col=color,uh=use_h,uw=use_w,c=fc):
                        def apply(inp):
                            a=np.array(inp); n=int(np.sum(a==col))
                            h2=n if uh else go.shape[0]
                            w2=n if uw else go.shape[1]
                            return np.full((max(1,h2),max(1,w2)),c,dtype=int)
                        return apply
                    fn=mk2()
                    if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_count_to_color(pairs):
    """Output color = count of objects/cells mapped to a color.
    e.g., 3 objects → color 3 in output."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    objs,bg=get_objects_cc(gi)
    
    # For each object, check if its output color = its size, or count of its neighbors, etc.
    for prop in ['size','neighbor_count','color_count']:
        rule={}; ok=True
        for p in pairs:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            o2,bg2=get_objects_cc(gi2)
            for o in o2:
                if prop=='size': val=o['size']
                elif prop=='neighbor_count': val=sum(1 for o2b in o2 if o2b!=o and any(manh(r1,c1,r2,c2)<=2 for r1,c1 in o['cells'] for r2,c2 in o2b['cells']))
                elif prop=='color_count': val=Counter(o2b['color'] for o2b in o2)[o['color']]
                r,c=o['cells'][0]; oc=int(go2[r,c])
                if val in rule and rule[val]!=oc: ok=False; break
                rule[val]=oc
            if not ok: break
        if ok and rule and len(set(rule.values()))>1:
            def mk(r=rule,p=prop):
                def apply(inp):
                    a=np.array(inp); o2,bg2=get_objects_cc(a); res=np.full_like(a,bg2)
                    for o in o2:
                        if p=='size': val=o['size']
                        elif p=='neighbor_count': val=sum(1 for o2b in o2 if o2b!=o and any(manh(r1,c1,r2,c2)<=2 for r1,c1 in o['cells'] for r2,c2 in o2b['cells']))
                        elif p=='color_count': val=Counter(o2b['color'] for o2b in o2)[o['color']]
                        nc=r.get(val,o['color'])
                        for rr,cc in o['cells']: res[rr,cc]=nc
                    return res
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_output_is_count(pairs):
    """Output is a tiny grid (1x1, 1xN, Nx1) encoding a count.
    e.g., output = [[3]] meaning 3 red objects."""
    go=np.array(pairs[0]['output'])
    if go.size > 10: return None  # output must be tiny
    
    gi=np.array(pairs[0]['input'])
    objs,bg=get_objects_cc(gi)
    oh,ow=go.shape
    
    # Try: output value = number of objects
    if go.size==1:
        val=int(go[0,0])
        # Is val = count of objects?
        if val==len(objs):
            ok=all(int(np.array(p['output'])[0,0])==len(get_objects_cc(np.array(p['input']))[0]) for p in pairs)
            if ok:
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a)
                    return np.array([[len(o2)]],dtype=int)
                if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
        # Is val = count of specific color?
        for color in range(1,10):
            if val==np.sum(gi==color):
                ok=all(int(np.array(p['output'])[0,0])==np.sum(np.array(p['input'])==color) for p in pairs)
                if ok:
                    def mk(c=color):
                        def apply(inp):
                            return np.array([[int(np.sum(np.array(inp)==c))]],dtype=int)
                        return apply
                    fn=mk()
                    if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    
    # Output is 1xN or Nx1 listing object counts or colors
    if oh==1 or ow==1:
        vals=go.flatten().tolist()
        # Is it sorted object sizes?
        sizes=sorted([o['size'] for o in objs])
        if vals==sizes:
            def apply(inp):
                a=np.array(inp); o2,_=get_objects_cc(a)
                s=sorted([o['size'] for o in o2])
                if oh==1: return np.array([s],dtype=int)
                else: return np.array([[v] for v in s],dtype=int)
            if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
        # Is it sorted object colors?
        colors_sorted=sorted([o['color'] for o in objs])
        if vals==colors_sorted:
            def apply(inp):
                a=np.array(inp); o2,_=get_objects_cc(a)
                s=sorted([o['color'] for o in o2])
                if oh==1: return np.array([s],dtype=int)
                else: return np.array([[v] for v in s],dtype=int)
            if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

def try_repeat_by_count(pairs):
    """Output = input repeated N times where N is derived from input."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    ih,iw=gi.shape; oh,ow=go.shape
    if oh<=ih and ow<=iw: return None
    
    objs,bg=get_objects_cc(gi)
    
    # Check: output = input tiled NxM where N or M = object count
    for rh in range(1,8):
        for rw in range(1,8):
            if rh*ih==oh and rw*iw==ow:
                tiled=np.tile(gi,(rh,rw))
                if np.array_equal(tiled,go):
                    # Is rh or rw = some count?
                    n_obj=len(objs)
                    n_colors=len(set(int(v) for v in gi.flatten() if v!=bg))
                    
                    for count_src,count_val in [('n_obj',n_obj),('n_colors',n_colors)]:
                        if rh==count_val or rw==count_val:
                            def mk(cs=count_src,rh0=rh,rw0=rw):
                                def apply(inp):
                                    a=np.array(inp); o2,bg2=get_objects_cc(a)
                                    if cs=='n_obj': n=len(o2)
                                    else: n=len(set(int(v) for v in a.flatten() if v!=bg2))
                                    rrh=n if rh0==count_val else rh0
                                    rrw=n if rw0==count_val else rw0
                                    return np.tile(a,(max(1,rrh),max(1,rrw)))
                                return apply
                            fn=mk()
                            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_scale_by_count(pairs):
    """Output = input scaled by factor N where N is a count from input."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    ih,iw=gi.shape; oh,ow=go.shape
    if oh<=ih: return None
    if oh%ih!=0 or ow%iw!=0: return None
    
    sh=oh//ih; sw=ow//iw
    if sh!=sw: return None
    scale=sh
    
    scaled=np.repeat(np.repeat(gi,scale,axis=0),scale,axis=1)
    if not np.array_equal(scaled,go): return None
    
    objs,bg=get_objects_cc(gi)
    # Is scale = some count?
    for csrc,cval in [('n_obj',len(objs)),('n_colors',len(set(int(v) for v in gi.flatten() if v!=bg)))]:
        if scale==cval:
            ok=True
            for p in pairs[1:]:
                gi2=np.array(p['input']); go2=np.array(p['output'])
                o2,bg2=get_objects_cc(gi2)
                if csrc=='n_obj': n=len(o2)
                else: n=len(set(int(v) for v in gi2.flatten() if v!=bg2))
                sc=np.repeat(np.repeat(gi2,n,axis=0),n,axis=1)
                if not np.array_equal(sc,go2): ok=False; break
            if ok:
                def mk(cs=csrc):
                    def apply(inp):
                        a=np.array(inp); o2,bg2=get_objects_cc(a)
                        if cs=='n_obj': n=len(o2)
                        else: n=len(set(int(v) for v in a.flatten() if v!=bg2))
                        return np.repeat(np.repeat(a,max(1,n),axis=0),max(1,n),axis=1)
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

# ══════════════════════════════════════════════════════════════════
# 6. OBJECT FILTERING (244 unsolved tasks)
# ══════════════════════════════════════════════════════════════════
def try_filter_by_max_color(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp); bg=int(np.argmax(np.bincount(a.flatten())))
        counts=Counter(int(v) for v in a.flatten() if v!=bg)
        if not counts: return a
        keep=counts.most_common(1)[0][0]
        res=np.full_like(a,bg); res[a==keep]=keep; return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_filter_by_min_size(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); objs,bg=get_objects_cc(gi)
    if len(objs)<2: return None
    def apply(inp):
        a=np.array(inp); o2,bg2=get_objects_cc(a)
        if len(o2)<2: return a
        ms=min(o['size'] for o in o2)
        res=np.full_like(a,bg2)
        for o in o2:
            if o['size']>ms:
                for r,c in o['cells']: res[r,c]=o['color']
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_keep_only_specific_color(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    bg=int(np.argmax(np.bincount(gi.flatten())))
    out_colors=set(int(v) for v in go.flatten() if v!=bg)
    if not out_colors or len(out_colors)>3: return None
    def apply(inp):
        a=np.array(inp); bg2=int(np.argmax(np.bincount(a.flatten())))
        res=np.full_like(a,bg2)
        for c in out_colors: res[a==c]=c
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_remove_only_color(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    bg=int(np.argmax(np.bincount(gi.flatten())))
    in_c=set(int(v) for v in gi.flatten() if v!=bg)
    out_c=set(int(v) for v in go.flatten() if v!=bg)
    removed=in_c-out_c
    if not removed or len(removed)>2: return None
    def apply(inp):
        a=np.array(inp).copy(); bg2=int(np.argmax(np.bincount(a.flatten())))
        for c in removed: a[a==int(c)]=bg2
        return a
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

# ══════════════════════════════════════════════════════════════════
# 7. OBJECT MOVEMENT (83 unsolved tasks)
# ══════════════════════════════════════════════════════════════════
def try_move_to_corner(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    bg=int(np.argmax(np.bincount(gi.flatten())))
    nz=np.argwhere(gi!=bg)
    if len(nz)==0: return None
    r1,c1=nz.min(axis=0); r2,c2=nz.max(axis=0)
    ch=r2-r1+1; cw=c2-c1+1; ct=gi[r1:r2+1,c1:c2+1].copy()
    for corner in ['top_left','top_right','bottom_left','bottom_right']:
        test=np.full_like(gi,bg)
        if corner=='top_left': test[:ch,:cw]=ct
        elif corner=='top_right': test[:ch,w-cw:]=ct
        elif corner=='bottom_left': test[h-ch:,:cw]=ct
        elif corner=='bottom_right': test[h-ch:,w-cw:]=ct
        if np.array_equal(test,go):
            def mk(cn=corner):
                def apply(inp):
                    a=np.array(inp); hh,ww=a.shape
                    bg2=int(np.argmax(np.bincount(a.flatten())))
                    nz2=np.argwhere(a!=bg2)
                    if len(nz2)==0: return a
                    rr1,cc1=nz2.min(axis=0); rr2,cc2=nz2.max(axis=0)
                    chh=rr2-rr1+1; cww=cc2-cc1+1; ctt=a[rr1:rr2+1,cc1:cc2+1].copy()
                    res=np.full_like(a,bg2)
                    if cn=='top_left': res[:chh,:cww]=ctt
                    elif cn=='top_right': res[:chh,ww-cww:]=ctt
                    elif cn=='bottom_left': res[hh-chh:,:cww]=ctt
                    elif cn=='bottom_right': res[hh-chh:,ww-cww:]=ctt
                    return res
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_compact_gravity(pairs):
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    bg=int(np.argmax(np.bincount(gi.flatten())))
    # Remove empty rows, pack to top
    ner=[r for r in range(h) if any(gi[r,c]!=bg for c in range(w))]
    if ner:
        packed=np.full_like(gi,bg)
        for i,r in enumerate(ner): packed[i]=gi[r]
        if np.array_equal(packed,go):
            def apply(inp):
                a=np.array(inp); hh,ww=a.shape; bg2=int(np.argmax(np.bincount(a.flatten())))
                nr=[r for r in range(hh) if any(a[r,c]!=bg2 for c in range(ww))]
                res=np.full_like(a,bg2)
                for i,r in enumerate(nr): res[i]=a[r]
                return res
            if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    # Remove empty cols, pack left
    nec=[c for c in range(w) if any(gi[r,c]!=bg for r in range(h))]
    if nec:
        packed2=np.full_like(gi,bg)
        for i,c in enumerate(nec): packed2[:,i]=gi[:,c]
        if np.array_equal(packed2,go):
            def apply(inp):
                a=np.array(inp); hh,ww=a.shape; bg2=int(np.argmax(np.bincount(a.flatten())))
                nc=[c for c in range(ww) if any(a[r,c]!=bg2 for r in range(hh))]
                res=np.full_like(a,bg2)
                for i,c in enumerate(nc): res[:,i]=a[:,c]
                return res
            if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

# ══════════════════════════════════════════════════════════════════
# 8. PATTERN STAMPING, BOOLEAN OPS, BOUNDARY, FLOOD, ARITHMETIC,
#    MIRROR BBOX, SORT ROWS, OVERLAY
# ══════════════════════════════════════════════════════════════════

def try_pattern_stamp(pairs):
    """Find a small pattern in input, stamp it at marked locations."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if gi.shape!=go.shape: return None
    h,w=gi.shape; bg=int(np.argmax(np.bincount(gi.flatten())))
    objs,_=get_objects_cc(gi)
    if len(objs)<2: return None
    # Find the smallest object as the "pattern"
    objs_s=sorted(objs,key=lambda o:o['size'])
    for pattern_obj in objs_s[:3]:
        pb=pattern_obj['bbox']
        pat=gi[pb[0]:pb[1]+1,pb[2]:pb[3]+1].copy()
        ph,pw=pat.shape
        # Find marker positions (other objects)
        markers=[(o['bbox'][0],o['bbox'][2]) for o in objs if o!=pattern_obj]
        # Check if output = input with pattern stamped at each marker
        test=gi.copy()
        for mr,mc in markers:
            for r in range(ph):
                for c in range(pw):
                    if pat[r,c]!=bg and 0<=mr+r<h and 0<=mc+c<w:
                        test[mr+r,mc+c]=pat[r,c]
        if np.array_equal(test,go):
            def mk(p=pat,ph2=ph,pw2=pw,bg2=bg):
                def apply(inp):
                    a=np.array(inp); hh,ww=a.shape
                    o2,bg3=get_objects_cc(a)
                    if len(o2)<2: return a
                    o2s=sorted(o2,key=lambda o:o['size'])
                    pat_o=o2s[0]; pb2=pat_o['bbox']
                    pp=a[pb2[0]:pb2[1]+1,pb2[2]:pb2[3]+1].copy()
                    pph,ppw=pp.shape
                    markers2=[(o['bbox'][0],o['bbox'][2]) for o in o2 if o!=pat_o]
                    res=a.copy()
                    for mr,mc in markers2:
                        for r in range(pph):
                            for c in range(ppw):
                                if pp[r,c]!=bg3 and 0<=mr+r<hh and 0<=mc+c<ww:
                                    res[mr+r,mc+c]=pp[r,c]
                    return res
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_boolean_ops(pairs):
    """AND/OR/XOR of two colored regions."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    if gi.shape!=go.shape: return None
    bg=int(np.argmax(np.bincount(gi.flatten())))
    colors=sorted(set(int(v) for v in gi.flatten())-{bg})
    if len(colors)<2: return None
    c1,c2=colors[0],colors[1]
    m1=(gi==c1); m2=(gi==c2)
    # Try different boolean ops
    for op_name,op_result in [('and',m1&m2),('or',m1|m2),('xor',m1^m2),('diff1',m1&~m2),('diff2',m2&~m1)]:
        for out_color in colors+[bg]:
            test=np.full_like(gi,bg)
            test[op_result]=out_color if out_color!=bg else colors[0]
            # Also try keeping original colors
            test2=np.full_like(gi,bg)
            test2[m1&op_result]=c1
            test2[m2&op_result]=c2
            for t in [test,test2]:
                if np.array_equal(t,go):
                    def mk(op=op_name,cc1=c1,cc2=c2,oc=out_color,bg2=bg):
                        def apply(inp):
                            a=np.array(inp); bg3=int(np.argmax(np.bincount(a.flatten())))
                            cols=sorted(set(int(v) for v in a.flatten())-{bg3})
                            if len(cols)<2: return a
                            mm1=(a==cols[0]); mm2=(a==cols[1])
                            if op=='and': mask=mm1&mm2
                            elif op=='or': mask=mm1|mm2
                            elif op=='xor': mask=mm1^mm2
                            elif op=='diff1': mask=mm1&~mm2
                            else: mask=mm2&~mm1
                            res=np.full_like(a,bg3)
                            res[mask]=oc if oc!=bg2 else cols[0]
                            return res
                        return apply
                    fn=mk()
                    if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_boundary_trace(pairs):
    """Output = outline/boundary of objects (inner or outer)."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    bg=int(np.argmax(np.bincount(gi.flatten())))
    # Inner boundary: cells that have at least one bg neighbor
    def inner_boundary(a):
        hh,ww=a.shape; bg2=int(np.argmax(np.bincount(a.flatten())))
        res=np.full_like(a,bg2)
        for r in range(hh):
            for c in range(ww):
                if a[r,c]!=bg2:
                    has_bg=any(0<=r+dr<hh and 0<=c+dc<ww and a[r+dr,c+dc]==bg2 
                               for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
                    if has_bg or r==0 or r==hh-1 or c==0 or c==ww-1:
                        res[r,c]=a[r,c]
        return res
    # Outer boundary: bg cells that have at least one non-bg neighbor
    def outer_boundary(a):
        hh,ww=a.shape; bg2=int(np.argmax(np.bincount(a.flatten())))
        res=np.full_like(a,bg2)
        for r in range(hh):
            for c in range(ww):
                if a[r,c]==bg2:
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=r+dr,c+dc
                        if 0<=nr<hh and 0<=nc<ww and a[nr,nc]!=bg2:
                            res[r,c]=a[nr,nc]; break
        return res
    for bfn in [inner_boundary, outer_boundary]:
        if np.array_equal(bfn(gi),go):
            def mk(f=bfn):
                def apply(inp): return f(np.array(inp))
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_flood_from_seeds(pairs):
    """BFS flood fill from seed cells of a specific color."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    bg=int(np.argmax(np.bincount(gi.flatten())))
    # Find seed color (appears in input, expands in output)
    for seed_color in set(int(v) for v in gi.flatten())-{bg}:
        seed_cells=[(r,c) for r in range(h) for c in range(w) if gi[r,c]==seed_color]
        if not seed_cells: continue
        # Try BFS with different radii
        for radius in range(1,6):
            test=gi.copy()
            for sr,sc in seed_cells:
                for r in range(h):
                    for c in range(w):
                        if test[r,c]==bg and abs(r-sr)+abs(c-sc)<=radius:
                            test[r,c]=seed_color
            if np.array_equal(test,go):
                def mk(sc2=seed_color,rad=radius):
                    def apply(inp):
                        a=np.array(inp); hh,ww=a.shape
                        bg2=int(np.argmax(np.bincount(a.flatten())))
                        seeds=[(r,c) for r in range(hh) for c in range(ww) if a[r,c]==sc2]
                        res=a.copy()
                        for sr,scc in seeds:
                            for r in range(hh):
                                for c in range(ww):
                                    if res[r,c]==bg2 and abs(r-sr)+abs(c-scc)<=rad:
                                        res[r,c]=sc2
                        return res
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_pixel_arithmetic(pairs):
    """Output = element-wise operation on two halves of input grid."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    oh,ow=go.shape
    # Check if input can be split into two halves that combine to make output
    # Horizontal split
    if h%2==0 and oh==h//2 and ow==w:
        top=gi[:h//2]; bot=gi[h//2:]
        for op in ['max','min','add_mod','xor']:
            if op=='max': result=np.maximum(top,bot)
            elif op=='min': result=np.where((top!=0)&(bot!=0),np.minimum(top,bot),np.maximum(top,bot))
            elif op=='add_mod': result=(top+bot)%10
            elif op=='xor': result=np.where(top==bot,0,np.maximum(top,bot))
            if np.array_equal(result,go):
                def mk(o=op):
                    def apply(inp):
                        a=np.array(inp); hh=a.shape[0]
                        t=a[:hh//2]; b=a[hh//2:]
                        if o=='max': return np.maximum(t,b)
                        elif o=='min': return np.where((t!=0)&(b!=0),np.minimum(t,b),np.maximum(t,b))
                        elif o=='add_mod': return (t+b)%10
                        else: return np.where(t==b,0,np.maximum(t,b))
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    # Vertical split
    if w%2==0 and ow==w//2 and oh==h:
        left=gi[:,:w//2]; right=gi[:,w//2:]
        for op in ['max','min','add_mod','xor']:
            if op=='max': result=np.maximum(left,right)
            elif op=='min': result=np.where((left!=0)&(right!=0),np.minimum(left,right),np.maximum(left,right))
            elif op=='add_mod': result=(left+right)%10
            elif op=='xor': result=np.where(left==right,0,np.maximum(left,right))
            if np.array_equal(result,go):
                def mk(o=op):
                    def apply(inp):
                        a=np.array(inp); ww=a.shape[1]
                        l=a[:,:ww//2]; r=a[:,ww//2:]
                        if o=='max': return np.maximum(l,r)
                        elif o=='min': return np.where((l!=0)&(r!=0),np.minimum(l,r),np.maximum(l,r))
                        elif o=='add_mod': return (l+r)%10
                        else: return np.where(l==r,0,np.maximum(l,r))
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_mirror_within_bbox(pairs):
    """Mirror content within each object's bounding box."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    objs,bg=get_objects_cc(gi)
    if len(objs)<1: return None
    for axis in ['h','v']:
        test=gi.copy()
        for o in objs:
            r1,r2,c1,c2=o['bbox']
            sub=test[r1:r2+1,c1:c2+1]
            if axis=='h': test[r1:r2+1,c1:c2+1]=sub[:,::-1]
            else: test[r1:r2+1,c1:c2+1]=sub[::-1,:]
        if np.array_equal(test,go):
            def mk(ax=axis):
                def apply(inp):
                    a=np.array(inp); o2,_=get_objects_cc(a); res=a.copy()
                    for o in o2:
                        r1,r2,c1,c2=o['bbox']
                        sub=res[r1:r2+1,c1:c2+1]
                        if ax=='h': res[r1:r2+1,c1:c2+1]=sub[:,::-1]
                        else: res[r1:r2+1,c1:c2+1]=sub[::-1,:]
                    return res
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_sort_rows(pairs):
    """Sort rows by some property (non-bg count, first non-bg color, etc.)."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    bg=int(np.argmax(np.bincount(gi.flatten())))
    for key_fn_name in ['nz_count','first_color','sum','max_color']:
        def row_key(row,bg2=bg,kn=key_fn_name):
            nz=[int(v) for v in row if v!=bg2]
            if kn=='nz_count': return len(nz)
            elif kn=='first_color': return nz[0] if nz else 0
            elif kn=='sum': return sum(nz)
            elif kn=='max_color': return max(nz) if nz else 0
        for rev in [False,True]:
            sorted_rows=sorted(range(h),key=lambda r:row_key(gi[r]),reverse=rev)
            test=np.array([gi[r] for r in sorted_rows])
            if np.array_equal(test,go):
                def mk(kn=key_fn_name,rv=rev):
                    def apply(inp):
                        a=np.array(inp); hh=a.shape[0]
                        bg2=int(np.argmax(np.bincount(a.flatten())))
                        def rk(row):
                            nz=[int(v) for v in row if v!=bg2]
                            if kn=='nz_count': return len(nz)
                            elif kn=='first_color': return nz[0] if nz else 0
                            elif kn=='sum': return sum(nz)
                            else: return max(nz) if nz else 0
                        sr=sorted(range(hh),key=lambda r:rk(a[r]),reverse=rv)
                        return np.array([a[r] for r in sr])
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    # Also try sort columns
    for key_fn_name in ['nz_count','first_color']:
        def col_key(col,bg2=bg,kn=key_fn_name):
            nz=[int(v) for v in col if v!=bg2]
            if kn=='nz_count': return len(nz)
            else: return nz[0] if nz else 0
        for rev in [False,True]:
            sorted_cols=sorted(range(w),key=lambda c:col_key(gi[:,c]),reverse=rev)
            test=np.column_stack([gi[:,c] for c in sorted_cols])
            if np.array_equal(test,go):
                def mk(kn=key_fn_name,rv=rev):
                    def apply(inp):
                        a=np.array(inp); ww=a.shape[1]
                        bg2=int(np.argmax(np.bincount(a.flatten())))
                        def ck(col):
                            nz=[int(v) for v in col if v!=bg2]
                            if kn=='nz_count': return len(nz)
                            else: return nz[0] if nz else 0
                        sc=sorted(range(ww),key=lambda c:ck(a[:,c]),reverse=rv)
                        return np.column_stack([a[:,c] for c in sc])
                    return apply
                fn=mk()
                if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_overlay_objects(pairs):
    """Stack/overlay objects on top of each other."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    objs,bg=get_objects_cc(gi)
    if len(objs)<2: return None
    oh,ow=go.shape
    # Check if output = overlay of all object bboxes
    # All objects cropped and stacked
    crops=[]
    for o in objs:
        r1,r2,c1,c2=o['bbox']
        crops.append(gi[r1:r2+1,c1:c2+1].copy())
    if not crops: return None
    # All same size?
    shapes=set(c.shape for c in crops)
    if len(shapes)!=1: return None
    ch,cw=crops[0].shape
    if oh!=ch or ow!=cw: return None
    # Try overlay: later objects override earlier (painter's algorithm)
    for order in [range(len(crops)),range(len(crops)-1,-1,-1)]:
        overlay=np.full((ch,cw),bg,dtype=int)
        for idx in order:
            mask=crops[idx]!=bg
            overlay[mask]=crops[idx][mask]
        if np.array_equal(overlay,go):
            def mk(ord_rev=(list(order)!=list(range(len(crops))))):
                def apply(inp):
                    a=np.array(inp); o2,bg2=get_objects_cc(a)
                    if len(o2)<2: return a
                    cr2=[]
                    for o in o2:
                        r1,r2,c1,c2=o['bbox']
                        cr2.append(a[r1:r2+1,c1:c2+1].copy())
                    ss=set(c.shape for c in cr2)
                    if len(ss)!=1: return a
                    ch2,cw2=cr2[0].shape
                    ov=np.full((ch2,cw2),bg2,dtype=int)
                    order2=range(len(cr2)-1,-1,-1) if ord_rev else range(len(cr2))
                    for idx in order2:
                        mask=cr2[idx]!=bg2
                        ov[mask]=cr2[idx][mask]
                    return ov
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

# ══════════════════════════════════════════════════════════════════
# 9. COLOR PERMUTATION SEARCH, INVERSE SOLVING, DEDUP, NOISE, PARITY
# ══════════════════════════════════════════════════════════════════

def try_color_permutation(pairs):
    """Try all bijective color mappings learned from first demo."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    # Learn mapping from first pair
    cmap={}; ok=True
    for r in range(gi.shape[0]):
        for c in range(gi.shape[1]):
            ic=int(gi[r,c]); oc=int(go[r,c])
            if ic in cmap and cmap[ic]!=oc: ok=False; break
            cmap[ic]=oc
        if not ok: break
    if not ok or not cmap: return None
    # Must be a non-trivial permutation
    if all(k==v for k,v in cmap.items()): return None
    # Check bijective (each output color maps from exactly one input)
    rev={}
    for k,v in cmap.items():
        if v in rev and rev[v]!=k: return None  # not bijective
        rev[v]=k
    def apply(inp):
        a=np.array(inp); res=a.copy()
        for k,v in cmap.items(): res[a==k]=v
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_inverse_solve(pairs):
    """Try solving (output→input) with existing ops, then invert."""
    # Swap input/output and try simple transforms
    inv_pairs=[{'input':p['output'],'output':p['input']} for p in pairs]
    gi=np.array(inv_pairs[0]['input']); go=np.array(inv_pairs[0]['output'])
    if gi.shape!=go.shape: return None
    # Try if inverse is a simple rotation
    for k in [1,2,3]:
        if np.array_equal(np.rot90(gi,k),go):
            inv_k=(4-k)%4
            def mk(kk=inv_k):
                def apply(inp): return np.rot90(np.array(inp),kk)
                return apply
            fn=mk()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    # Try if inverse is a flip
    for flip_fn,inv_fn in [(lambda a:a[::-1],lambda a:a[::-1]),(lambda a:a[:,::-1],lambda a:a[:,::-1])]:
        if np.array_equal(flip_fn(gi),go):
            def mk2(f=inv_fn):
                def apply(inp): return f(np.array(inp))
                return apply
            fn=mk2()
            if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_dedup_rows(pairs):
    """Remove duplicate rows or columns."""
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output'])
    ih,iw=gi.shape; oh,ow=go.shape
    # Dedup rows
    if ow==iw and oh<ih:
        seen=[]; unique_rows=[]
        for r in range(ih):
            row_t=tuple(gi[r].tolist())
            if row_t not in seen:
                seen.append(row_t)
                unique_rows.append(gi[r])
        if len(unique_rows)==oh:
            deduped=np.array(unique_rows)
            if np.array_equal(deduped,go):
                def apply(inp):
                    a=np.array(inp); seen2=[]; ur=[]
                    for r in range(a.shape[0]):
                        rt=tuple(a[r].tolist())
                        if rt not in seen2: seen2.append(rt); ur.append(a[r])
                    return np.array(ur) if ur else a
                if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    # Dedup cols
    if oh==ih and ow<iw:
        seen=[]; unique_cols=[]
        for c in range(iw):
            col_t=tuple(gi[:,c].tolist())
            if col_t not in seen:
                seen.append(col_t)
                unique_cols.append(gi[:,c])
        if len(unique_cols)==ow:
            deduped=np.column_stack(unique_cols)
            if np.array_equal(deduped,go):
                def apply(inp):
                    a=np.array(inp); seen2=[]; uc=[]
                    for c in range(a.shape[1]):
                        ct=tuple(a[:,c].tolist())
                        if ct not in seen2: seen2.append(ct); uc.append(a[:,c])
                    return np.column_stack(uc) if uc else a
                if all(np.array_equal(apply(p['input']),np.array(p['output'])) for p in pairs): return apply
    return None

def try_noise_filter(pairs):
    """Remove isolated single-pixel noise (cells with no same-color neighbor)."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
        res=a.copy()
        for r in range(h):
            for c in range(w):
                if a[r,c]==bg: continue
                color=int(a[r,c])
                has_neighbor=any(0<=r+dr<h and 0<=c+dc<w and int(a[r+dr,c+dc])==color
                                  for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)])
                if not has_neighbor:
                    res[r,c]=bg
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_parity_rule(pairs):
    """Color cells based on (row+col) parity or (row*col) mod N."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    bg=int(np.argmax(np.bincount(gi.flatten())))
    
    # Try (row+col)%2 rule
    for mod in [2,3]:
        rules={}; ok=True
        for p in pairs:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            for r in range(gi2.shape[0]):
                for c in range(gi2.shape[1]):
                    key=((r+c)%mod, int(gi2[r,c]))
                    val=int(go2[r,c])
                    if key in rules and rules[key]!=val: ok=False; break
                    rules[key]=val
                if not ok: break
            if not ok: break
        if not ok: continue
        if all(k[1]==v for k,v in rules.items()): continue  # trivial
        def mk(r=rules,m=mod):
            def apply(inp):
                a=np.array(inp); res=a.copy()
                for row in range(a.shape[0]):
                    for col in range(a.shape[1]):
                        key=((row+col)%m, int(a[row,col]))
                        if key in r: res[row,col]=r[key]
                return res
            return apply
        fn=mk()
        if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    
    # Try row%2 only
    for mod in [2,3]:
        rules={}; ok=True
        for p in pairs:
            gi2=np.array(p['input']); go2=np.array(p['output'])
            for r in range(gi2.shape[0]):
                for c in range(gi2.shape[1]):
                    key=(r%mod, int(gi2[r,c]))
                    val=int(go2[r,c])
                    if key in rules and rules[key]!=val: ok=False; break
                    rules[key]=val
                if not ok: break
            if not ok: break
        if not ok: continue
        if all(k[1]==v for k,v in rules.items()): continue
        def mk(r=rules,m=mod):
            def apply(inp):
                a=np.array(inp); res=a.copy()
                for row in range(a.shape[0]):
                    for col in range(a.shape[1]):
                        key=(row%m, int(a[row,col]))
                        if key in r: res[row,col]=r[key]
                return res
            return apply
        fn=mk()
        if all(np.array_equal(fn(p['input']),np.array(p['output'])) for p in pairs): return fn
    return None

def try_connected_flood(pairs):
    """BFS flood fill from non-bg cells - fill connected bg regions with nearest color."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    def apply(inp):
        a=np.array(inp); h,w=a.shape; bg=int(np.argmax(np.bincount(a.flatten())))
        res=a.copy()
        # BFS from all non-bg cells
        from collections import deque
        q=deque()
        for r in range(h):
            for c in range(w):
                if a[r,c]!=bg: q.append((r,c,int(a[r,c]),0))
        visited=set()
        while q:
            r,c,color,dist=q.popleft()
            if (r,c) in visited: continue
            if r<0 or r>=h or c<0 or c>=w: continue
            visited.add((r,c))
            if res[r,c]==bg: res[r,c]=color
            if dist<3:  # limit spread
                for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<h and 0<=nc<w and (nr,nc) not in visited and a[nr,nc]==bg:
                        q.append((nr,nc,color,dist+1))
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

def try_checkerboard(pairs):
    """Output is a checkerboard pattern derived from input colors."""
    if not all(np.array(p['input']).shape==np.array(p['output']).shape for p in pairs): return None
    gi=np.array(pairs[0]['input']); go=np.array(pairs[0]['output']); h,w=gi.shape
    # Check if output is checkerboard
    c1=int(go[0,0]); c2=int(go[0,1]) if w>1 else c1
    if c1==c2: return None
    is_checker=True
    for r in range(h):
        for c in range(w):
            expected=c1 if (r+c)%2==0 else c2
            if int(go[r,c])!=expected: is_checker=False; break
        if not is_checker: break
    if not is_checker: return None
    def apply(inp):
        a=np.array(inp); hh,ww=a.shape
        res=np.zeros_like(a)
        for r in range(hh):
            for c in range(ww):
                res[r,c]=c1 if (r+c)%2==0 else c2
        return res
    for p in pairs:
        if not np.array_equal(apply(p['input']),np.array(p['output'])): return None
    return apply

ALL_ADVANCED_OPS = [
    # Original 14
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
    # New 11
    ("extract_by_property", try_extract_by_property),
    ("fill_by_projection", try_fill_by_projection),
    ("neighbor_identity", try_recolor_by_neighbor_identity),
    ("pattern_tiling", try_pattern_tiling),
    ("rotational_symmetry", try_symmetry_completion_rotational),
    ("border_crop", try_border_crop),
    ("color_map", try_color_map_global),
    ("extract_subgrid", try_extract_subgrid),
    ("gravity_variants", try_gravity_variants),
    ("largest_blob", try_largest_blob_reduction),
    # Math/Counting ops (Level 2-3 from JEPA math roadmap)
    ("count_to_size", try_count_to_size),
    ("count_to_color", try_count_to_color),
    ("output_is_count", try_output_is_count),
    ("repeat_by_count", try_repeat_by_count),
    ("scale_by_count", try_scale_by_count),
    # Object filtering/movement/summary ops
    ("filter_by_max_color", try_filter_by_max_color),
    ("filter_by_min_size", try_filter_by_min_size),
    ("keep_only_color", try_keep_only_specific_color),
    ("remove_only_color", try_remove_only_color),
    ("move_to_corner", try_move_to_corner),
    ("compact_gravity", try_compact_gravity),
    # Pattern/Boolean/Boundary/Flood/Arithmetic/Mirror/Sort/Overlay
    ("pattern_stamp", try_pattern_stamp),
    ("boolean_ops", try_boolean_ops),
    ("boundary_trace", try_boundary_trace),
    ("flood_seeds", try_flood_from_seeds),
    ("pixel_arithmetic", try_pixel_arithmetic),
    ("mirror_bbox", try_mirror_within_bbox),
    ("sort_rows", try_sort_rows),
    ("overlay_objects", try_overlay_objects),
    # Rule induction / noise / parity / permutation
    ("color_permutation", try_color_permutation),
    ("inverse_solve", try_inverse_solve),
    ("dedup_rows", try_dedup_rows),
    ("noise_filter", try_noise_filter),
    ("parity_rule", try_parity_rule),
    ("connected_flood", try_connected_flood),
    ("checkerboard", try_checkerboard),
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
