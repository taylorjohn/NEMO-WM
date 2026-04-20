"""
NeMo-WM Task Similarity Retrieval
====================================
Encode ARC tasks as compact feature vectors, build a kNN index
of solved tasks, and when a new task arrives, retrieve the most
similar solved tasks and try their programs first.

This is Chollet's Pillar 5: "intelligence is the rate at which
a learner turns experience into new skills."

Feature vector (20-dim):
  Grid features:     input_h, input_w, output_h, output_w
  Size ratios:       h_ratio, w_ratio, area_ratio
  Color features:    n_colors_in, n_colors_out, color_overlap
  Object features:   n_objects, avg_obj_size, max_obj_size, size_variance
  Symmetry:          sym_h, sym_v, sym_180
  Structural:        density, has_dividers, has_frame

Usage:
    from arc_task_similarity import TaskIndex
    
    index = TaskIndex()
    index.add_solved(task, program_chain=['crop', 'fliph'])
    
    # When new task arrives:
    candidates = index.find_similar(new_task, k=5)
    for prog, sim_score in candidates:
        result = apply_program(prog, new_task)
"""
import numpy as np
from collections import Counter
from arc_object_graph import extract_objects


def extract_task_features(task):
    """Extract a 20-dim feature vector from an ARC task."""
    pairs = task['train']
    features = np.zeros(20, dtype=np.float32)
    
    # Average over training pairs
    all_feats = []
    for pair in pairs:
        gi = np.array(pair['input'])
        go = np.array(pair['output'])
        f = _extract_pair_features(gi, go)
        all_feats.append(f)
    
    if all_feats:
        features = np.mean(all_feats, axis=0)
    
    return features


def _extract_pair_features(gi, go):
    """Extract features from a single input-output pair."""
    f = np.zeros(20, dtype=np.float32)
    hi, wi = gi.shape
    ho, wo = go.shape
    
    bg_i = int(np.argmax(np.bincount(gi.flatten())))
    bg_o = int(np.argmax(np.bincount(go.flatten())))
    
    # Grid dimensions (0-3)
    f[0] = hi / 30.0
    f[1] = wi / 30.0
    f[2] = ho / 30.0
    f[3] = wo / 30.0
    
    # Size ratios (4-6)
    f[4] = ho / max(hi, 1)  # height ratio
    f[5] = wo / max(wi, 1)  # width ratio
    f[6] = (ho * wo) / max(hi * wi, 1)  # area ratio
    
    # Color features (7-9)
    colors_in = set(int(v) for v in gi.flatten()) - {bg_i}
    colors_out = set(int(v) for v in go.flatten()) - {bg_o}
    f[7] = len(colors_in) / 9.0
    f[8] = len(colors_out) / 9.0
    overlap = len(colors_in & colors_out) / max(len(colors_in | colors_out), 1)
    f[9] = overlap
    
    # Object features (10-13)
    try:
        objs, _ = extract_objects(gi, bg=bg_i)
        f[10] = len(objs) / 20.0  # n_objects normalized
        if objs:
            sizes = [o.size for o in objs]
            f[11] = np.mean(sizes) / (hi * wi)  # avg size ratio
            f[12] = max(sizes) / (hi * wi)  # max size ratio
            f[13] = np.std(sizes) / max(np.mean(sizes), 1)  # size variance
    except:
        pass
    
    # Symmetry (14-16)
    f[14] = 1.0 if np.array_equal(gi, np.fliplr(gi)) else 0.0
    f[15] = 1.0 if np.array_equal(gi, np.flipud(gi)) else 0.0
    f[16] = 1.0 if np.array_equal(gi, np.rot90(gi, 2)) else 0.0
    
    # Structural (17-19)
    f[17] = np.sum(gi != bg_i) / (hi * wi)  # density
    
    # Has dividers (full row or column of single color)
    has_div = False
    for r in range(hi):
        row = gi[r]
        if len(set(int(v) for v in row)) == 1 and row[0] != bg_i:
            has_div = True
            break
    if not has_div:
        for c in range(wi):
            col = gi[:, c]
            if len(set(int(v) for v in col)) == 1 and col[0] != bg_i:
                has_div = True
                break
    f[18] = 1.0 if has_div else 0.0
    
    # Has frame (border pixels all same non-bg color)
    border = list(gi[0, :]) + list(gi[-1, :]) + list(gi[1:-1, 0]) + list(gi[1:-1, -1])
    border_colors = set(int(v) for v in border) - {bg_i}
    f[19] = 1.0 if len(border_colors) == 1 and border_colors else 0.0
    
    return f


class TaskIndex:
    """kNN index of solved ARC tasks for similarity retrieval."""
    
    def __init__(self):
        self.features = []      # list of 20-dim vectors
        self.programs = []      # list of (task_id, program_chain)
        self.task_ids = []
    
    def add_solved(self, task, task_id, program):
        """Add a solved task to the index."""
        feat = extract_task_features(task)
        self.features.append(feat)
        self.programs.append(program)
        self.task_ids.append(task_id)
    
    def find_similar(self, task, k=5):
        """Find k most similar solved tasks. Returns [(program, distance), ...]."""
        if not self.features:
            return []
        
        query = extract_task_features(task)
        feats = np.array(self.features)
        
        # Euclidean distance
        dists = np.sqrt(np.sum((feats - query) ** 2, axis=1))
        
        top_k = np.argsort(dists)[:k]
        return [(self.programs[i], float(dists[i]), self.task_ids[i]) for i in top_k]
    
    def size(self):
        return len(self.features)
    
    def save(self, path):
        """Save index to JSON."""
        import json
        data = {
            'features': [f.tolist() for f in self.features],
            'programs': self.programs,
            'task_ids': self.task_ids,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path):
        """Load index from JSON."""
        import json, os
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.features = [np.array(f, dtype=np.float32) for f in data['features']]
        self.programs = data['programs']
        self.task_ids = data['task_ids']


def cluster_tasks(tasks, n_clusters=10):
    """Cluster tasks by feature similarity for analysis."""
    feats = np.array([extract_task_features(t) for t in tasks])
    
    # Simple k-means
    from collections import defaultdict
    centers = feats[np.random.choice(len(feats), n_clusters, replace=False)]
    
    for _ in range(20):
        # Assign
        dists = np.array([[np.sqrt(np.sum((f - c) ** 2)) for c in centers] for f in feats])
        labels = np.argmin(dists, axis=1)
        # Update
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = feats[mask].mean(axis=0)
    
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[int(label)].append(i)
    
    return clusters, centers, labels


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    import json, os, sys, time
    
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM Task Similarity Retrieval                          ║")
    print("║  20-dim feature vectors + kNN index (Chollet Pillar 5)       ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else None
    
    if data_dir and os.path.exists(data_dir):
        files = sorted(os.listdir(data_dir))
        tasks = []
        task_ids = []
        
        for f in files:
            if not f.endswith('.json'): continue
            task = json.load(open(os.path.join(data_dir, f)))
            tasks.append(task)
            task_ids.append(f.replace('.json', ''))
        
        print(f"\n  Extracting features for {len(tasks)} tasks...")
        t0 = time.time()
        feats = [extract_task_features(t) for t in tasks]
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.2f}s ({elapsed/len(tasks)*1000:.1f}ms/task)")
        
        # Build index with some known solved tasks
        index = TaskIndex()
        known_programs = {
            '0c786b71': 'rot180→mirror_hv',
            '48131b3c': 'tile_2x2→invert',
            'b94a9452': 'crop→invert',
            'be03b35f': 'top_half→left_half→rot90',
        }
        for tid, prog in known_programs.items():
            idx = task_ids.index(tid) if tid in task_ids else -1
            if idx >= 0:
                index.add_solved(tasks[idx], tid, prog)
        
        print(f"\n  Index built with {index.size()} solved tasks")
        
        # Find similar tasks for a few test cases
        test_cases = ['833dafe3', '7468f01a', '5614dbcf']
        for tid in test_cases:
            idx = task_ids.index(tid) if tid in task_ids else -1
            if idx < 0: continue
            similar = index.find_similar(tasks[idx], k=3)
            print(f"\n  Similar to {tid}:")
            for prog, dist, sim_id in similar:
                print(f"    {sim_id}: {prog} (dist={dist:.3f})")
        
        # Cluster analysis
        print(f"\n  Clustering {len(tasks)} tasks into 8 groups...")
        clusters, centers, labels = cluster_tasks(tasks, n_clusters=8)
        for k, indices in sorted(clusters.items()):
            print(f"    Cluster {k}: {len(indices)} tasks")
            # Show a few task ids
            sample = [task_ids[i] for i in indices[:3]]
            print(f"      sample: {sample}")
    else:
        # Demo with synthetic tasks
        print("\n  Demo: feature extraction")
        task = {
            'train': [
                {'input': [[1,2],[3,4]], 'output': [[3,1],[4,2]]},
                {'input': [[5,6],[7,8]], 'output': [[7,5],[8,6]]},
            ],
            'test': [{'input': [[9,0],[1,2]], 'output': [[1,9],[2,0]]}]
        }
        feat = extract_task_features(task)
        print(f"  Feature vector (20-dim): {feat}")
        print(f"  Grid: h_ratio={feat[4]:.2f}, w_ratio={feat[5]:.2f}, area_ratio={feat[6]:.2f}")
        print(f"  Colors: in={feat[7]:.2f}, out={feat[8]:.2f}, overlap={feat[9]:.2f}")
        print(f"  Objects: n={feat[10]:.2f}, density={feat[17]:.2f}")
