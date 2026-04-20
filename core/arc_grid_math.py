"""
ARC Grid-Math Dataset Generator
Generates grid-based math puzzles for training numerical reasoning.
Covers: counting, comparison, modular arithmetic, parameterized transforms.

Each task is ARC-format: {'train': [{input, output}, ...], 'test': [{input, output}]}
"""
import numpy as np
import random
import json
from collections import defaultdict

# ARC color palette
COLORS = list(range(1, 10))  # 1-9, 0 is background


def random_grid(h, w, density=0.3, n_colors=3):
    """Generate a random grid with given density of non-bg pixels."""
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(COLORS, min(n_colors, 9))
    for r in range(h):
        for c in range(w):
            if random.random() < density:
                grid[r, c] = random.choice(colors)
    return grid


def random_objects(h, w, n_objs, sizes=(1, 4)):
    """Place n_objs random rectangular objects on a grid."""
    grid = np.zeros((h, w), dtype=int)
    objs = []
    for _ in range(n_objs):
        color = random.choice(COLORS)
        oh = random.randint(sizes[0], sizes[1])
        ow = random.randint(sizes[0], sizes[1])
        r = random.randint(0, h - oh)
        c = random.randint(0, w - ow)
        grid[r:r+oh, c:c+ow] = color
        objs.append({'color': color, 'r': r, 'c': c, 'h': oh, 'w': ow})
    return grid, objs


# ═══════════════════════════════════════════════════════════
# COUNTING TASKS
# ═══════════════════════════════════════════════════════════

def gen_count_objects_to_1x1():
    """Input: grid with N objects. Output: 1×1 grid with value N."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(5, 10), random.randint(5, 10)
        n = random.randint(2, 8)
        grid, _ = random_objects(h, w, n, sizes=(1, 2))
        # Count actual connected components
        from scipy.ndimage import label
        labeled, n_actual = label(grid > 0)
        pairs.append({
            'input': grid.tolist(),
            'output': [[n_actual]]
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_count_colors():
    """Input: grid with K colors. Output: 1×1 with value K."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(4, 8), random.randint(4, 8)
        k = random.randint(2, 6)
        colors = random.sample(COLORS, k)
        grid = np.zeros((h, w), dtype=int)
        for color in colors:
            n_pixels = random.randint(2, 5)
            for _ in range(n_pixels):
                r, c = random.randint(0, h-1), random.randint(0, w-1)
                grid[r, c] = color
        actual_k = len(set(int(v) for v in grid.flatten()) - {0})
        pairs.append({
            'input': grid.tolist(),
            'output': [[actual_k]]
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_count_determines_color():
    """Input: N objects (all same color). Output: all recolored to color N."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(6, 10), random.randint(6, 10)
        n = random.randint(2, 8)
        grid, _ = random_objects(h, w, n, sizes=(1, 2))
        # Recolor everything to color = n (clamped to 1-9)
        n_clamped = min(n, 9)
        output = np.where(grid > 0, n_clamped, 0)
        pairs.append({
            'input': grid.tolist(),
            'output': output.tolist()
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


# ═══════════════════════════════════════════════════════════
# COMPARISON TASKS
# ═══════════════════════════════════════════════════════════

def gen_keep_largest_object():
    """Input: multiple objects. Output: only the largest survives."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(6, 10), random.randint(6, 10)
        n = random.randint(3, 5)
        grid, objs = random_objects(h, w, n, sizes=(1, 3))
        # Find largest
        from scipy.ndimage import label
        labeled, n_comp = label(grid > 0)
        sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        if not sizes: continue
        largest_id = np.argmax(sizes) + 1
        output = np.where(labeled == largest_id, grid, 0)
        pairs.append({
            'input': grid.tolist(),
            'output': output.tolist()
        })
    if len(pairs) < 4: return None
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_sort_by_size():
    """Input: objects in random order. Output: objects sorted by size L→R."""
    pairs = []
    for _ in range(4):
        n = random.randint(3, 5)
        max_size = 4
        sizes = sorted([random.randint(1, max_size) for _ in range(n)])
        h = max_size + 2
        w = sum(s + 1 for s in sizes) + 1
        # Build output: objects arranged L→R by size
        output = np.zeros((h, w), dtype=int)
        c_pos = 1
        for s in sizes:
            color = random.choice(COLORS)
            for r in range(h - s, h):
                for c in range(c_pos, c_pos + s):
                    if r < h and c < w:
                        output[r, c] = color
            c_pos += s + 1
        # Build input: same objects but shuffled horizontally
        random.shuffle(sizes)
        inp = np.zeros((h, w), dtype=int)
        c_pos = 1
        for s in sizes:
            color = random.choice(COLORS)
            for r in range(h - s, h):
                for c in range(c_pos, c_pos + s):
                    if r < h and c < w:
                        inp[r, c] = color
            c_pos += s + 1
        pairs.append({
            'input': inp.tolist(),
            'output': output.tolist()
        })
    if len(pairs) < 4: return None
    return {'train': pairs[:3], 'test': pairs[3:]}


# ═══════════════════════════════════════════════════════════
# MODULAR ARITHMETIC TASKS
# ═══════════════════════════════════════════════════════════

def gen_checkerboard():
    """Input: grid size. Output: checkerboard with (r+c)%2 coloring."""
    pairs = []
    c1, c2 = random.sample(COLORS, 2)
    for _ in range(4):
        h, w = random.randint(3, 8), random.randint(3, 8)
        inp = np.zeros((h, w), dtype=int)  # or some marker
        out = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                out[r, c] = c1 if (r + c) % 2 == 0 else c2
        pairs.append({
            'input': inp.tolist(),
            'output': out.tolist()
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_modular_coloring():
    """Recolor pixels based on (r+c) % N."""
    pairs = []
    n = random.randint(2, 4)
    color_map = {i: random.choice(COLORS) for i in range(n)}
    for _ in range(4):
        h, w = random.randint(4, 8), random.randint(4, 8)
        grid = random_grid(h, w, density=0.5, n_colors=2)
        output = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    output[r, c] = color_map[(r + c) % n]
        pairs.append({
            'input': grid.tolist(),
            'output': output.tolist()
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_periodic_stripe():
    """Color rows/cols based on r%N or c%N pattern."""
    pairs = []
    n = random.randint(2, 4)
    stripe_colors = [random.choice(COLORS) for _ in range(n)]
    axis = random.choice(['row', 'col'])
    for _ in range(4):
        h, w = random.randint(4, 10), random.randint(4, 10)
        output = np.zeros((h, w), dtype=int)
        for r in range(h):
            for c in range(w):
                if axis == 'row':
                    output[r, c] = stripe_colors[r % n]
                else:
                    output[r, c] = stripe_colors[c % n]
        # Input: output with some pixels removed
        inp = output.copy()
        for _ in range(h * w // 3):
            r, c = random.randint(0, h-1), random.randint(0, w-1)
            inp[r, c] = 0
        pairs.append({
            'input': inp.tolist(),
            'output': output.tolist()
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


# ═══════════════════════════════════════════════════════════
# PARAMETERIZED TRANSFORM TASKS
# ═══════════════════════════════════════════════════════════

def gen_scale_by_count():
    """Input: small grid with N colored pixels. Output: grid scaled N×."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(2, 4), random.randint(2, 4)
        grid = random_grid(h, w, density=0.4, n_colors=2)
        n = int(np.sum(grid > 0))
        if n < 2 or n > 5:
            n = random.randint(2, 4)
            grid = np.zeros((h, w), dtype=int)
            positions = random.sample([(r, c) for r in range(h) for c in range(w)], n)
            for r, c in positions:
                grid[r, c] = random.choice(COLORS)
        output = np.repeat(np.repeat(grid, n, axis=0), n, axis=1)
        pairs.append({
            'input': grid.tolist(),
            'output': output.tolist()
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_repeat_by_count():
    """Each row is repeated N times where N = number of non-bg pixels in it."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(3, 5), random.randint(3, 6)
        grid = random_grid(h, w, density=0.3, n_colors=3)
        rows = []
        for r in range(h):
            n = max(1, int(np.sum(grid[r] > 0)))
            for _ in range(n):
                rows.append(grid[r].tolist())
        pairs.append({
            'input': grid.tolist(),
            'output': rows
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


# ═══════════════════════════════════════════════════════════
# SEQUENCE / PATTERN TASKS
# ═══════════════════════════════════════════════════════════

def gen_arithmetic_sequence():
    """Row contains arithmetic sequence with one missing → fill it."""
    pairs = []
    for _ in range(4):
        start = random.randint(1, 5)
        step = random.randint(1, 2)
        length = random.randint(4, 7)
        seq = [min(start + i * step, 9) for i in range(length)]
        missing = random.randint(1, length - 2)
        inp = [seq[:]]
        inp[0][missing] = 0
        pairs.append({
            'input': inp,
            'output': [seq]
        })
    return {'train': pairs[:3], 'test': pairs[3:]}


def gen_frequency_determines_output():
    """Most frequent color in input → output is solid grid of that color."""
    pairs = []
    for _ in range(4):
        h, w = random.randint(3, 6), random.randint(3, 6)
        grid = random_grid(h, w, density=0.7, n_colors=3)
        colors = [int(v) for v in grid.flatten() if v > 0]
        if not colors: continue
        from collections import Counter
        most_common = Counter(colors).most_common(1)[0][0]
        output = np.full((h, w), most_common, dtype=int)
        pairs.append({
            'input': grid.tolist(),
            'output': output.tolist()
        })
    if len(pairs) < 4: return None
    return {'train': pairs[:3], 'test': pairs[3:]}


# ═══════════════════════════════════════════════════════════
# MASTER GENERATOR
# ═══════════════════════════════════════════════════════════

GENERATORS = {
    # Counting
    'count_objects': gen_count_objects_to_1x1,
    'count_colors': gen_count_colors,
    'count_determines_color': gen_count_determines_color,
    
    # Comparison
    'keep_largest': gen_keep_largest_object,
    'sort_by_size': gen_sort_by_size,
    
    # Modular arithmetic
    'checkerboard': gen_checkerboard,
    'modular_coloring': gen_modular_coloring,
    'periodic_stripe': gen_periodic_stripe,
    
    # Parameterized transforms
    'scale_by_count': gen_scale_by_count,
    'repeat_by_count': gen_repeat_by_count,
    
    # Sequences
    'arithmetic_sequence': gen_arithmetic_sequence,
    'frequency_output': gen_frequency_determines_output,
}


def generate_dataset(n_per_type=100, output_path=None):
    """Generate full grid-math dataset."""
    dataset = {}
    total = 0
    
    for name, gen_fn in GENERATORS.items():
        tasks = []
        attempts = 0
        while len(tasks) < n_per_type and attempts < n_per_type * 3:
            attempts += 1
            try:
                task = gen_fn()
                if task is not None:
                    tasks.append(task)
            except:
                pass
        dataset[name] = tasks
        total += len(tasks)
        print(f"  {name}: {len(tasks)} tasks")
    
    print(f"\nTotal: {total} tasks across {len(GENERATORS)} types")
    
    if output_path:
        # Save as flat list with type labels
        flat = []
        for name, tasks in dataset.items():
            for i, task in enumerate(tasks):
                flat.append({
                    'id': f'{name}_{i:04d}',
                    'type': name,
                    'task': task
                })
        with open(output_path, 'w') as f:
            json.dump(flat, f, indent=2)
        print(f"Saved to {output_path}")
    
    return dataset


if __name__ == '__main__':
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    out = sys.argv[2] if len(sys.argv) > 2 else 'grid_math_dataset.json'
    generate_dataset(n_per_type=n, output_path=out)
