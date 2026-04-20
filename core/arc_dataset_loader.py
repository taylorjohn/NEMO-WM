"""
NeMo-WM Unified Dataset Loader
Loads training data from ALL available ARC-related sources.

Sources:
1. ARC-AGI-2 Training (1000 tasks, official)
2. RE-ARC (400 generators × 1000 examples = 400K)
3. Google ARC-GEN (900 generators, 100K on Kaggle)
4. ConceptARC (16 concept groups × 10 tasks = 160)
5. DeepMind Mathematics Dataset (school-level math)
6. ARC Dataset Collection (9356 tasks from multiple sources)
7. Our Grid-Math Generator (12 types × N tasks)

Usage:
    loader = ArcDatasetLoader(base_dir='C:\\Users\\MeteorAI\\Desktop')
    
    # Load specific sources
    tasks = loader.load_arc_agi2()           # 1000 official tasks
    tasks = loader.load_re_arc(n_per=10)     # 4000 RE-ARC examples
    tasks = loader.load_concept_arc()        # 160 concept tasks
    tasks = loader.load_arc_collection()     # 9356 community tasks
    tasks = loader.load_grid_math(n=100)     # 1200 grid-math tasks
    
    # Load everything
    all_tasks = loader.load_all()
"""
import json
import os
import glob
import random
from pathlib import Path


class ArcDatasetLoader:
    def __init__(self, base_dir=None):
        """Initialize with base directory containing all repos."""
        if base_dir is None:
            # Auto-detect
            for bd in [r'C:\Users\MeteorAI\Desktop', '/home/claude', '.']:
                if os.path.exists(bd):
                    base_dir = bd
                    break
        self.base_dir = Path(base_dir)
        
        # Standard paths
        self.paths = {
            'arc_agi2': self.base_dir / 'ARC-AGI-2' / 'data' / 'training',
            're_arc': self.base_dir / 're-arc' / 're_arc_data' / 're_arc' / 'tasks',
            're_arc_alt': self.base_dir / 're-arc',
            'arc_gen': self.base_dir / 'ARC-GEN' / 'tasks',
            'concept_arc': self.base_dir / 'ConceptARC' / 'corpus',
            'math_dataset': self.base_dir / 'mathematics_dataset',
            'arc_collection': self.base_dir / 'arc-dataset-collection' / 'dataset',
            'grid_math': self.base_dir / 'CORTEX' / 'grid_math_dataset.json',
        }
    
    def load_arc_agi2(self):
        """Load official ARC-AGI-2 training tasks (1000)."""
        path = self.paths['arc_agi2']
        if not path.exists():
            print(f"  ARC-AGI-2 not found at {path}")
            return {}
        tasks = {}
        for f in sorted(path.glob('*.json')):
            task = json.loads(f.read_text())
            tasks[f.stem] = task
        print(f"  ARC-AGI-2: {len(tasks)} tasks")
        return tasks
    
    def load_re_arc(self, n_per_task=10):
        """Load RE-ARC generated examples (400 tasks × n_per_task)."""
        path = self.paths['re_arc']
        if not path.exists():
            # Try alternative path
            alt = self.paths['re_arc_alt'] / 're_arc.zip'
            if alt.exists():
                print(f"  RE-ARC zip found, unzip first: cd {self.paths['re_arc_alt']} && unzip re_arc.zip")
            else:
                print(f"  RE-ARC not found at {path}")
            return {}
        tasks = {}
        for f in sorted(path.glob('*.json')):
            data = json.loads(f.read_text())
            # data is array of {input, output} pairs
            if len(data) < 4:
                continue
            # Sample n_per_task examples, split into train/test
            examples = data[:min(len(data), n_per_task + 1)]
            task = {
                'train': [{'input': ex['input'], 'output': ex['output']} for ex in examples[:-1]],
                'test': [{'input': examples[-1]['input'], 'output': examples[-1]['output']}]
            }
            tasks[f'rearc_{f.stem}'] = task
        print(f"  RE-ARC: {len(tasks)} tasks ({n_per_task} examples each)")
        return tasks
    
    def load_concept_arc(self):
        """Load ConceptARC tasks (16 concepts × 10 tasks = 160)."""
        path = self.paths['concept_arc']
        if not path.exists():
            print(f"  ConceptARC not found at {path}")
            return {}
        tasks = {}
        for concept_dir in sorted(path.iterdir()):
            if not concept_dir.is_dir():
                continue
            concept = concept_dir.name
            for f in sorted(concept_dir.glob('*.json')):
                task = json.loads(f.read_text())
                tasks[f'concept_{concept}_{f.stem}'] = task
        print(f"  ConceptARC: {len(tasks)} tasks (16 concepts)")
        return tasks
    
    def load_arc_gen_tasks(self):
        """List available ARC-GEN generators (need to run them to get data)."""
        path = self.paths['arc_gen']
        if not path.exists():
            print(f"  ARC-GEN not found at {path}")
            return []
        generators = sorted(path.glob('task_*.py'))
        print(f"  ARC-GEN: {len(generators)} generators available")
        return [g.stem.replace('task_', '') for g in generators]
    
    def load_arc_collection(self, max_per_source=200):
        """Load from neoneye's ARC dataset collection."""
        path = self.paths['arc_collection']
        if not path.exists():
            print(f"  ARC Collection not found at {path}")
            return {}
        tasks = {}
        for source_dir in sorted(path.iterdir()):
            if not source_dir.is_dir():
                continue
            source = source_dir.name
            count = 0
            for f in sorted(source_dir.rglob('*.json')):
                if count >= max_per_source:
                    break
                try:
                    task = json.loads(f.read_text())
                    # Validate it has train/test format
                    if 'train' in task and 'test' in task:
                        tasks[f'col_{source}_{f.stem}'] = task
                        count += 1
                except:
                    pass
            if count > 0:
                print(f"    {source}: {count} tasks")
        print(f"  ARC Collection: {len(tasks)} total tasks")
        return tasks
    
    def load_grid_math(self, n_per_type=50):
        """Load or generate grid-math dataset."""
        path = self.paths['grid_math']
        if path.exists():
            data = json.loads(path.read_text())
            tasks = {item['id']: item['task'] for item in data}
            print(f"  Grid-Math: {len(tasks)} tasks (pre-generated)")
            return tasks
        
        # Generate fresh
        try:
            from arc_grid_math import generate_dataset
            dataset = generate_dataset(n_per_type=n_per_type)
            tasks = {}
            for name, task_list in dataset.items():
                for i, task in enumerate(task_list):
                    tasks[f'math_{name}_{i:04d}'] = task
            print(f"  Grid-Math: {len(tasks)} tasks (generated)")
            return tasks
        except ImportError:
            print("  Grid-Math: generator not available")
            return {}
    
    def load_all(self, re_arc_n=5, collection_max=100):
        """Load all available datasets."""
        print("Loading all datasets:")
        all_tasks = {}
        
        # Official
        all_tasks.update(self.load_arc_agi2())
        
        # RE-ARC
        all_tasks.update(self.load_re_arc(n_per_task=re_arc_n))
        
        # ConceptARC
        all_tasks.update(self.load_concept_arc())
        
        # ARC Collection
        all_tasks.update(self.load_arc_collection(max_per_source=collection_max))
        
        # Grid-Math
        all_tasks.update(self.load_grid_math())
        
        # List ARC-GEN
        self.load_arc_gen_tasks()
        
        print(f"\n{'='*50}")
        print(f"TOTAL: {len(all_tasks)} tasks loaded")
        print(f"{'='*50}")
        
        return all_tasks
    
    def get_stats(self, tasks):
        """Print statistics about loaded tasks."""
        import numpy as np
        
        sources = {}
        for tid in tasks:
            prefix = tid.split('_')[0] if '_' in tid else 'arc'
            sources[prefix] = sources.get(prefix, 0) + 1
        
        print(f"\nDataset Statistics:")
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Sources:")
        for src, count in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"    {src}: {count}")
        
        # Grid size stats
        sizes = []
        for task in tasks.values():
            if 'train' in task and task['train']:
                gi = task['train'][0].get('input', [])
                if gi:
                    sizes.append((len(gi), len(gi[0]) if gi else 0))
        
        if sizes:
            hs, ws = zip(*sizes)
            print(f"  Grid sizes: {min(hs)}–{max(hs)} rows, {min(ws)}–{max(ws)} cols")
            print(f"  Median: {np.median(hs):.0f}×{np.median(ws):.0f}")


if __name__ == '__main__':
    loader = ArcDatasetLoader()
    all_tasks = loader.load_all()
    loader.get_stats(all_tasks)
