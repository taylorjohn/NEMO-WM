"""
NeMo-WM DreamCoder-Lite Library Compression
==============================================
After solving tasks, analyze discovered programs to find
common sub-chains and promote them to named primitives.

This is Chollet's Pillar 3: "the ability to learn new abstractions
from experience and reuse them."

DreamCoder's wake-sleep loop simplified:
  WAKE:  Search for programs (beam search, compose)
  SLEEP: Compress discovered programs into reusable abstractions

Example:
  If beam search discovers:
    task_A: crop -> uniq_rows -> uniq_cols
    task_B: extract_lg -> uniq_rows -> uniq_cols
    task_C: rm_bg_rows -> uniq_rows -> uniq_cols
  
  Then "uniq_rows -> uniq_cols" is a common sub-chain.
  Promote it to a named primitive: "dedup_2d"
  
  Next search uses "dedup_2d" as a single step, enabling
  depth-1 search to find what previously needed depth-3.

Usage:
    from arc_library_compress import LibraryCompressor
    
    compressor = LibraryCompressor()
    compressor.add_program('task_A', ['crop', 'uniq_rows', 'uniq_cols'])
    compressor.add_program('task_B', ['extract_lg', 'uniq_rows', 'uniq_cols'])
    
    new_prims = compressor.compress(min_frequency=2)
    # Returns: {'dedup_2d': ['uniq_rows', 'uniq_cols']}
"""
import numpy as np
from collections import Counter, defaultdict
from arc_beam_search import PRIMITIVES, apply_chain


# ═══════════════════════════════════════════════════════════
# LIBRARY COMPRESSOR
# ═══════════════════════════════════════════════════════════

class LibraryCompressor:
    """
    Analyze solved programs, find common sub-chains,
    and create compressed primitives.
    """

    def __init__(self):
        self.programs = {}  # task_id -> chain list
        self.compressed = {}  # name -> chain

    def add_program(self, task_id, chain):
        """Add a discovered program."""
        self.programs[task_id] = list(chain)

    def load_from_results(self, results):
        """Load programs from unified solver results.
        results: dict of {filename: method_string}
        """
        for filename, method in results.items():
            # Parse method strings like "BEAM:crop→invert" or "COMPOSE:rot180→mirror_hv"
            if ':' in method:
                parts = method.split(':', 1)
                if len(parts) == 2:
                    chain_str = parts[1]
                    chain = chain_str.replace('→', '->').split('->')
                    chain = [c.strip() for c in chain if c.strip()]
                    if len(chain) >= 1:
                        tid = filename.replace('.json', '')
                        self.programs[tid] = chain

    def find_common_subchains(self, min_length=2, min_frequency=2):
        """Find sub-chains that appear in multiple programs."""
        subchain_counts = Counter()
        subchain_tasks = defaultdict(list)

        for tid, chain in self.programs.items():
            if len(chain) < min_length:
                continue
            # Extract all sub-chains of length min_length to len(chain)
            for length in range(min_length, len(chain) + 1):
                for start in range(len(chain) - length + 1):
                    sub = tuple(chain[start:start + length])
                    subchain_counts[sub] += 1
                    subchain_tasks[sub].append(tid)

        # Filter by frequency
        common = {
            sub: count
            for sub, count in subchain_counts.items()
            if count >= min_frequency
        }

        # Sort by utility: frequency × length (longer common chains = more compression)
        ranked = sorted(common.items(), key=lambda x: x[1] * len(x[0]), reverse=True)
        return ranked, subchain_tasks

    def compress(self, min_frequency=2):
        """
        Find common sub-chains and create compressed primitives.
        Returns dict: {compressed_name: chain_list}
        """
        ranked, tasks = self.find_common_subchains(min_frequency=min_frequency)

        compressed = {}
        used_names = set()

        for subchain, count in ranked:
            # Generate a meaningful name
            name = self._generate_name(subchain, used_names)
            if name:
                compressed[name] = list(subchain)
                used_names.add(name)
                # Create the actual function
                chain_list = list(subchain)
                fn = lambda g, c=chain_list: apply_chain(g, c)
                self.compressed[name] = {
                    'chain': chain_list,
                    'fn': fn,
                    'frequency': count,
                    'tasks': tasks[subchain],
                }

        return compressed

    def _generate_name(self, subchain, used_names):
        """Generate a meaningful name for a sub-chain."""
        parts = list(subchain)

        # Known useful combinations
        name_map = {
            ('uniq_rows', 'uniq_cols'): 'dedup_2d',
            ('uniq_cols', 'uniq_rows'): 'dedup_2d_cr',
            ('crop', 'fliph'): 'crop_fliph',
            ('crop', 'flipv'): 'crop_flipv',
            ('crop', 'invert'): 'crop_invert',
            ('crop', 'tile_1x2'): 'crop_tile_h',
            ('crop', 'tile_2x1'): 'crop_tile_v',
            ('crop', 'tile_2x2'): 'crop_tile_4',
            ('rot180', 'mirror_hv'): 'rot_mirror_4',
            ('flipv', 'mirror_v'): 'flip_mirror_v',
            ('fliph', 'mirror_h'): 'flip_mirror_h',
            ('tile_2x2', 'invert'): 'tile_invert',
            ('crop', 'up_2x'): 'crop_scale2',
            ('crop', 'up_3x'): 'crop_scale3',
            ('extract_lg', 'crop'): 'extract_crop',
            ('grav_left', 'sort_rows'): 'sort_grav',
            ('sym_h', 'left_half'): 'sym_crop_h',
            ('num_keep_min', 'crop'): 'keep_min_crop',
            ('num_keep_min', 'num_NxN_maj'): 'min_to_NxN',
            ('top_half', 'left_half'): 'quarter_tl',
            ('crop', 'top_half', 'left_half'): 'crop_quarter',
            ('flipv', 'down_3x', 'flipv'): 'vflip_down3_vflip',
            ('fliph', 'mirror_h', 'tile_1x2'): 'flip_mirror_tile',
            ('extract_lg', 'rm_border', 'add_border'): 'reborder_lg',
            ('grav_down', 'down_3x', 'up_3x'): 'grav_resample',
            ('rot270', 'top_half', 'left_half'): 'rot_quarter',
            ('crop', 'uniq_rows', 'uniq_cols'): 'crop_dedup',
        }

        key = tuple(parts)
        if key in name_map and name_map[key] not in used_names:
            return name_map[key]

        # Fallback: join abbreviations
        abbrev = '_'.join(p[:4] for p in parts)
        if abbrev not in used_names:
            return f'lib_{abbrev}'

        return None

    def register_compressed(self, primitives_dict):
        """Register compressed primitives into an existing primitives dict."""
        added = 0
        for name, info in self.compressed.items():
            if name not in primitives_dict:
                primitives_dict[name] = info['fn']
                added += 1
        return added

    def report(self):
        """Print compression report."""
        print(f"\n{'═'*60}")
        print(f"  LIBRARY COMPRESSION REPORT")
        print(f"{'═'*60}")
        print(f"  Programs analyzed: {len(self.programs)}")
        print(f"  Compressed primitives: {len(self.compressed)}")

        if self.compressed:
            print(f"\n  New primitives:")
            for name, info in sorted(self.compressed.items(),
                                      key=lambda x: -x[1]['frequency']):
                chain = ' → '.join(info['chain'])
                print(f"    {name}: {chain} (used {info['frequency']}x in {info['tasks']})")

        # Show original programs
        if self.programs:
            print(f"\n  Original programs ({len(self.programs)}):")
            for tid, chain in sorted(self.programs.items()):
                print(f"    {tid}: {' → '.join(chain)}")


# ═══════════════════════════════════════════════════════════
# STANDALONE TEST
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  NeMo-WM DreamCoder-Lite Library Compression                ║")
    print("║  Learn reusable abstractions from solved programs            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Load the 22 discovered compose chains
    known_programs = {
        '0b148d64': ['num_keep_min', 'crop'],
        '0c786b71': ['rot180', 'mirror_hv'],
        '1a2e2828': ['num_keep_min', 'num_NxN_maj'],
        '1c786137': ['extract_lg', 'crop'],
        '2013d3e2': ['crop', 'top_half', 'left_half'],
        '28bf18c6': ['crop', 'tile_1x2'],
        '445eab21': ['num_NxN_maj'],
        '4c4377d9': ['flipv', 'mirror_v'],
        '5582e5ca': ['grav_down', 'down_3x', 'up_3x'],
        '5614dbcf': ['flipv', 'down_3x', 'flipv'],
        '59341089': ['fliph', 'mirror_h', 'tile_1x2'],
        '73182012': ['crop', 'top_half', 'left_half'],
        '7468f01a': ['fliph', 'crop'],
        '7b7f7511': ['uniq_rows', 'uniq_cols'],
        '833dafe3': ['rot180', 'mirror_hv'],
        '90c28cc7': ['crop', 'uniq_rows', 'uniq_cols'],
        'be03b35f': ['rot270', 'top_half', 'left_half'],
        'beb8660c': ['grav_left', 'sort_rows'],
        'e1baa8a4': ['uniq_rows', 'uniq_cols'],
        'e3497940': ['sym_h', 'left_half'],
        'f25fbde4': ['crop', 'up_2x'],
        'fcb5c309': ['extract_lg', 'rm_border', 'add_border'],
        # Beam search discoveries
        '48131b3c': ['tile_2x2', 'invert'],
        'b94a9452': ['crop', 'invert'],
    }

    compressor = LibraryCompressor()
    for tid, chain in known_programs.items():
        compressor.add_program(tid, chain)

    # Compress
    compressed = compressor.compress(min_frequency=2)
    compressor.report()

    # Show what the search would look like with compressed library
    print(f"\n{'═'*60}")
    print(f"  SEARCH EFFICIENCY GAIN")
    print(f"{'═'*60}")

    base_prims = len(PRIMITIVES)
    new_prims = len(compressed)
    print(f"  Original primitives: {base_prims}")
    print(f"  Compressed primitives: {new_prims}")
    print(f"  New total: {base_prims + new_prims}")
    print(f"\n  Depth reduction examples:")
    for name, info in compressor.compressed.items():
        chain = info['chain']
        print(f"    {name} = {' → '.join(chain)}")
        print(f"      Before: depth {len(chain)} search needed")
        print(f"      After:  depth 1 (single primitive)")
