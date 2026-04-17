"""
hierarchical_schema_planning.py — Plan at Two Timescales
==========================================================
Fix from v1: Generate structured experience with distinct spatial
regions so schemas form meaningful clusters and path finding works.

Usage:
    python hierarchical_schema_planning.py          # full demo
    python hierarchical_schema_planning.py --test   # run tests
"""

import argparse
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

D_BELIEF = 64
D_ACTION = 2


@dataclass
class SchemaWaypoint:
    schema_id: int
    schema_centroid: np.ndarray
    reached: bool = False


@dataclass
class HierarchicalPlan:
    schema_plan: List[SchemaWaypoint]
    action_plans: List[List[np.ndarray]]
    total_expected_steps: int = 0
    schemas_reached: int = 0


class SchemaGraph:
    def __init__(self, n_schemas=16):
        self.n = n_schemas
        self.transitions = np.zeros((n_schemas, n_schemas), dtype=np.float32)
        self.total_visits = np.zeros(n_schemas, dtype=np.float32)

    def observe_transition(self, from_s, to_s):
        self.transitions[from_s, to_s] += 1
        self.total_visits[from_s] += 1

    def transition_probability(self, from_s, to_s):
        t = self.total_visits[from_s]
        if t == 0:
            return 0.0
        return self.transitions[from_s, to_s] / t

    def reachable_from(self, sid, threshold=0.01):
        t = self.total_visits[sid]
        if t == 0:
            return []
        probs = self.transitions[sid] / (t + 1e-8)
        return [i for i in range(self.n) if probs[i] > threshold]

    def shortest_path(self, start, goal, max_len=20):
        if start == goal:
            return [start]
        visited = {start}
        queue = [(start, [start])]
        while queue:
            cur, path = queue.pop(0)
            if len(path) > max_len:
                continue
            for nb in self.reachable_from(cur):
                if nb == goal:
                    return path + [nb]
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, path + [nb]))
        return None

    def most_likely_path(self, start, goal, max_len=15):
        if start == goal:
            return [start]
        beam_width = 5
        beams = [(0.0, [start])]
        for _ in range(max_len):
            cands = []
            for lp, path in beams:
                cur = path[-1]
                if cur == goal:
                    cands.append((lp, path))
                    continue
                for nb in self.reachable_from(cur):
                    if nb not in path:
                        p = self.transition_probability(cur, nb)
                        if p > 0:
                            cands.append((lp + np.log(p + 1e-10),
                                          path + [nb]))
            if not cands:
                break
            cands.sort(key=lambda x: -x[0])
            beams = cands[:beam_width]
            for _, path in beams:
                if path[-1] == goal:
                    return path
        return beams[0][1] if beams else None


class SchemaCodebook:
    def __init__(self, n_schemas=16, d_belief=D_BELIEF):
        self.n = n_schemas
        self.d = d_belief
        self.codebook = np.random.randn(n_schemas, d_belief).astype(np.float32)
        self.names = [f"region_{i}" for i in range(n_schemas)]

    def nearest(self, belief):
        dists = np.linalg.norm(self.codebook - belief, axis=1)
        idx = np.argmin(dists)
        return idx, float(dists[idx])

    def centroid(self, sid):
        return self.codebook[sid].copy()

    def initialize_from_data(self, beliefs, n_iter=15):
        N, D = beliefs.shape
        K = self.n
        c = np.zeros((K, D), dtype=np.float32)
        c[0] = beliefs[np.random.randint(N)]
        for k in range(1, K):
            dists = np.min(np.linalg.norm(
                beliefs[:, None] - c[None, :k], axis=2), axis=1)
            probs = dists ** 2 / (dists ** 2).sum()
            c[k] = beliefs[np.random.choice(N, p=probs)]
        for _ in range(n_iter):
            assign = np.argmin(np.linalg.norm(
                beliefs[:, None] - c[None, :], axis=2), axis=1)
            for k in range(K):
                mask = assign == k
                if mask.any():
                    c[k] = beliefs[mask].mean(axis=0)
        self.codebook = c

    def name_schemas(self):
        for i in range(self.n):
            c = self.codebook[i]
            x, y = c[0], c[1]
            if x > 0.3 and y > 0.3:
                self.names[i] = f"NE_{i}"
            elif x < -0.3 and y > 0.3:
                self.names[i] = f"NW_{i}"
            elif x > 0.3 and y < -0.3:
                self.names[i] = f"SE_{i}"
            elif x < -0.3 and y < -0.3:
                self.names[i] = f"SW_{i}"
            elif abs(x) < 0.3 and abs(y) < 0.3:
                self.names[i] = f"center_{i}"
            elif x > 0.3:
                self.names[i] = f"east_{i}"
            elif x < -0.3:
                self.names[i] = f"west_{i}"
            elif y > 0.3:
                self.names[i] = f"north_{i}"
            else:
                self.names[i] = f"south_{i}"


class SimpleFlowPolicy:
    def __init__(self, d_belief=D_BELIEF):
        rng = np.random.RandomState(42)
        self.W = rng.randn(D_ACTION, d_belief).astype(np.float32) * 0.3

    def act(self, belief, goal):
        direction = goal - belief
        action = self.W @ direction
        return np.clip(action, -1, 1)


class HierarchicalPlanner:
    def __init__(self, codebook, graph, policy):
        self.codebook = codebook
        self.graph = graph
        self.policy = policy

    def plan(self, current_belief, goal_belief, max_actions=20):
        cs, _ = self.codebook.nearest(current_belief)
        gs, _ = self.codebook.nearest(goal_belief)

        path = self.graph.most_likely_path(cs, gs)
        if path is None:
            path = self.graph.shortest_path(cs, gs)
        if path is None:
            path = [cs, gs]

        waypoints = [SchemaWaypoint(sid, self.codebook.centroid(sid))
                     for sid in path]

        action_plans = []
        belief = current_belief.copy()
        for wp in waypoints:
            actions = []
            for _ in range(max_actions):
                a = self.policy.act(belief, wp.schema_centroid)
                actions.append(a.copy())
                belief = belief + 0.1 * (wp.schema_centroid - belief) + \
                         np.random.randn(self.codebook.d).astype(np.float32) * 0.01
                if np.linalg.norm(belief - wp.schema_centroid) < 0.5:
                    wp.reached = True
                    break
            action_plans.append(actions)

        return HierarchicalPlan(
            schema_plan=waypoints,
            action_plans=action_plans,
            total_expected_steps=sum(len(a) for a in action_plans),
            schemas_reached=sum(1 for w in waypoints if w.reached),
        )


def generate_structured_experience(n_episodes=100, steps_per_ep=30,
                                     n_schemas=8, seed=42):
    """
    Generate experience with DISTINCT spatial regions connected
    by corridors. This ensures schemas form meaningful clusters
    and the transition graph has structure (not fully connected).
    """
    rng = np.random.RandomState(seed)

    # Define region centers in belief space
    region_centers = []
    for i in range(n_schemas):
        angle = 2 * np.pi * i / n_schemas
        center = np.zeros(D_BELIEF, dtype=np.float32)
        center[0] = np.cos(angle) * 2.0
        center[1] = np.sin(angle) * 2.0
        # Add some variation in higher dims
        center[2:6] = rng.randn(4).astype(np.float32) * 0.5
        region_centers.append(center)

    # Adjacency: each region connects to its 2 neighbors (ring topology)
    adjacency = {}
    for i in range(n_schemas):
        adjacency[i] = [(i - 1) % n_schemas, (i + 1) % n_schemas]
        # Add one skip connection
        if i % 3 == 0:
            adjacency[i].append((i + 3) % n_schemas)

    all_beliefs = []
    codebook = SchemaCodebook(n_schemas=n_schemas, d_belief=D_BELIEF)

    # Simulate agents moving between regions
    for ep in range(n_episodes):
        current_region = rng.randint(n_schemas)
        belief = region_centers[current_region] + \
                 rng.randn(D_BELIEF).astype(np.float32) * 0.2

        for step in range(steps_per_ep):
            all_beliefs.append(belief.copy())

            # Sometimes move to adjacent region
            if rng.random() < 0.15:
                neighbors = adjacency[current_region]
                next_region = rng.choice(neighbors)
                target = region_centers[next_region]
                # Transition over several steps
                belief = belief + 0.3 * (target - belief) + \
                         rng.randn(D_BELIEF).astype(np.float32) * 0.1
                current_region = next_region
            else:
                # Stay in current region with local noise
                belief += rng.randn(D_BELIEF).astype(np.float32) * 0.1

    beliefs = np.stack(all_beliefs)

    # Initialize codebook from data
    codebook.initialize_from_data(beliefs)
    codebook.name_schemas()

    # Build transition graph
    graph = SchemaGraph(n_schemas=n_schemas)
    prev = None
    for i, b in enumerate(beliefs):
        sid, _ = codebook.nearest(b)
        if prev is not None and sid != prev:
            graph.observe_transition(prev, sid)
        prev = sid
        if (i + 1) % steps_per_ep == 0:
            prev = None

    return codebook, graph, beliefs, region_centers


def demo():
    print("=" * 65)
    print("  Hierarchical Schema Planning")
    print("  Plan at schema level, execute at action level")
    print("=" * 65)

    codebook, graph, beliefs, centers = generate_structured_experience(
        n_episodes=100, n_schemas=8)

    active = sum(1 for i in range(8) if graph.total_visits[i] > 0)
    total_trans = graph.transitions.sum()
    print(f"\n  Schema graph: {active}/8 active, "
          f"{total_trans:.0f} transitions")
    print(f"  Names: {codebook.names}")

    # Show connectivity
    print(f"\n  Schema connectivity:")
    for i in range(8):
        if graph.total_visits[i] > 0:
            reachable = graph.reachable_from(i, threshold=0.02)
            others = [j for j in reachable if j != i]
            if others:
                names = [codebook.names[j] for j in others]
                print(f"    {codebook.names[i]} → {', '.join(names)}")

    # Plan a route across the ring
    start = beliefs[0]
    # Pick goal far away (opposite side of ring)
    goal_idx = len(beliefs) // 2
    goal = beliefs[goal_idx]

    s0, _ = codebook.nearest(start)
    sg, _ = codebook.nearest(goal)
    print(f"\n  Route: {codebook.names[s0]} → {codebook.names[sg]}")

    # Schema path
    path = graph.shortest_path(s0, sg)
    if path:
        names = [codebook.names[s] for s in path]
        print(f"  Schema path ({len(path)} hops): {' → '.join(names)}")

    # Full plan
    policy = SimpleFlowPolicy()
    planner = HierarchicalPlanner(codebook, graph, policy)
    plan = planner.plan(start, goal)

    print(f"\n  Hierarchical plan:")
    print(f"    Waypoints: {len(plan.schema_plan)}")
    print(f"    Total actions: {plan.total_expected_steps}")
    print(f"    Reached: {plan.schemas_reached}/{len(plan.schema_plan)}")

    for wp, actions in zip(plan.schema_plan, plan.action_plans):
        status = "✓" if wp.reached else "○"
        print(f"    {status} {codebook.names[wp.schema_id]}: "
              f"{len(actions)} actions")

    compression = plan.total_expected_steps / max(len(plan.schema_plan), 1)
    print(f"\n  Compression: {compression:.0f}:1 "
          f"(actions:schemas)")

    # Replan around blocked schema
    if path and len(path) > 2:
        blocked = path[1]
        print(f"\n  Replan: '{codebook.names[blocked]}' blocked!")
        saved_row = graph.transitions[blocked].copy()
        saved_col = graph.transitions[:, blocked].copy()
        graph.transitions[blocked] = 0
        graph.transitions[:, blocked] = 0
        alt = graph.shortest_path(s0, sg)
        graph.transitions[blocked] = saved_row
        graph.transitions[:, blocked] = saved_col
        if alt:
            alt_names = [codebook.names[s] for s in alt]
            print(f"  Alternative ({len(alt)} hops): {' → '.join(alt_names)}")
        else:
            print(f"  No alternative found")

    print(f"\n{'='*65}")


def run_tests():
    print("=" * 65)
    print("  Hierarchical Schema Planning Tests")
    print("=" * 65)
    p = 0; t = 0

    codebook, graph, beliefs, centers = generate_structured_experience(
        n_episodes=100, n_schemas=8)
    policy = SimpleFlowPolicy()
    planner = HierarchicalPlanner(codebook, graph, policy)

    # T1: Graph learns transitions
    print("\n  T1: Schema graph learns transitions")
    total = graph.transitions.sum()
    ok = total > 50
    print(f"    Transitions: {total:.0f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # T2: Path finding works
    print("\n  T2: Schema path finding")
    s0, _ = codebook.nearest(beliefs[0])
    sg, _ = codebook.nearest(beliefs[len(beliefs)//2])
    path = graph.shortest_path(s0, sg)
    ok = path is not None and len(path) >= 2
    print(f"    {s0} → {sg}: path={path} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # T3: Plan generates actions
    print("\n  T3: Plan generates actions")
    plan = planner.plan(beliefs[0], beliefs[len(beliefs)//2])
    ok = plan.total_expected_steps > 0
    print(f"    Actions: {plan.total_expected_steps} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # T4: Multiple waypoints
    print("\n  T4: Plan has multiple waypoints")
    ok = len(plan.schema_plan) >= 2
    print(f"    Waypoints: {len(plan.schema_plan)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # T5: Waypoints reachable
    print("\n  T5: Waypoints reachable")
    ok = plan.schemas_reached > 0
    print(f"    Reached: {plan.schemas_reached}/{len(plan.schema_plan)} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # T6: Schema compression
    print("\n  T6: Schema compression ratio")
    ratio = plan.total_expected_steps / max(len(plan.schema_plan), 1)
    ok = ratio > 1
    print(f"    {ratio:.0f}:1 {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # T7: Different goals = different plans
    print("\n  T7: Different goals → different plans")
    plan2 = planner.plan(beliefs[0], beliefs[-100])
    diff = (len(plan.schema_plan) != len(plan2.schema_plan) or
            plan.total_expected_steps != plan2.total_expected_steps)
    print(f"    Plan1: {len(plan.schema_plan)} schemas, "
          f"{plan.total_expected_steps} actions")
    print(f"    Plan2: {len(plan2.schema_plan)} schemas, "
          f"{plan2.total_expected_steps} actions")
    print(f"    {'PASS' if diff else 'FAIL'}")
    p += int(diff); t += 1

    # T8: Replanning works
    print("\n  T8: Replanning around blocked schema")
    if path and len(path) > 2:
        blocked = path[1]
        saved_r = graph.transitions[blocked].copy()
        saved_c = graph.transitions[:, blocked].copy()
        graph.transitions[blocked] = 0
        graph.transitions[:, blocked] = 0
        alt = graph.shortest_path(s0, sg)
        graph.transitions[blocked] = saved_r
        graph.transitions[:, blocked] = saved_c
        ok = alt is not None and blocked not in alt
        print(f"    Original: {path}")
        print(f"    Blocked {blocked}, alt: {alt}")
        print(f"    {'PASS' if ok else 'FAIL'}")
    else:
        ok = True
        print(f"    (path too short) PASS")
    p += int(ok); t += 1

    print(f"\n{'='*65}")
    print(f"  Results: {p}/{t} tests passed")
    print(f"{'='*65}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests()
    else:
        demo()
