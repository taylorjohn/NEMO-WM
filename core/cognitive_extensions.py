"""
cognitive_extensions.py — Three New Cognitive Features
========================================================
1. PROSPECTIVE MEMORY (Q20)
   "Remember to do X when Y happens."
   Store future intentions as schema triggers. When the trigger
   condition is met, the stored action plan activates.
   Reference: Einstein & McDaniel (2005) — prospective memory
   relies on PFC-hippocampal loop for intention maintenance.

2. EMOTIONAL TAGGING
   CRT (stress) and DA (surprise) tag episodic memories for
   enhanced consolidation. High-emotion episodes consolidate
   5× faster during sleep. Amygdala-hippocampal interaction.
   Reference: McGaugh (2004) — emotional arousal enhances
   memory consolidation via amygdala modulation.

3. RECONSOLIDATION VULNERABILITY
   Retrieved memories become temporarily labile and can be
   updated with new information. This is how the system
   corrects false memories and integrates new context.
   Reference: Nader et al. (2000) — reactivated memories
   require protein synthesis to restabilize.

Usage:
    python cognitive_extensions.py          # full demo
    python cognitive_extensions.py --test   # run tests
"""

import argparse
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

D_BELIEF = 64


# ──────────────────────────────────────────────────────────────────────────────
# 1. PROSPECTIVE MEMORY — "Remember to do X when Y happens"
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Intention:
    """A stored future intention."""
    name: str
    trigger_schema: int           # activate when this schema is reached
    trigger_belief: np.ndarray    # or when belief is near this
    trigger_threshold: float      # cosine similarity threshold
    action_plan: List[np.ndarray] # actions to execute when triggered
    priority: float               # DA at encoding time
    created_step: int
    expiry_step: int              # forget after this step (-1 = never)
    fired: bool = False
    fire_step: int = -1


class ProspectiveMemory:
    """
    Q20: "What do I need to remember to do?"

    Stores future intentions and monitors for trigger conditions.
    When the agent reaches a state matching the trigger, the
    stored action plan activates automatically.

    Einstein & McDaniel (2005): prospective memory = remembering
    to perform a planned action in the future. Relies on
    PFC maintaining the intention + hippocampus detecting the cue.

    Two retrieval modes:
    - Monitoring: continuously check all intentions (costly but reliable)
    - Spontaneous: trigger fires only when cue is salient (efficient)
    """

    def __init__(self, max_intentions=20):
        self.intentions: List[Intention] = []
        self.max = max_intentions
        self.fired_log: List[Dict] = []
        self.check_count = 0
        self.total_checks = 0

    def set_intention(self, name: str, trigger_belief: np.ndarray,
                       action_plan: List[np.ndarray],
                       trigger_schema: int = -1,
                       priority: float = 0.5,
                       current_step: int = 0,
                       expiry_steps: int = -1) -> Intention:
        """Store a future intention."""
        threshold = 0.7  # cosine similarity to trigger

        intention = Intention(
            name=name,
            trigger_schema=trigger_schema,
            trigger_belief=trigger_belief,
            trigger_threshold=threshold,
            action_plan=action_plan,
            priority=priority,
            created_step=current_step,
            expiry_step=current_step + expiry_steps if expiry_steps > 0 else -1,
        )

        self.intentions.append(intention)

        # Prune low-priority if over capacity
        if len(self.intentions) > self.max:
            self.intentions.sort(key=lambda i: -i.priority)
            self.intentions = self.intentions[:self.max]

        return intention

    def check(self, current_belief: np.ndarray,
              current_schema: int = -1,
              current_step: int = 0) -> List[Intention]:
        """
        Check all active intentions against current state.
        Returns list of triggered intentions.
        """
        self.total_checks += 1
        triggered = []

        for intention in self.intentions:
            if intention.fired:
                continue

            # Check expiry
            if intention.expiry_step > 0 and current_step > intention.expiry_step:
                continue

            # Check schema trigger
            if intention.trigger_schema >= 0:
                if current_schema == intention.trigger_schema:
                    intention.fired = True
                    intention.fire_step = current_step
                    triggered.append(intention)
                    continue

            # Check belief similarity trigger
            sim = np.dot(current_belief, intention.trigger_belief) / (
                np.linalg.norm(current_belief) *
                np.linalg.norm(intention.trigger_belief) + 1e-8)

            if sim > intention.trigger_threshold:
                intention.fired = True
                intention.fire_step = current_step
                triggered.append(intention)

        self.check_count += len(triggered)

        for t in triggered:
            self.fired_log.append({
                "name": t.name,
                "created": t.created_step,
                "fired": t.fire_step,
                "delay": t.fire_step - t.created_step,
                "priority": t.priority,
            })

        return triggered

    def active_intentions(self, current_step=0) -> List[Intention]:
        """Return unfired, unexpired intentions."""
        return [i for i in self.intentions
                if not i.fired and
                (i.expiry_step < 0 or current_step <= i.expiry_step)]

    def forget_expired(self, current_step: int) -> int:
        """Remove expired intentions."""
        before = len(self.intentions)
        self.intentions = [i for i in self.intentions
                           if i.expiry_step < 0 or
                           current_step <= i.expiry_step or
                           i.fired]
        return before - len(self.intentions)

    def stats(self) -> Dict:
        return {
            "total_set": len(self.intentions),
            "fired": sum(1 for i in self.intentions if i.fired),
            "active": sum(1 for i in self.intentions if not i.fired),
            "total_checks": self.total_checks,
            "avg_delay": (np.mean([f["delay"] for f in self.fired_log])
                         if self.fired_log else 0),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. EMOTIONAL TAGGING — Amygdala-gated consolidation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TaggedEpisode:
    """Episode with emotional tags."""
    beliefs: List[np.ndarray]
    rewards: List[float]
    da_mean: float          # average surprise
    crt_mean: float         # average stress
    emotional_intensity: float  # combined tag strength
    consolidation_priority: float
    times_consolidated: int = 0
    last_consolidated: int = -1


class EmotionalTagging:
    """
    Amygdala-inspired emotional tagging of episodic memories.

    McGaugh (2004): "Memory consolidation is modulated by
    emotional arousal acting through the amygdala."

    High DA (surprise) + High CRT (stress) = strong emotional tag.
    Strongly tagged episodes consolidate 5× faster during sleep.
    This explains why traumatic and surprising events are remembered
    vividly while mundane events fade.

    Tagging mechanism:
    - emotional_intensity = sqrt(DA² + CRT²) / sqrt(2)
    - consolidation_priority = intensity × recency_weight
    - High priority → replayed more during sleep → stronger schema
    """

    def __init__(self, base_consolidation_rate=1.0):
        self.episodes: List[TaggedEpisode] = []
        self.base_rate = base_consolidation_rate
        self.consolidation_log = []

    def tag_episode(self, beliefs: List[np.ndarray],
                     rewards: List[float],
                     da_signals: List[float],
                     crt_signals: List[float]) -> TaggedEpisode:
        """Tag an episode with emotional intensity."""
        da_mean = np.mean(da_signals)
        crt_mean = np.mean(crt_signals)

        # Emotional intensity: geometric combination of surprise + stress
        intensity = np.sqrt(da_mean**2 + crt_mean**2) / np.sqrt(2)
        intensity = float(np.clip(intensity, 0, 1))

        # Priority: emotional episodes get up to 5× consolidation rate
        priority = self.base_rate + 4.0 * intensity

        episode = TaggedEpisode(
            beliefs=beliefs,
            rewards=rewards,
            da_mean=float(da_mean),
            crt_mean=float(crt_mean),
            emotional_intensity=intensity,
            consolidation_priority=priority,
        )
        self.episodes.append(episode)
        return episode

    def select_for_consolidation(self, n: int,
                                   current_step: int = 0) -> List[TaggedEpisode]:
        """
        Select episodes for sleep consolidation.
        Probability proportional to consolidation_priority.
        """
        if not self.episodes:
            return []

        priorities = np.array([ep.consolidation_priority
                               for ep in self.episodes])
        probs = priorities / (priorities.sum() + 1e-8)

        n_select = min(n, len(self.episodes))
        indices = np.random.choice(len(self.episodes), size=n_select,
                                    replace=False, p=probs)

        selected = []
        for idx in indices:
            ep = self.episodes[idx]
            ep.times_consolidated += 1
            ep.last_consolidated = current_step
            selected.append(ep)

        self.consolidation_log.append({
            "step": current_step,
            "n_selected": n_select,
            "mean_intensity": np.mean([self.episodes[i].emotional_intensity
                                        for i in indices]),
            "mean_priority": np.mean([self.episodes[i].consolidation_priority
                                       for i in indices]),
        })

        return selected

    def stats(self) -> Dict:
        if not self.episodes:
            return {"n_episodes": 0}

        intensities = [ep.emotional_intensity for ep in self.episodes]
        priorities = [ep.consolidation_priority for ep in self.episodes]
        consol_counts = [ep.times_consolidated for ep in self.episodes]

        return {
            "n_episodes": len(self.episodes),
            "mean_intensity": float(np.mean(intensities)),
            "max_intensity": float(np.max(intensities)),
            "mean_priority": float(np.mean(priorities)),
            "emotional_episodes": sum(1 for i in intensities if i > 0.5),
            "neutral_episodes": sum(1 for i in intensities if i <= 0.5),
            "total_consolidations": sum(consol_counts),
            "consolidation_ratio": (
                np.mean([c for c, i in zip(consol_counts, intensities) if i > 0.5])
                / (np.mean([c for c, i in zip(consol_counts, intensities) if i <= 0.5]) + 1e-8)
                if any(i > 0.5 for i in intensities) and any(i <= 0.5 for i in intensities)
                else 0
            ),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. RECONSOLIDATION VULNERABILITY — Retrieved memories become labile
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LabileMemory:
    """A memory in the labile (modifiable) state."""
    original_belief: np.ndarray
    current_belief: np.ndarray
    retrieval_step: int
    labile_until: int           # step when it restabilizes
    modifications: int = 0
    is_labile: bool = True


class ReconsolidationEngine:
    """
    Nader et al. (2000): "Fear memories require protein synthesis
    in the amygdala for reconsolidation after retrieval."

    When a memory is retrieved, it enters a LABILE window where:
    1. New information can UPDATE the memory (learning from correction)
    2. The memory can be WEAKENED (extinction)
    3. The memory can be STRENGTHENED (reinforcement)

    After the labile window closes, the memory restabilizes.
    This is how the system corrects false memories and integrates
    new context into old experiences.

    Implementation:
    - retrieve(belief) → memory enters labile state for N steps
    - update_labile(new_info) → modify the retrieved memory
    - restabilize() → memory becomes stable again (stronger or weaker)
    """

    def __init__(self, labile_window: int = 50, update_rate: float = 0.3):
        self.labile_window = labile_window
        self.update_rate = update_rate
        self.memories: Dict[int, np.ndarray] = {}  # id → belief
        self.labile_memories: Dict[int, LabileMemory] = {}
        self.modification_log: List[Dict] = []
        self._next_id = 0

    def store(self, belief: np.ndarray) -> int:
        """Store a stable memory."""
        mid = self._next_id
        self.memories[mid] = belief.copy()
        self._next_id += 1
        return mid

    def retrieve(self, query: np.ndarray, current_step: int,
                  k: int = 1) -> List[Tuple[int, np.ndarray, float]]:
        """
        Retrieve nearest memories. Retrieved memories become LABILE.
        """
        if not self.memories:
            return []

        # Find nearest
        ids = list(self.memories.keys())
        beliefs = np.stack([self.memories[i] for i in ids])
        sims = np.dot(beliefs, query) / (
            np.linalg.norm(beliefs, axis=1) * np.linalg.norm(query) + 1e-8)

        top_k = np.argsort(-sims)[:k]

        results = []
        for idx in top_k:
            mid = ids[idx]
            belief = self.memories[mid]
            sim = float(sims[idx])

            # Enter labile state
            self.labile_memories[mid] = LabileMemory(
                original_belief=belief.copy(),
                current_belief=belief.copy(),
                retrieval_step=current_step,
                labile_until=current_step + self.labile_window,
            )

            results.append((mid, belief.copy(), sim))

        return results

    def update_labile(self, memory_id: int, new_info: np.ndarray,
                       current_step: int) -> bool:
        """
        Update a labile memory with new information.
        Only works if the memory is in its labile window.
        """
        if memory_id not in self.labile_memories:
            return False

        labile = self.labile_memories[memory_id]

        if current_step > labile.labile_until:
            labile.is_labile = False
            return False

        # Blend old memory with new information
        old = labile.current_belief
        updated = (1 - self.update_rate) * old + self.update_rate * new_info
        labile.current_belief = updated
        labile.modifications += 1

        # Update the stable memory too
        self.memories[memory_id] = updated.copy()

        self.modification_log.append({
            "memory_id": memory_id,
            "step": current_step,
            "shift": float(np.linalg.norm(updated - old)),
            "from_original": float(np.linalg.norm(
                updated - labile.original_belief)),
        })

        return True

    def restabilize(self, current_step: int) -> int:
        """Close labile windows that have expired."""
        restabilized = 0
        expired = []
        for mid, labile in self.labile_memories.items():
            if current_step > labile.labile_until and labile.is_labile:
                labile.is_labile = False
                restabilized += 1
                if not labile.modifications:
                    expired.append(mid)

        for mid in expired:
            del self.labile_memories[mid]

        return restabilized

    def get_labile_count(self) -> int:
        return sum(1 for l in self.labile_memories.values() if l.is_labile)

    def stats(self) -> Dict:
        total_mods = sum(l.modifications for l in self.labile_memories.values())
        shifts = [m["from_original"] for m in self.modification_log]
        return {
            "total_memories": len(self.memories),
            "currently_labile": self.get_labile_count(),
            "total_modifications": total_mods,
            "mean_shift": float(np.mean(shifts)) if shifts else 0,
            "max_shift": float(np.max(shifts)) if shifts else 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    print("=" * 65)
    print("  Three Cognitive Extensions")
    print("  Q20: Prospective Memory")
    print("  Emotional Tagging (McGaugh 2004)")
    print("  Reconsolidation Vulnerability (Nader 2000)")
    print("=" * 65)

    rng = np.random.RandomState(42)

    # ── 1. Prospective Memory ──
    print("\n" + "─" * 65)
    print("  1. PROSPECTIVE MEMORY — Q20")
    print("─" * 65)

    pm = ProspectiveMemory()

    # Set intentions
    charging_belief = rng.randn(D_BELIEF).astype(np.float32)
    charging_belief[:2] = [3.0, 1.0]  # charging station location

    intersection_belief = rng.randn(D_BELIEF).astype(np.float32)
    intersection_belief[:2] = [5.0, 5.0]

    pm.set_intention("recharge_battery",
                      trigger_belief=charging_belief,
                      action_plan=[np.array([0.0, -0.5])],
                      priority=0.9, current_step=0, expiry_steps=500)

    pm.set_intention("report_anomaly_at_intersection",
                      trigger_belief=intersection_belief,
                      action_plan=[np.array([0.0, 0.0])],
                      trigger_schema=3,
                      priority=0.7, current_step=0)

    pm.set_intention("check_room_42",
                      trigger_belief=rng.randn(D_BELIEF).astype(np.float32),
                      action_plan=[np.array([1.0, 0.0])],
                      priority=0.3, current_step=0, expiry_steps=100)

    print(f"\n  Set 3 intentions:")
    for i in pm.intentions:
        print(f"    '{i.name}' priority={i.priority:.1f} "
              f"expires={'step '+str(i.expiry_step) if i.expiry_step > 0 else 'never'}")

    # Simulate navigation
    print(f"\n  Simulating 200 steps...")
    belief = rng.randn(D_BELIEF).astype(np.float32)
    for step in range(200):
        belief += rng.randn(D_BELIEF).astype(np.float32) * 0.1

        # At step 50, approach charging station
        if step == 50:
            belief[:2] = [3.1, 1.1]
            belief += rng.randn(D_BELIEF).astype(np.float32) * 0.05

        # At step 120, reach intersection (schema 3)
        schema = 3 if step == 120 else 0

        triggered = pm.check(belief, current_schema=schema, current_step=step)
        for t in triggered:
            print(f"    Step {step}: TRIGGERED '{t.name}' "
                  f"(set at step {t.created_step}, delay={step - t.created_step})")

    pm.forget_expired(200)
    stats = pm.stats()
    print(f"\n  Stats: {stats['fired']}/{stats['total_set']} fired, "
          f"avg delay={stats['avg_delay']:.0f} steps")

    # ── 2. Emotional Tagging ──
    print("\n" + "─" * 65)
    print("  2. EMOTIONAL TAGGING")
    print("─" * 65)

    et = EmotionalTagging()

    # Generate episodes with varying emotion
    for i in range(20):
        n_steps = 20
        beliefs = [rng.randn(D_BELIEF).astype(np.float32) for _ in range(n_steps)]
        rewards = [float(rng.random()) for _ in range(n_steps)]

        if i < 5:  # emotional episodes (high DA + CRT)
            da = [0.7 + rng.random() * 0.3 for _ in range(n_steps)]
            crt = [0.6 + rng.random() * 0.3 for _ in range(n_steps)]
        elif i < 10:  # surprising only
            da = [0.6 + rng.random() * 0.3 for _ in range(n_steps)]
            crt = [0.1 + rng.random() * 0.1 for _ in range(n_steps)]
        else:  # neutral
            da = [0.1 + rng.random() * 0.2 for _ in range(n_steps)]
            crt = [0.1 + rng.random() * 0.1 for _ in range(n_steps)]

        et.tag_episode(beliefs, rewards, da, crt)

    print(f"\n  Tagged 20 episodes:")
    stats = et.stats()
    print(f"    Emotional (intensity > 0.5): {stats['emotional_episodes']}")
    print(f"    Neutral: {stats['neutral_episodes']}")
    print(f"    Mean intensity: {stats['mean_intensity']:.3f}")

    # Sleep consolidation
    for cycle in range(5):
        selected = et.select_for_consolidation(5, current_step=cycle)
        mean_int = np.mean([s.emotional_intensity for s in selected])
        print(f"    Sleep cycle {cycle+1}: selected {len(selected)}, "
              f"mean intensity={mean_int:.3f}")

    stats = et.stats()
    ratio = stats["consolidation_ratio"]
    print(f"\n  Consolidation ratio (emotional/neutral): {ratio:.1f}×")

    # ── 3. Reconsolidation ──
    print("\n" + "─" * 65)
    print("  3. RECONSOLIDATION VULNERABILITY")
    print("─" * 65)

    rc = ReconsolidationEngine(labile_window=30, update_rate=0.3)

    # Store memories
    for i in range(10):
        b = rng.randn(D_BELIEF).astype(np.float32)
        b[0] = i * 0.5  # each memory at different location
        rc.store(b)

    print(f"\n  Stored {len(rc.memories)} memories")

    # Retrieve (makes memory labile)
    query = rng.randn(D_BELIEF).astype(np.float32)
    query[0] = 2.5
    results = rc.retrieve(query, current_step=10, k=2)
    print(f"  Retrieved {len(results)} memories (now labile)")
    for mid, belief, sim in results:
        print(f"    Memory {mid}: sim={sim:.3f}")

    print(f"  Labile count: {rc.get_labile_count()}")

    # Update during labile window
    new_info = rng.randn(D_BELIEF).astype(np.float32)
    new_info[0] = 3.0
    success = rc.update_labile(results[0][0], new_info, current_step=15)
    print(f"\n  Update at step 15 (within window): {success}")

    success2 = rc.update_labile(results[0][0], new_info * 1.1, current_step=20)
    print(f"  Update at step 20 (within window): {success2}")

    # Try after window closes
    restab = rc.restabilize(current_step=50)
    print(f"  Restabilized at step 50: {restab} memories")

    success3 = rc.update_labile(results[0][0], new_info, current_step=55)
    print(f"  Update at step 55 (after window): {success3}")

    stats = rc.stats()
    print(f"\n  Stats: {stats['total_modifications']} modifications, "
          f"mean shift={stats['mean_shift']:.3f}")

    print(f"\n{'='*65}")
    print(f"  Paper 2 additions:")
    print(f"  Q20: Prospective memory — remember future intentions")
    print(f"  Emotional tagging — 5× consolidation for emotional episodes")
    print(f"  Reconsolidation — retrieved memories modifiable for 50 steps")
    print(f"{'='*65}")


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 65)
    print("  Cognitive Extensions Tests")
    print("=" * 65)
    p = 0; t = 0
    rng = np.random.RandomState(42)

    # ── Prospective Memory ──

    print("\n  T1: Intention fires at correct trigger")
    pm = ProspectiveMemory()
    trigger = rng.randn(D_BELIEF).astype(np.float32)
    pm.set_intention("test", trigger_belief=trigger,
                      action_plan=[np.zeros(2)], priority=0.8)
    # Check with similar belief
    similar = trigger + rng.randn(D_BELIEF).astype(np.float32) * 0.1
    fired = pm.check(similar, current_step=10)
    ok = len(fired) == 1 and fired[0].name == "test"
    print(f"    Fired: {len(fired)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T2: Intention doesn't fire on dissimilar belief")
    pm2 = ProspectiveMemory()
    trigger2 = rng.randn(D_BELIEF).astype(np.float32)
    pm2.set_intention("test2", trigger_belief=trigger2,
                       action_plan=[np.zeros(2)])
    distant = -trigger2  # opposite direction
    fired2 = pm2.check(distant, current_step=5)
    ok = len(fired2) == 0
    print(f"    Fired: {len(fired2)} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T3: Schema trigger works")
    pm3 = ProspectiveMemory()
    pm3.set_intention("schema_test", trigger_belief=rng.randn(D_BELIEF).astype(np.float32),
                       trigger_schema=5, action_plan=[np.zeros(2)])
    fired3 = pm3.check(rng.randn(D_BELIEF).astype(np.float32),
                        current_schema=5, current_step=20)
    ok = len(fired3) == 1
    print(f"    Schema trigger: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T4: Expired intentions don't fire")
    pm4 = ProspectiveMemory()
    trigger4 = rng.randn(D_BELIEF).astype(np.float32)
    pm4.set_intention("expiring", trigger_belief=trigger4,
                       action_plan=[np.zeros(2)], expiry_steps=10, current_step=0)
    fired4 = pm4.check(trigger4, current_step=20)
    ok = len(fired4) == 0
    print(f"    Expired: {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T5: Intention fires only once")
    pm5 = ProspectiveMemory()
    trigger5 = rng.randn(D_BELIEF).astype(np.float32)
    pm5.set_intention("once", trigger_belief=trigger5,
                       action_plan=[np.zeros(2)])
    pm5.check(trigger5 + rng.randn(D_BELIEF).astype(np.float32) * 0.05, current_step=1)
    fired5b = pm5.check(trigger5 + rng.randn(D_BELIEF).astype(np.float32) * 0.05, current_step=2)
    ok = len(fired5b) == 0
    print(f"    Second check: {len(fired5b)} fired {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # ── Emotional Tagging ──

    print("\n  T6: Emotional episodes have higher intensity")
    et = EmotionalTagging()
    emotional_ep = et.tag_episode(
        [rng.randn(D_BELIEF).astype(np.float32) for _ in range(10)],
        [0.5] * 10, [0.8] * 10, [0.7] * 10)
    neutral_ep = et.tag_episode(
        [rng.randn(D_BELIEF).astype(np.float32) for _ in range(10)],
        [0.5] * 10, [0.1] * 10, [0.1] * 10)
    ok = emotional_ep.emotional_intensity > neutral_ep.emotional_intensity
    print(f"    Emotional={emotional_ep.emotional_intensity:.3f} "
          f"> Neutral={neutral_ep.emotional_intensity:.3f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T7: Emotional episodes consolidate more")
    et2 = EmotionalTagging()
    for i in range(20):
        da = [0.8] * 10 if i < 5 else [0.1] * 10
        crt = [0.7] * 10 if i < 5 else [0.1] * 10
        et2.tag_episode(
            [rng.randn(D_BELIEF).astype(np.float32) for _ in range(10)],
            [0.5] * 10, da, crt)
    for _ in range(10):
        et2.select_for_consolidation(5)
    emotional_consol = np.mean([e.times_consolidated for e in et2.episodes[:5]])
    neutral_consol = np.mean([e.times_consolidated for e in et2.episodes[10:]])
    ok = emotional_consol > neutral_consol
    print(f"    Emotional consol={emotional_consol:.1f} "
          f"> Neutral={neutral_consol:.1f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T8: Consolidation priority scales with intensity")
    et3 = EmotionalTagging()
    low = et3.tag_episode([rng.randn(D_BELIEF).astype(np.float32)],
                           [0.5], [0.1], [0.1])
    high = et3.tag_episode([rng.randn(D_BELIEF).astype(np.float32)],
                            [0.5], [0.9], [0.8])
    ok = high.consolidation_priority > low.consolidation_priority
    print(f"    High={high.consolidation_priority:.2f} "
          f"> Low={low.consolidation_priority:.2f} "
          f"{'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    # ── Reconsolidation ──

    print("\n  T9: Retrieved memory becomes labile")
    rc = ReconsolidationEngine(labile_window=20)
    b = rng.randn(D_BELIEF).astype(np.float32)
    mid = rc.store(b)
    rc.retrieve(b, current_step=0)
    ok = rc.get_labile_count() == 1
    print(f"    Labile: {rc.get_labile_count()} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T10: Labile memory can be updated")
    new = rng.randn(D_BELIEF).astype(np.float32)
    ok = rc.update_labile(mid, new, current_step=5)
    print(f"    Updated: {ok} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T11: Update changes the stored memory")
    original = b.copy()
    current = rc.memories[mid]
    shift = np.linalg.norm(current - original)
    ok = shift > 0.1
    print(f"    Shift: {shift:.3f} {'PASS' if ok else 'FAIL'}")
    p += int(ok); t += 1

    print("\n  T12: Memory restabilizes after window")
    rc.restabilize(current_step=30)
    ok2 = not rc.update_labile(mid, new, current_step=35)
    print(f"    Update after window: {not ok2} → blocked "
          f"{'PASS' if ok2 else 'FAIL'}")
    p += int(ok2); t += 1

    print("\n  T13: Unretrieved memories stay stable")
    rc2 = ReconsolidationEngine(labile_window=20)
    mid2 = rc2.store(rng.randn(D_BELIEF).astype(np.float32))
    # Don't retrieve — try to update directly
    ok = not rc2.update_labile(mid2, rng.randn(D_BELIEF).astype(np.float32),
                                current_step=5)
    print(f"    Direct update blocked: {'PASS' if ok else 'FAIL'}")
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
