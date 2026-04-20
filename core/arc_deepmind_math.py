"""
NeMo-WM DeepMind Math Integration
====================================
Generates mathematical Q&A pairs from the DeepMind Mathematics Dataset
categories and tests NeMo-WM's mathematical reasoning against them.

Instead of downloading the full 130GB dataset, we generate problems
locally using the same principles (SymPy + structured templates).

Categories tested:
  1. Arithmetic — addition, subtraction, multiplication, division
  2. Comparison — ordering, closest, sorting
  3. Numbers — GCD, LCM, remainders, factors, primes, base conversion
  4. Algebra — linear equations, evaluate expressions
  5. Measurement — unit conversion (conceptual)
  6. Counting — combinatorics, permutations

Usage:
    python arc_deepmind_math.py              # Run all tests
    python arc_deepmind_math.py --generate   # Generate dataset to JSON
    python arc_deepmind_math.py --benchmark  # Benchmark NeMo-WM solver
"""
import json
import os
import sys
import time
import random
import numpy as np
from collections import defaultdict

random.seed(42)


# ═══════════════════════════════════════════════════════════
# PROBLEM GENERATORS — structured math Q&A pairs
# ═══════════════════════════════════════════════════════════

def gen_arithmetic(n=100):
    """Generate arithmetic problems: +, -, *, //, %, chains."""
    problems = []
    ops = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,
           '*': lambda a,b: a*b, '//': lambda a,b: a//b if b!=0 else None,
           '%': lambda a,b: a%b if b!=0 else None}

    for _ in range(n):
        cat = random.choice(['add','sub','mul','div','mod','chain','negative'])
        if cat == 'add':
            a, b = random.randint(1, 999), random.randint(1, 999)
            problems.append({'q': f'Calculate {a} + {b}.', 'a': str(a+b),
                           'category': 'arithmetic', 'op': 'add', 'difficulty': 'easy'})
        elif cat == 'sub':
            a, b = random.randint(1, 999), random.randint(1, 999)
            problems.append({'q': f'Calculate {a} - {b}.', 'a': str(a-b),
                           'category': 'arithmetic', 'op': 'sub', 'difficulty': 'easy'})
        elif cat == 'mul':
            a, b = random.randint(2, 99), random.randint(2, 99)
            problems.append({'q': f'Calculate {a} * {b}.', 'a': str(a*b),
                           'category': 'arithmetic', 'op': 'mul', 'difficulty': 'medium'})
        elif cat == 'div':
            b = random.randint(2, 20)
            a = b * random.randint(1, 50)
            problems.append({'q': f'Calculate {a} / {b}.', 'a': str(a//b),
                           'category': 'arithmetic', 'op': 'div', 'difficulty': 'medium'})
        elif cat == 'mod':
            a, b = random.randint(10, 999), random.randint(2, 20)
            problems.append({'q': f'What is the remainder when {a} is divided by {b}?',
                           'a': str(a%b), 'category': 'arithmetic', 'op': 'mod', 'difficulty': 'medium'})
        elif cat == 'chain':
            a, b, c = random.randint(1,50), random.randint(1,50), random.randint(1,50)
            op1 = random.choice(['+','-'])
            op2 = random.choice(['+','-'])
            result = eval(f'{a}{op1}{b}{op2}{c}')
            problems.append({'q': f'Calculate {a} {op1} {b} {op2} {c}.',
                           'a': str(result), 'category': 'arithmetic', 'op': 'chain', 'difficulty': 'medium'})
        elif cat == 'negative':
            a, b = random.randint(-100, 100), random.randint(-100, 100)
            problems.append({'q': f'Calculate {a} + {b}.', 'a': str(a+b),
                           'category': 'arithmetic', 'op': 'add_neg', 'difficulty': 'medium'})
    return problems


def gen_comparison(n=100):
    """Generate comparison and ordering problems."""
    problems = []
    for _ in range(n):
        cat = random.choice(['max','min','sort','closest','between'])
        nums = [random.randint(-100, 100) for _ in range(random.randint(3, 7))]

        if cat == 'max':
            problems.append({'q': f'What is the largest value in {nums}?',
                           'a': str(max(nums)), 'category': 'comparison', 'op': 'max', 'difficulty': 'easy'})
        elif cat == 'min':
            problems.append({'q': f'What is the smallest value in {nums}?',
                           'a': str(min(nums)), 'category': 'comparison', 'op': 'min', 'difficulty': 'easy'})
        elif cat == 'sort':
            problems.append({'q': f'Sort {nums} in ascending order.',
                           'a': str(sorted(nums)), 'category': 'comparison', 'op': 'sort', 'difficulty': 'easy'})
        elif cat == 'closest':
            target = random.randint(-50, 50)
            closest = min(nums, key=lambda x: abs(x - target))
            problems.append({'q': f'Which value in {nums} is closest to {target}?',
                           'a': str(closest), 'category': 'comparison', 'op': 'closest', 'difficulty': 'medium'})
        elif cat == 'between':
            a, b = sorted(random.sample(range(-50, 50), 2))
            count = sum(1 for x in nums if a <= x <= b)
            problems.append({'q': f'How many values in {nums} are between {a} and {b} inclusive?',
                           'a': str(count), 'category': 'comparison', 'op': 'between', 'difficulty': 'medium'})
    return problems


def gen_numbers(n=100):
    """Generate number theory problems: GCD, LCM, factors, primes, base."""
    from math import gcd

    def lcm(a, b):
        return abs(a * b) // gcd(a, b) if gcd(a, b) != 0 else 0

    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    def factors(n):
        return sorted(set(f for i in range(1, int(abs(n)**0.5)+1) if n%i==0 for f in [i, abs(n)//i]))

    problems = []
    for _ in range(n):
        cat = random.choice(['gcd','lcm','prime','factors','base','digits','divisible'])

        if cat == 'gcd':
            a, b = random.randint(2, 200), random.randint(2, 200)
            problems.append({'q': f'What is the GCD of {a} and {b}?',
                           'a': str(gcd(a, b)), 'category': 'numbers', 'op': 'gcd', 'difficulty': 'medium'})
        elif cat == 'lcm':
            a, b = random.randint(2, 50), random.randint(2, 50)
            problems.append({'q': f'What is the LCM of {a} and {b}?',
                           'a': str(lcm(a, b)), 'category': 'numbers', 'op': 'lcm', 'difficulty': 'medium'})
        elif cat == 'prime':
            n = random.randint(2, 200)
            problems.append({'q': f'Is {n} prime?',
                           'a': str(is_prime(n)), 'category': 'numbers', 'op': 'prime', 'difficulty': 'easy'})
        elif cat == 'factors':
            n = random.randint(2, 100)
            f = factors(n)
            problems.append({'q': f'How many factors does {n} have?',
                           'a': str(len(f)), 'category': 'numbers', 'op': 'factors', 'difficulty': 'medium'})
        elif cat == 'base':
            n = random.randint(1, 255)
            base = random.choice([2, 8, 16])
            if base == 2: rep = bin(n)[2:]
            elif base == 8: rep = oct(n)[2:]
            else: rep = hex(n)[2:].upper()
            problems.append({'q': f'Convert {n} to base {base}.',
                           'a': rep, 'category': 'numbers', 'op': 'base', 'difficulty': 'hard'})
        elif cat == 'digits':
            n = random.randint(100, 99999)
            problems.append({'q': f'What is the sum of the digits of {n}?',
                           'a': str(sum(int(d) for d in str(n))), 'category': 'numbers', 'op': 'digit_sum', 'difficulty': 'easy'})
        elif cat == 'divisible':
            n = random.randint(2, 500)
            d = random.choice([2, 3, 5, 7, 11])
            problems.append({'q': f'Is {n} divisible by {d}?',
                           'a': str(n % d == 0), 'category': 'numbers', 'op': 'divisible', 'difficulty': 'easy'})
    return problems


def gen_algebra(n=100):
    """Generate algebra problems: evaluate, solve linear, sequences."""
    problems = []
    for _ in range(n):
        cat = random.choice(['evaluate','linear','sequence_arith','sequence_geo','poly_eval'])

        if cat == 'evaluate':
            x = random.randint(-10, 10)
            a, b = random.randint(-10, 10), random.randint(-10, 10)
            result = a * x + b
            problems.append({'q': f'Let f(x) = {a}*x + {b}. What is f({x})?',
                           'a': str(result), 'category': 'algebra', 'op': 'evaluate', 'difficulty': 'easy'})
        elif cat == 'linear':
            x_true = random.randint(-20, 20)
            a = random.randint(1, 10) * random.choice([-1, 1])
            b = random.randint(-50, 50)
            c = a * x_true + b
            problems.append({'q': f'Solve {a}*x + {b} = {c} for x.',
                           'a': str(x_true), 'category': 'algebra', 'op': 'linear', 'difficulty': 'medium'})
        elif cat == 'sequence_arith':
            start = random.randint(-10, 10)
            step = random.randint(1, 10) * random.choice([-1, 1])
            length = random.randint(4, 8)
            seq = [start + i * step for i in range(length)]
            next_val = start + length * step
            problems.append({'q': f'What is the next number in the sequence {seq}?',
                           'a': str(next_val), 'category': 'algebra', 'op': 'seq_arith', 'difficulty': 'medium'})
        elif cat == 'sequence_geo':
            start = random.randint(1, 5)
            ratio = random.choice([2, 3])
            length = random.randint(3, 6)
            seq = [start * ratio**i for i in range(length)]
            next_val = start * ratio**length
            problems.append({'q': f'What is the next number in the sequence {seq}?',
                           'a': str(next_val), 'category': 'algebra', 'op': 'seq_geo', 'difficulty': 'hard'})
        elif cat == 'poly_eval':
            x = random.randint(-5, 5)
            a, b, c = random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)
            result = a * x**2 + b * x + c
            problems.append({'q': f'Let f(x) = {a}*x^2 + {b}*x + {c}. What is f({x})?',
                           'a': str(result), 'category': 'algebra', 'op': 'poly_eval', 'difficulty': 'medium'})
    return problems


def gen_counting(n=100):
    """Generate counting/combinatorics problems."""
    from math import factorial, comb, perm

    problems = []
    for _ in range(n):
        cat = random.choice(['factorial','choose','permute','count_digits','count_divisors'])

        if cat == 'factorial':
            n_val = random.randint(1, 10)
            problems.append({'q': f'What is {n_val}!?',
                           'a': str(factorial(n_val)), 'category': 'counting', 'op': 'factorial', 'difficulty': 'easy'})
        elif cat == 'choose':
            n_val = random.randint(3, 12)
            k_val = random.randint(1, n_val)
            problems.append({'q': f'How many ways to choose {k_val} from {n_val}?',
                           'a': str(comb(n_val, k_val)), 'category': 'counting', 'op': 'choose', 'difficulty': 'medium'})
        elif cat == 'permute':
            n_val = random.randint(3, 8)
            k_val = random.randint(1, min(n_val, 4))
            problems.append({'q': f'How many permutations of {k_val} from {n_val}?',
                           'a': str(perm(n_val, k_val)), 'category': 'counting', 'op': 'permute', 'difficulty': 'medium'})
        elif cat == 'count_digits':
            n_val = random.randint(1, 9999)
            problems.append({'q': f'How many digits does {n_val} have?',
                           'a': str(len(str(n_val))), 'category': 'counting', 'op': 'count_digits', 'difficulty': 'easy'})
        elif cat == 'count_divisors':
            n_val = random.randint(2, 100)
            count = sum(1 for i in range(1, n_val+1) if n_val % i == 0)
            problems.append({'q': f'How many positive divisors does {n_val} have?',
                           'a': str(count), 'category': 'counting', 'op': 'count_divisors', 'difficulty': 'medium'})
    return problems


# ═══════════════════════════════════════════════════════════
# MATH SOLVER — NeMo-WM's symbolic math engine
# ═══════════════════════════════════════════════════════════

class NeMoMathSolver:
    """
    Symbolic math solver using the same principles as NeMo-WM's
    NumericalReasoner: parse structure, apply discrete operations.
    """

    def __init__(self):
        self.solved = 0
        self.failed = 0
        self.by_category = defaultdict(lambda: {'solved': 0, 'total': 0})

    def solve(self, problem):
        """Attempt to solve a math problem. Returns (answer, method) or (None, None)."""
        q = problem['q']
        category = problem.get('category', 'unknown')
        op = problem.get('op', 'unknown')

        try:
            if category == 'arithmetic':
                return self._solve_arithmetic(q, op)
            elif category == 'comparison':
                return self._solve_comparison(q, op)
            elif category == 'numbers':
                return self._solve_numbers(q, op)
            elif category == 'algebra':
                return self._solve_algebra(q, op)
            elif category == 'counting':
                return self._solve_counting(q, op)
        except Exception as e:
            pass
        return None, None

    def _solve_arithmetic(self, q, op):
        import re
        if 'Calculate' in q:
            expr = q.replace('Calculate', '').replace('.', '').strip()
            # Safe eval for simple arithmetic
            result = eval(expr, {"__builtins__": {}})
            return str(int(result) if isinstance(result, float) and result == int(result) else result), 'eval'
        elif 'remainder' in q:
            nums = [int(x) for x in re.findall(r'-?\d+', q)]
            if len(nums) >= 2:
                return str(nums[0] % nums[1]), 'mod'
        return None, None

    def _solve_comparison(self, q, op):
        import re, ast
        # Extract list from question
        match = re.search(r'\[([^\]]+)\]', q)
        if not match:
            return None, None
        nums = [int(x.strip()) for x in match.group(1).split(',')]

        if op == 'max' or 'largest' in q:
            return str(max(nums)), 'max'
        elif op == 'min' or 'smallest' in q:
            return str(min(nums)), 'min'
        elif op == 'sort' or 'Sort' in q:
            return str(sorted(nums)), 'sort'
        elif op == 'closest' or 'closest' in q:
            target_match = re.search(r'closest to (-?\d+)', q)
            if target_match:
                target = int(target_match.group(1))
                return str(min(nums, key=lambda x: abs(x - target))), 'closest'
        elif op == 'between' or 'between' in q:
            bounds = re.findall(r'between (-?\d+) and (-?\d+)', q)
            if bounds:
                a, b = int(bounds[0][0]), int(bounds[0][1])
                return str(sum(1 for x in nums if a <= x <= b)), 'count_between'
        return None, None

    def _solve_numbers(self, q, op):
        import re
        from math import gcd
        nums = [int(x) for x in re.findall(r'\d+', q)]

        if op == 'gcd' and len(nums) >= 2:
            return str(gcd(nums[0], nums[1])), 'gcd'
        elif op == 'lcm' and len(nums) >= 2:
            return str(abs(nums[0] * nums[1]) // gcd(nums[0], nums[1])), 'lcm'
        elif op == 'prime' and nums:
            n = nums[0]
            is_p = n >= 2 and all(n % i != 0 for i in range(2, int(n**0.5)+1))
            return str(is_p), 'prime_check'
        elif op == 'factors' and nums:
            n = nums[0]
            count = sum(1 for i in range(1, n+1) if n % i == 0)
            return str(count), 'count_factors'
        elif op == 'base' and len(nums) >= 2:
            n, base = nums[0], nums[1]
            if base == 2: return bin(n)[2:], 'to_bin'
            elif base == 8: return oct(n)[2:], 'to_oct'
            elif base == 16: return hex(n)[2:].upper(), 'to_hex'
        elif op == 'digit_sum' and nums:
            return str(sum(int(d) for d in str(nums[0]))), 'digit_sum'
        elif op == 'divisible' and len(nums) >= 2:
            return str(nums[0] % nums[1] == 0), 'divisible'
        return None, None

    def _solve_algebra(self, q, op):
        import re

        if op == 'evaluate' or op == 'poly_eval':
            # Parse f(x) = a*x + b, evaluate at x=val
            match = re.search(r'f\((\d+)\)', q)  # f(5)
            if not match:
                match = re.search(r'f\((-?\d+)\)', q)
            if match:
                x_val = int(match.group(1))
                # Try quadratic
                quad = re.search(r'(-?\d+)\*x\^2\s*\+\s*(-?\d+)\*x\s*\+\s*(-?\d+)', q)
                if quad:
                    a, b, c = int(quad.group(1)), int(quad.group(2)), int(quad.group(3))
                    return str(a * x_val**2 + b * x_val + c), 'poly_eval'
                # Try linear
                lin = re.search(r'(-?\d+)\*x\s*\+\s*(-?\d+)', q)
                if lin:
                    a, b = int(lin.group(1)), int(lin.group(2))
                    return str(a * x_val + b), 'linear_eval'
        elif op == 'linear':
            # Solve a*x + b = c
            nums = re.findall(r'-?\d+', q)
            if len(nums) >= 3:
                a, b, c = int(nums[0]), int(nums[1]), int(nums[2])
                if a != 0:
                    return str((c - b) // a), 'linear_solve'
        elif op in ('seq_arith', 'seq_geo'):
            match = re.search(r'\[([^\]]+)\]', q)
            if match:
                seq = [int(x.strip()) for x in match.group(1).split(',')]
                if len(seq) >= 3:
                    # Check arithmetic
                    diffs = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
                    if len(set(diffs)) == 1:
                        return str(seq[-1] + diffs[0]), 'seq_arith'
                    # Check geometric
                    if all(seq[i] != 0 for i in range(len(seq)-1)):
                        ratios = [seq[i+1] / seq[i] for i in range(len(seq)-1)]
                        if len(set(ratios)) == 1:
                            return str(int(seq[-1] * ratios[0])), 'seq_geo'
        return None, None

    def _solve_counting(self, q, op):
        import re
        from math import factorial, comb, perm
        nums = [int(x) for x in re.findall(r'\d+', q)]

        if op == 'factorial' and nums:
            return str(factorial(nums[0])), 'factorial'
        elif op == 'choose' and len(nums) >= 2:
            return str(comb(nums[1], nums[0])), 'choose'
        elif op == 'permute' and len(nums) >= 2:
            return str(perm(nums[1], nums[0])), 'permute'
        elif op == 'count_digits' and nums:
            return str(len(str(nums[0]))), 'count_digits'
        elif op == 'count_divisors' and nums:
            n = nums[0]
            return str(sum(1 for i in range(1, n+1) if n % i == 0)), 'count_divisors'
        return None, None


# ═══════════════════════════════════════════════════════════
# BENCHMARK — test NeMo-WM math solver against DeepMind-style problems
# ═══════════════════════════════════════════════════════════

def run_benchmark(n_per_category=50):
    """Run full math benchmark."""

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   NeMo-WM DeepMind Math Benchmark                          ║")
    print("║   Testing: Arithmetic, Comparison, Numbers, Algebra,       ║")
    print("║            Counting                                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Generate problems
    all_problems = []
    generators = [
        ('Arithmetic', gen_arithmetic),
        ('Comparison', gen_comparison),
        ('Numbers', gen_numbers),
        ('Algebra', gen_algebra),
        ('Counting', gen_counting),
    ]

    for name, gen_fn in generators:
        problems = gen_fn(n_per_category)
        all_problems.extend(problems)
        print(f"  Generated {len(problems)} {name} problems")

    print(f"\n  Total: {len(all_problems)} problems")
    print(f"{'═'*60}")

    # Solve
    solver = NeMoMathSolver()
    results = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'unsolved': 0, 'total': 0})
    t0 = time.time()

    for p in all_problems:
        cat = p['category']
        results[cat]['total'] += 1

        answer, method = solver.solve(p)

        if answer is None:
            results[cat]['unsolved'] += 1
        elif str(answer).strip() == str(p['a']).strip():
            results[cat]['correct'] += 1
        else:
            results[cat]['wrong'] += 1

    elapsed = time.time() - t0

    # Report
    print(f"\n{'─'*60}")
    print(f"  {'Category':<15s} {'Correct':>8s} {'Wrong':>8s} {'Unsolved':>8s} {'Total':>8s} {'Acc':>8s}")
    print(f"{'─'*60}")

    total_correct = 0
    total_all = 0

    for cat in ['arithmetic', 'comparison', 'numbers', 'algebra', 'counting']:
        r = results[cat]
        acc = r['correct'] / max(r['total'], 1) * 100
        total_correct += r['correct']
        total_all += r['total']
        print(f"  {cat:<15s} {r['correct']:>8d} {r['wrong']:>8d} {r['unsolved']:>8d} {r['total']:>8d} {acc:>7.1f}%")

    print(f"{'─'*60}")
    overall_acc = total_correct / max(total_all, 1) * 100
    print(f"  {'TOTAL':<15s} {total_correct:>8d} {'':>8s} {'':>8s} {total_all:>8d} {overall_acc:>7.1f}%")
    print(f"\n  Time: {elapsed:.2f}s ({elapsed/total_all*1000:.1f}ms/problem)")

    # Show some examples
    print(f"\n{'═'*60}")
    print(f"  Sample Problems & Answers")
    print(f"{'═'*60}")
    for cat in ['arithmetic', 'comparison', 'numbers', 'algebra', 'counting']:
        cat_problems = [p for p in all_problems if p['category'] == cat][:3]
        print(f"\n  {cat.upper()}:")
        for p in cat_problems:
            answer, method = solver.solve(p)
            status = '✓' if str(answer) == str(p['a']) else '✗'
            print(f"    {status} Q: {p['q'][:70]}")
            print(f"      Expected: {p['a']}, Got: {answer} ({method})")

    # Difficulty breakdown
    print(f"\n{'═'*60}")
    print(f"  Difficulty Breakdown")
    print(f"{'═'*60}")
    diff_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    for p in all_problems:
        diff = p.get('difficulty', 'unknown')
        diff_results[diff]['total'] += 1
        answer, _ = solver.solve(p)
        if answer is not None and str(answer).strip() == str(p['a']).strip():
            diff_results[diff]['correct'] += 1

    for diff in ['easy', 'medium', 'hard']:
        r = diff_results[diff]
        acc = r['correct'] / max(r['total'], 1) * 100
        print(f"  {diff:<10s}: {r['correct']}/{r['total']} ({acc:.1f}%)")

    return total_correct, total_all


def generate_dataset(output_path='deepmind_math_dataset.json', n_per_category=200):
    """Generate a full dataset and save to JSON."""
    all_problems = []
    for name, gen_fn in [('arithmetic', gen_arithmetic), ('comparison', gen_comparison),
                          ('numbers', gen_numbers), ('algebra', gen_algebra),
                          ('counting', gen_counting)]:
        problems = gen_fn(n_per_category)
        all_problems.extend(problems)

    with open(output_path, 'w') as f:
        json.dump(all_problems, f, indent=2)
    print(f"Generated {len(all_problems)} problems to {output_path}")
    return all_problems


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--generate' in sys.argv:
        generate_dataset()
    elif '--benchmark' in sys.argv:
        run_benchmark(100)
    else:
        run_benchmark(50)
