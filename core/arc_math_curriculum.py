"""
NeMo-WM Math Curriculum
=========================
Progressive math learning system: lesson → quiz → test → master → next level.

10 Levels:
  L1: Counting & Recognition     (age 4-5)
  L2: Addition & Subtraction     (age 5-6)
  L3: Multiplication & Division  (age 6-7)
  L4: Fractions & Decimals       (age 7-8)
  L5: Order of Operations        (age 8-9)
  L6: Factors, Primes, GCD/LCM   (age 9-10)
  L7: Algebra Basics             (age 10-11)
  L8: Sequences & Patterns       (age 11-12)
  L9: Geometry & Measurement     (age 12-13)
  L10: Combinatorics & Probability (age 13-14)

Each level has:
  - LESSON: Teach the concept with examples
  - QUIZ: 5 quick problems (need 4/5 to pass)
  - TEST: 10 harder problems (need 8/10 to advance)
  - MASTERY: Track attempts, accuracy, speed

Usage:
    python arc_math_curriculum.py                    # Interactive mode
    python arc_math_curriculum.py --auto             # Auto-run all levels
    python arc_math_curriculum.py --level 5          # Start at level 5
    python arc_math_curriculum.py --report           # Show mastery report
"""
import json
import os
import sys
import time
import random
from collections import defaultdict
from math import gcd, factorial, comb, perm, sqrt, pi

random.seed(int(time.time()))


# ═══════════════════════════════════════════════════════════
# MASTERY TRACKER — persistent progress across sessions
# ═══════════════════════════════════════════════════════════

class MasteryTracker:
    """Track learning progress across levels and sessions."""

    def __init__(self, save_path='math_mastery.json'):
        self.save_path = save_path
        self.data = {
            'current_level': 1,
            'levels': {},
            'total_problems': 0,
            'total_correct': 0,
            'total_time': 0.0,
            'sessions': 0,
        }
        self.load()

    def load(self):
        if os.path.exists(self.save_path):
            try:
                with open(self.save_path) as f:
                    self.data = json.load(f)
            except:
                pass

    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def record(self, level, phase, correct, total, elapsed):
        key = f'L{level}'
        if key not in self.data['levels']:
            self.data['levels'][key] = {
                'quiz_attempts': 0, 'quiz_best': 0,
                'test_attempts': 0, 'test_best': 0,
                'mastered': False, 'total_problems': 0,
                'total_correct': 0, 'best_time': 999,
            }
        lv = self.data['levels'][key]
        lv['total_problems'] += total
        lv['total_correct'] += correct
        if phase == 'quiz':
            lv['quiz_attempts'] += 1
            lv['quiz_best'] = max(lv['quiz_best'], correct)
        elif phase == 'test':
            lv['test_attempts'] += 1
            lv['test_best'] = max(lv['test_best'], correct)
            if correct >= 8:
                lv['mastered'] = True
                if level >= self.data['current_level']:
                    self.data['current_level'] = level + 1
        lv['best_time'] = min(lv['best_time'], elapsed)
        self.data['total_problems'] += total
        self.data['total_correct'] += correct
        self.data['total_time'] += elapsed
        self.save()

    def is_mastered(self, level):
        key = f'L{level}'
        return self.data['levels'].get(key, {}).get('mastered', False)

    def report(self):
        print(f"\n{'═'*60}")
        print(f"  MASTERY REPORT")
        print(f"{'═'*60}")
        print(f"  Current Level: {self.data['current_level']}")
        print(f"  Total Problems: {self.data['total_problems']}")
        total = self.data['total_problems']
        correct = self.data['total_correct']
        acc = correct / max(total, 1) * 100
        print(f"  Overall Accuracy: {correct}/{total} ({acc:.1f}%)")
        print(f"  Total Time: {self.data['total_time']:.1f}s")
        print()
        for i in range(1, 11):
            key = f'L{i}'
            lv = self.data['levels'].get(key, {})
            mastered = lv.get('mastered', False)
            icon = '★' if mastered else '○'
            quiz_best = lv.get('quiz_best', 0)
            test_best = lv.get('test_best', 0)
            attempts = lv.get('quiz_attempts', 0) + lv.get('test_attempts', 0)
            name = LEVEL_NAMES.get(i, f'Level {i}')
            print(f"  {icon} L{i:2d}: {name:30s} quiz={quiz_best}/5  test={test_best}/10  attempts={attempts}")
        print(f"{'═'*60}")


LEVEL_NAMES = {
    1: 'Counting & Recognition',
    2: 'Addition & Subtraction',
    3: 'Multiplication & Division',
    4: 'Fractions & Decimals',
    5: 'Order of Operations',
    6: 'Factors, Primes, GCD/LCM',
    7: 'Algebra Basics',
    8: 'Sequences & Patterns',
    9: 'Geometry & Measurement',
    10: 'Combinatorics & Probability',
}


# ═══════════════════════════════════════════════════════════
# LESSONS — teach concepts with examples
# ═══════════════════════════════════════════════════════════

def lesson(level):
    """Print a lesson for the given level."""
    print(f"\n{'═'*60}")
    print(f"  LESSON {level}: {LEVEL_NAMES[level]}")
    print(f"{'═'*60}")

    if level == 1:
        print("""
  COUNTING & RECOGNITION

  Numbers represent quantities. We can count objects:
    • 1 apple, 2 apples, 3 apples...
    • Numbers go: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10...

  Key skills:
    • Count objects in a group
    • Compare: which is more? which is less?
    • Recognize odd vs even numbers
      Even: 2, 4, 6, 8, 10...  (divisible by 2)
      Odd:  1, 3, 5, 7, 9...   (not divisible by 2)

  Examples:
    Q: How many items in [apple, banana, cherry]?  A: 3
    Q: Which is larger, 7 or 4?                    A: 7
    Q: Is 6 even or odd?                           A: even
""")
    elif level == 2:
        print("""
  ADDITION & SUBTRACTION

  Addition (+): combining two groups
    3 + 5 = 8     (3 apples plus 5 apples = 8 apples)

  Subtraction (-): taking away from a group
    9 - 4 = 5     (9 apples minus 4 eaten = 5 left)

  Key properties:
    • a + b = b + a          (commutative)
    • a + 0 = a              (identity)
    • a - a = 0              (inverse)
    • Negative results: 3 - 7 = -4

  Examples:
    Q: 15 + 27 = ?     A: 42
    Q: 100 - 37 = ?    A: 63
    Q: -5 + 12 = ?     A: 7
""")
    elif level == 3:
        print("""
  MULTIPLICATION & DIVISION

  Multiplication (*): repeated addition
    4 * 3 = 12     (4 groups of 3, or 3+3+3+3)

  Division (/): splitting into equal groups
    12 / 4 = 3     (12 items into 4 groups = 3 each)
    Remainder: 13 / 4 = 3 remainder 1

  Key properties:
    • a * b = b * a          (commutative)
    • a * 1 = a              (identity)
    • a * 0 = 0              (zero property)
    • a / a = 1   (for a ≠ 0)

  Examples:
    Q: 7 * 8 = ?           A: 56
    Q: 144 / 12 = ?        A: 12
    Q: 17 mod 5 = ?        A: 2
""")
    elif level == 4:
        print("""
  FRACTIONS & DECIMALS

  Fractions: parts of a whole
    1/2 = one half = 0.5
    3/4 = three quarters = 0.75

  Operations:
    • a/b + c/b = (a+c)/b         (same denominator)
    • a/b * c/d = (a*c)/(b*d)     (multiply across)
    • a/b ÷ c/d = a/b * d/c       (flip and multiply)

  Decimals: base-10 fractions
    0.1 = 1/10,  0.25 = 1/4,  0.333... = 1/3

  Examples:
    Q: 1/2 + 1/4 = ?      A: 3/4  (= 0.75)
    Q: 2/3 * 3/5 = ?      A: 6/15 = 2/5
    Q: 0.5 + 0.25 = ?     A: 0.75
""")
    elif level == 5:
        print("""
  ORDER OF OPERATIONS (PEMDAS/BODMAS)

  When an expression has multiple operations, follow this order:
    1. Parentheses / Brackets
    2. Exponents / Powers
    3. Multiplication and Division (left to right)
    4. Addition and Subtraction (left to right)

  Examples:
    Q: 2 + 3 * 4 = ?          A: 14  (not 20!)
    Q: (2 + 3) * 4 = ?        A: 20
    Q: 2 ** 3 + 1 = ?         A: 9
    Q: 10 - 2 * 3 + 1 = ?     A: 5
""")
    elif level == 6:
        print("""
  FACTORS, PRIMES, GCD, LCM

  Factor: a divides b evenly → a is a factor of b
    Factors of 12: 1, 2, 3, 4, 6, 12

  Prime: a number with exactly 2 factors (1 and itself)
    Primes: 2, 3, 5, 7, 11, 13, 17, 19, 23...

  GCD (Greatest Common Divisor): largest shared factor
    GCD(12, 18) = 6

  LCM (Least Common Multiple): smallest shared multiple
    LCM(4, 6) = 12

  Relationship: GCD(a,b) * LCM(a,b) = a * b

  Examples:
    Q: Is 17 prime?            A: Yes
    Q: GCD(24, 36) = ?         A: 12
    Q: LCM(8, 12) = ?         A: 24
""")
    elif level == 7:
        print("""
  ALGEBRA BASICS

  Variables represent unknown numbers:
    x + 5 = 12  →  x = 7

  Solving linear equations:
    ax + b = c  →  x = (c - b) / a

  Evaluating expressions:
    f(x) = 3x + 2
    f(4) = 3*4 + 2 = 14

  Examples:
    Q: Solve 3x + 7 = 22 for x.     A: x = 5
    Q: If f(x) = 2x - 1, find f(6). A: 11
    Q: Simplify 4x + 3x.            A: 7x
""")
    elif level == 8:
        print("""
  SEQUENCES & PATTERNS

  Arithmetic sequence: constant difference
    2, 5, 8, 11, 14, ...  (difference = 3)
    nth term = first + (n-1) * difference

  Geometric sequence: constant ratio
    3, 6, 12, 24, ...  (ratio = 2)
    nth term = first * ratio^(n-1)

  Fibonacci-like: each term = sum of previous two
    1, 1, 2, 3, 5, 8, 13, ...

  Examples:
    Q: Next in 4, 7, 10, 13, ?      A: 16 (diff=3)
    Q: Next in 2, 6, 18, 54, ?      A: 162 (ratio=3)
    Q: Sum of 1+2+3+...+10 = ?      A: 55
""")
    elif level == 9:
        print("""
  GEOMETRY & MEASUREMENT

  Perimeter: distance around a shape
    Rectangle: P = 2*(length + width)
    Circle: C = 2*pi*r

  Area: space inside a shape
    Rectangle: A = length * width
    Triangle: A = base * height / 2
    Circle: A = pi * r^2

  Volume:
    Box: V = length * width * height
    Sphere: V = (4/3) * pi * r^3

  Examples:
    Q: Area of rectangle 5x8 = ?     A: 40
    Q: Perimeter of square side 6 = ? A: 24
    Q: Area of triangle base=10 height=4 = ?  A: 20
""")
    elif level == 10:
        print("""
  COMBINATORICS & PROBABILITY

  Factorial: n! = n * (n-1) * ... * 1
    5! = 120

  Combinations (choose): C(n,k) = n! / (k! * (n-k)!)
    C(5,2) = 10  (ways to choose 2 from 5)

  Permutations (arrange): P(n,k) = n! / (n-k)!
    P(5,2) = 20  (ways to arrange 2 from 5)

  Probability: P(event) = favorable / total
    P(heads on coin) = 1/2

  Examples:
    Q: 6! = ?                        A: 720
    Q: C(10, 3) = ?                  A: 120
    Q: P(rolling 6 on a die) = ?     A: 1/6
""")


# ═══════════════════════════════════════════════════════════
# PROBLEM GENERATORS PER LEVEL
# ═══════════════════════════════════════════════════════════

def gen_level(level, n, hard=False):
    """Generate n problems for a given level."""
    problems = []
    for _ in range(n):
        p = _gen_one(level, hard)
        if p:
            problems.append(p)
    return problems


def _gen_one(level, hard=False):
    """Generate a single problem for a level."""

    if level == 1:
        cat = random.choice(['count', 'compare', 'even_odd'])
        if cat == 'count':
            n = random.randint(2, 15 if hard else 9)
            items = random.sample(['apple','ball','cat','dog','egg','fish','grape','hat'], min(n, 8))
            return {'q': f'How many items: {items}?', 'a': str(len(items)), 'method': 'len'}
        elif cat == 'compare':
            a, b = random.randint(1, 50 if hard else 20), random.randint(1, 50 if hard else 20)
            while a == b: b = random.randint(1, 50)
            return {'q': f'Which is larger, {a} or {b}?', 'a': str(max(a, b)), 'method': 'max'}
        else:
            n = random.randint(1, 100 if hard else 20)
            return {'q': f'Is {n} even or odd?', 'a': 'even' if n%2==0 else 'odd', 'method': 'mod2'}

    elif level == 2:
        if hard:
            a, b = random.randint(-100, 100), random.randint(-100, 100)
        else:
            a, b = random.randint(1, 50), random.randint(1, 50)
        op = random.choice(['+', '-'])
        result = a + b if op == '+' else a - b
        return {'q': f'Calculate {a} {op} {b}.', 'a': str(result), 'method': 'arith'}

    elif level == 3:
        if random.random() < 0.5:
            a = random.randint(2, 15 if hard else 12)
            b = random.randint(2, 15 if hard else 12)
            return {'q': f'Calculate {a} * {b}.', 'a': str(a*b), 'method': 'mul'}
        else:
            b = random.randint(2, 15 if hard else 12)
            a = b * random.randint(1, 15 if hard else 10)
            if random.random() < 0.3:
                return {'q': f'What is {a} mod {b}?', 'a': str(a%b), 'method': 'mod'}
            return {'q': f'Calculate {a} / {b}.', 'a': str(a//b), 'method': 'div'}

    elif level == 4:
        cat = random.choice(['decimal_add', 'fraction_mul', 'pct'])
        if cat == 'decimal_add':
            a = round(random.uniform(0.1, 10.0), 1 if not hard else 2)
            b = round(random.uniform(0.1, 10.0), 1 if not hard else 2)
            return {'q': f'Calculate {a} + {b}.', 'a': str(round(a+b, 2)), 'method': 'decimal_add'}
        elif cat == 'fraction_mul':
            n1, d1 = random.randint(1, 5), random.randint(2, 8)
            n2, d2 = random.randint(1, 5), random.randint(2, 8)
            rn, rd = n1*n2, d1*d2
            g = gcd(rn, rd)
            return {'q': f'Calculate {n1}/{d1} * {n2}/{d2}. Give as simplified fraction.', 
                    'a': f'{rn//g}/{rd//g}', 'method': 'frac_mul'}
        else:
            n = random.randint(10, 200)
            pct = random.choice([10, 20, 25, 50])
            return {'q': f'What is {pct}% of {n}?', 'a': str(n * pct // 100), 'method': 'pct'}

    elif level == 5:
        cat = random.choice(['pemdas', 'power', 'mixed'])
        if cat == 'pemdas':
            a, b, c = random.randint(1, 10), random.randint(1, 10), random.randint(1, 10)
            return {'q': f'Calculate {a} + {b} * {c}.', 'a': str(a + b*c), 'method': 'pemdas'}
        elif cat == 'power':
            base = random.randint(2, 5 if not hard else 10)
            exp = random.randint(2, 4 if not hard else 5)
            return {'q': f'Calculate {base} ** {exp}.', 'a': str(base**exp), 'method': 'power'}
        else:
            a, b, c = random.randint(1, 20), random.randint(1, 10), random.randint(1, 10)
            return {'q': f'Calculate ({a} + {b}) * {c}.', 'a': str((a+b)*c), 'method': 'parens'}

    elif level == 6:
        cat = random.choice(['prime', 'gcd', 'lcm', 'factors', 'digit_sum'])
        if cat == 'prime':
            n = random.randint(2, 200 if hard else 50)
            is_p = n >= 2 and all(n % i != 0 for i in range(2, int(sqrt(n))+1))
            return {'q': f'Is {n} prime?', 'a': str(is_p), 'method': 'prime'}
        elif cat == 'gcd':
            a, b = random.randint(4, 200 if hard else 60), random.randint(4, 200 if hard else 60)
            return {'q': f'GCD of {a} and {b}?', 'a': str(gcd(a, b)), 'method': 'gcd'}
        elif cat == 'lcm':
            a, b = random.randint(2, 30 if hard else 15), random.randint(2, 30 if hard else 15)
            return {'q': f'LCM of {a} and {b}?', 'a': str(a*b//gcd(a,b)), 'method': 'lcm'}
        elif cat == 'factors':
            n = random.randint(2, 100)
            count = sum(1 for i in range(1, n+1) if n%i==0)
            return {'q': f'How many factors does {n} have?', 'a': str(count), 'method': 'factors'}
        else:
            n = random.randint(100, 99999)
            return {'q': f'Sum of digits of {n}?', 'a': str(sum(int(d) for d in str(n))), 'method': 'digit_sum'}

    elif level == 7:
        cat = random.choice(['eval_linear', 'solve_linear', 'eval_quad'])
        if cat == 'eval_linear':
            a, b = random.randint(-10, 10), random.randint(-10, 10)
            x = random.randint(-10, 10)
            return {'q': f'f(x) = {a}*x + {b}. f({x}) = ?', 'a': str(a*x+b), 'method': 'eval'}
        elif cat == 'solve_linear':
            x = random.randint(-20, 20)
            a = random.randint(1, 10) * random.choice([-1,1])
            b = random.randint(-50, 50)
            c = a*x + b
            return {'q': f'Solve {a}*x + {b} = {c} for x.', 'a': str(x), 'method': 'solve'}
        else:
            a, b, c = random.randint(-5, 5), random.randint(-5, 5), random.randint(-5, 5)
            x = random.randint(-5, 5)
            return {'q': f'f(x) = {a}*x^2 + {b}*x + {c}. f({x}) = ?',
                    'a': str(a*x**2 + b*x + c), 'method': 'eval_quad'}

    elif level == 8:
        cat = random.choice(['arith_seq', 'geo_seq', 'sum_n', 'triangular'])
        if cat == 'arith_seq':
            start = random.randint(-10, 10)
            step = random.randint(1, 8) * random.choice([-1, 1])
            length = random.randint(4, 7)
            seq = [start + i*step for i in range(length)]
            return {'q': f'Next in {seq}?', 'a': str(seq[-1]+step), 'method': 'arith_seq'}
        elif cat == 'geo_seq':
            start = random.randint(1, 4)
            ratio = random.choice([2, 3])
            length = random.randint(3, 5)
            seq = [start * ratio**i for i in range(length)]
            return {'q': f'Next in {seq}?', 'a': str(seq[-1]*ratio), 'method': 'geo_seq'}
        elif cat == 'sum_n':
            n = random.randint(5, 20 if hard else 10)
            return {'q': f'Sum of 1+2+3+...+{n}?', 'a': str(n*(n+1)//2), 'method': 'gauss'}
        else:
            n = random.randint(3, 10)
            return {'q': f'What is the {n}th triangular number?', 'a': str(n*(n+1)//2), 'method': 'triangular'}

    elif level == 9:
        cat = random.choice(['rect_area', 'rect_perim', 'tri_area', 'circle', 'volume'])
        if cat == 'rect_area':
            l, w = random.randint(2, 20), random.randint(2, 20)
            return {'q': f'Area of rectangle {l} x {w}?', 'a': str(l*w), 'method': 'rect_area'}
        elif cat == 'rect_perim':
            l, w = random.randint(2, 20), random.randint(2, 20)
            return {'q': f'Perimeter of rectangle {l} x {w}?', 'a': str(2*(l+w)), 'method': 'rect_perim'}
        elif cat == 'tri_area':
            b, h = random.randint(2, 20), random.randint(2, 20)
            area = b * h / 2
            return {'q': f'Area of triangle base={b} height={h}?',
                    'a': str(int(area) if area==int(area) else area), 'method': 'tri_area'}
        elif cat == 'circle':
            r = random.randint(1, 10)
            return {'q': f'Area of circle radius {r}? (round to integer)',
                    'a': str(round(pi * r**2)), 'method': 'circle_area'}
        else:
            l, w, h = random.randint(2, 10), random.randint(2, 10), random.randint(2, 10)
            return {'q': f'Volume of box {l} x {w} x {h}?', 'a': str(l*w*h), 'method': 'volume'}

    elif level == 10:
        cat = random.choice(['factorial', 'choose', 'permute', 'probability'])
        if cat == 'factorial':
            n = random.randint(1, 8 if not hard else 10)
            return {'q': f'{n}! = ?', 'a': str(factorial(n)), 'method': 'factorial'}
        elif cat == 'choose':
            n = random.randint(3, 10 if not hard else 15)
            k = random.randint(1, n)
            return {'q': f'C({n},{k}) = ?', 'a': str(comb(n, k)), 'method': 'choose'}
        elif cat == 'permute':
            n = random.randint(3, 8)
            k = random.randint(1, min(n, 4))
            return {'q': f'P({n},{k}) = ?', 'a': str(perm(n, k)), 'method': 'permute'}
        else:
            n = random.choice([6, 8, 10, 12, 20])
            k = random.randint(1, n//2)
            return {'q': f'Probability of picking one of {k} items from {n}? (as fraction)',
                    'a': f'{k}/{n}' if gcd(k,n)==1 else f'{k//gcd(k,n)}/{n//gcd(k,n)}',
                    'method': 'probability'}

    return None


# ═══════════════════════════════════════════════════════════
# SOLVER — NeMo-WM's math engine
# ═══════════════════════════════════════════════════════════

def solve_problem(problem):
    """Solve a curriculum problem. Returns (answer, method)."""
    q = problem['q']
    method = problem.get('method', 'unknown')
    import re

    try:
        if method in ('len',):
            match = re.search(r'\[([^\]]+)\]', q)
            if match:
                items = [x.strip().strip("'\"") for x in match.group(1).split(',')]
                return str(len(items)), method

        elif method in ('max', 'compare'):
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(max(nums)), method

        elif method == 'mod2':
            n = int(re.search(r'(\d+)', q).group(1))
            return 'even' if n % 2 == 0 else 'odd', method

        elif method in ('arith', 'mul', 'div', 'mod', 'pemdas', 'power', 'parens'):
            expr = q.replace('Calculate', '').strip()
            if expr.endswith('.'): expr = expr[:-1]
            expr = expr.replace('mod', '%')
            if '?' in expr: expr = expr.replace('?', '').replace('=', '').strip()
            result = eval(expr, {"__builtins__": {}})
            if isinstance(result, float):
                result = round(result, 2)
                if result == int(result): result = int(result)
            return str(result), method

        elif method == 'decimal_add':
            # Extract two decimal numbers
            nums = re.findall(r'[\d]+\.[\d]+', q)
            if len(nums) >= 2:
                result = float(nums[0]) + float(nums[1])
                result = round(result, 2)
                if result == int(result): result = int(result)
                return str(result), method

        elif method == 'frac_mul':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            if len(nums) >= 4:
                n1, d1, n2, d2 = nums[0], nums[1], nums[2], nums[3]
                rn, rd = n1*n2, d1*d2
                g = gcd(rn, rd)
                return f'{rn//g}/{rd//g}', method

        elif method == 'pct':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(nums[0] * nums[1] // 100), method

        elif method == 'prime':
            n = int(re.search(r'(\d+)', q).group(1))
            is_p = n >= 2 and all(n % i != 0 for i in range(2, int(sqrt(n))+1))
            return str(is_p), method

        elif method == 'gcd':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(gcd(nums[0], nums[1])), method

        elif method == 'lcm':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(nums[0]*nums[1]//gcd(nums[0],nums[1])), method

        elif method == 'factors':
            n = int(re.search(r'(\d+)', q).group(1))
            return str(sum(1 for i in range(1, n+1) if n%i==0)), method

        elif method == 'digit_sum':
            n = re.search(r'(\d+)', q).group(1)
            return str(sum(int(d) for d in n)), method

        elif method in ('eval', 'eval_quad'):
            # Parse f(x) = a*x + b or a*x^2 + b*x + c
            x_match = re.search(r'f\((-?\d+)\)', q)
            if x_match:
                x = int(x_match.group(1))
                quad = re.search(r'(-?\d+)\*x\^2\s*\+\s*(-?\d+)\*x\s*\+\s*(-?\d+)', q)
                if quad:
                    a, b, c = int(quad.group(1)), int(quad.group(2)), int(quad.group(3))
                    return str(a*x**2 + b*x + c), method
                lin = re.search(r'(-?\d+)\*x\s*\+\s*(-?\d+)', q)
                if lin:
                    a, b = int(lin.group(1)), int(lin.group(2))
                    return str(a*x + b), method

        elif method == 'solve':
            nums = [int(x) for x in re.findall(r'-?\d+', q)]
            if len(nums) >= 3:
                a, b, c = nums[0], nums[1], nums[2]
                if a != 0: return str((c-b)//a), method

        elif method in ('arith_seq', 'geo_seq'):
            match = re.search(r'\[([^\]]+)\]', q)
            if match:
                seq = [int(x.strip()) for x in match.group(1).split(',')]
                diffs = [seq[i+1]-seq[i] for i in range(len(seq)-1)]
                if len(set(diffs)) == 1:
                    return str(seq[-1]+diffs[0]), method
                if all(seq[i]!=0 for i in range(len(seq)-1)):
                    ratios = [seq[i+1]/seq[i] for i in range(len(seq)-1)]
                    if len(set(ratios)) == 1:
                        return str(int(seq[-1]*ratios[0])), method

        elif method == 'gauss':
            n = int(re.search(r'\+(\d+)\?', q).group(1))
            return str(n*(n+1)//2), method

        elif method == 'triangular':
            n = int(re.search(r'(\d+)(?:th|st|nd|rd)', q).group(1))
            return str(n*(n+1)//2), method

        elif method == 'rect_area':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(nums[0]*nums[1]), method

        elif method == 'rect_perim':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(2*(nums[0]+nums[1])), method

        elif method == 'tri_area':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            area = nums[0]*nums[1]/2
            return str(int(area) if area==int(area) else area), method

        elif method == 'circle_area':
            r = int(re.search(r'radius (\d+)', q).group(1))
            return str(round(pi * r**2)), method

        elif method == 'volume':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(nums[0]*nums[1]*nums[2]), method

        elif method == 'factorial':
            n = int(re.search(r'(\d+)!', q).group(1))
            return str(factorial(n)), method

        elif method == 'choose':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(comb(nums[0], nums[1])), method

        elif method == 'permute':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            return str(perm(nums[0], nums[1])), method

        elif method == 'probability':
            nums = [int(x) for x in re.findall(r'\d+', q)]
            k, n = nums[0], nums[1]
            g = gcd(k, n)
            return f'{k//g}/{n//g}', method

    except:
        pass

    return None, None


# ═══════════════════════════════════════════════════════════
# RUN PHASE — quiz or test
# ═══════════════════════════════════════════════════════════

def run_phase(level, phase='quiz', tracker=None):
    """Run a quiz (5 problems) or test (10 problems)."""
    n = 5 if phase == 'quiz' else 10
    hard = phase == 'test'
    threshold = 4 if phase == 'quiz' else 8

    problems = gen_level(level, n, hard=hard)

    print(f"\n  {'─'*50}")
    print(f"  {phase.upper()} — Level {level}: {LEVEL_NAMES[level]}")
    print(f"  {n} problems, need {threshold}/{n} to {'pass' if phase=='quiz' else 'master'}")
    print(f"  {'─'*50}")

    correct = 0
    t0 = time.time()

    for i, p in enumerate(problems):
        answer, method = solve_problem(p)
        expected = str(p['a']).strip()
        got = str(answer).strip() if answer else 'None'
        ok = got == expected

        if ok:
            correct += 1
            icon = '✓'
        else:
            icon = '✗'

        print(f"    {icon} [{i+1}/{n}] {p['q'][:65]}")
        if not ok:
            print(f"         Expected: {expected}, Got: {got}")

    elapsed = time.time() - t0
    passed = correct >= threshold

    print(f"\n  Result: {correct}/{n} {'PASS ✓' if passed else 'FAIL ✗'} ({elapsed:.2f}s)")

    if tracker:
        tracker.record(level, phase, correct, n, elapsed)

    return passed, correct, n


# ═══════════════════════════════════════════════════════════
# MAIN CURRICULUM LOOP
# ═══════════════════════════════════════════════════════════

def run_curriculum(start_level=1, auto=True):
    """Run the full curriculum: lesson → quiz → test → next level."""
    tracker = MasteryTracker()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║        NeMo-WM Math Curriculum                              ║")
    print("║        10 Levels: Counting → Combinatorics                  ║")
    print("║        Lesson → Quiz → Test → Master → Next Level           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    level = max(start_level, tracker.data.get('current_level', 1))
    if start_level > 1:
        level = start_level

    while level <= 10:
        print(f"\n{'═'*60}")
        print(f"  LEVEL {level}: {LEVEL_NAMES[level]}")
        print(f"  {'Already mastered ★' if tracker.is_mastered(level) else 'Not yet mastered'}")
        print(f"{'═'*60}")

        # Lesson
        lesson(level)

        # Quiz
        print(f"\n  Starting quiz...")
        quiz_passed, _, _ = run_phase(level, 'quiz', tracker)

        if not quiz_passed:
            print(f"\n  Quiz not passed. Reviewing lesson...")
            if not auto:
                input("  Press Enter to retry...")
            # Retry quiz once
            quiz_passed, _, _ = run_phase(level, 'quiz', tracker)
            if not quiz_passed:
                print(f"  Quiz failed twice. Moving to next level anyway (for learning).")

        # Test
        print(f"\n  Starting test...")
        test_passed, correct, total = run_phase(level, 'test', tracker)

        if test_passed:
            print(f"\n  ★ LEVEL {level} MASTERED! ★")
        else:
            print(f"\n  Test not passed ({correct}/{total}). Retrying...")
            test_passed, correct, total = run_phase(level, 'test', tracker)
            if test_passed:
                print(f"\n  ★ LEVEL {level} MASTERED on retry! ★")
            else:
                print(f"\n  Level {level} not mastered yet ({correct}/{total}). Continuing to next level.")

        level += 1

    # Final report
    tracker.data['sessions'] += 1
    tracker.save()
    tracker.report()

    # Summary
    mastered = sum(1 for i in range(1, 11) if tracker.is_mastered(i))
    print(f"\n  ★ Mastered {mastered}/10 levels")
    print(f"  Overall accuracy: {tracker.data['total_correct']}/{tracker.data['total_problems']} "
          f"({tracker.data['total_correct']/max(tracker.data['total_problems'],1)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    if '--report' in sys.argv:
        MasteryTracker().report()
    elif '--auto' in sys.argv:
        start = 1
        if '--level' in sys.argv:
            idx = sys.argv.index('--level')
            start = int(sys.argv[idx+1])
        run_curriculum(start_level=start, auto=True)
    elif '--level' in sys.argv:
        idx = sys.argv.index('--level')
        start = int(sys.argv[idx+1])
        run_curriculum(start_level=start, auto=True)
    else:
        run_curriculum(auto=True)
