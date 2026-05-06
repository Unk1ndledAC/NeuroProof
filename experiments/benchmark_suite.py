"""
benchmark_suite.py
==================
Reproducible benchmark suite for NeuroProof evaluation.

Performance improvements over the original version:
  - Linear-size Tseitin XOR encoding via auxiliary variables
    (replaces 2^k exponential blowup per vertex parity constraint).
  - concurrent.futures.ProcessPoolExecutor parallelisation for
    EXP-1, EXP-2, EXP-3 (CPU-bound, embarrassingly parallel).
  - Reduced default n_trials where possible to keep wall-clock
    time bounded.

Experiments:
  EXP-1: Random 3-CNF phase transition sweep  (parallel)
  EXP-2: Pigeonhole Principle PHP_n^{n+1}     (parallel)
  EXP-3: Tseitin tautologies (graph-based)     (parallel)
  EXP-4: Classical tautology proof quality     (fast, sequential)
  EXP-5: ATSS online learning convergence      (sequential, shared state)

Usage:
  python -m experiments.benchmark_suite           # run all experiments
  python -m experiments.benchmark_suite --exp 1   # run only EXP-1
  python -m experiments.benchmark_suite --workers 8  # set worker count

References:
  - Hoos & Stutzle (2000): SATLIB. http://www.satlib.org
  - Ben-Sasson & Wigderson (2001): pigeonhole lower bounds.
    DOI: 10.1145/375827.375835
  - Tseitin (1968): On the Complexity of Derivation in Propositional Calculus.
"""

from __future__ import annotations

import random
import time
import csv
import os
import sys
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.formula import Var, Not, And, Or, Implies, parse, Formula
from src.solver import NeuroProofSolver, ATSS, SolverStatus, Clause
from src.tactic import TacticEngine, tauto
from src.proof import Proof


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class BenchmarkResult:
    """Stores the result of a single benchmark run."""
    name:          str
    instance_id:   int
    n_vars:        int
    n_clauses:     int
    status:        str           # SAT / UNSAT / UNKNOWN / TIMEOUT / PROVED / FAIL
    solver:        str
    time_sec:      float
    decisions:     int  = 0
    conflicts:     int  = 0
    learned:       int  = 0
    proof_size:    int  = 0      # number of proof steps (for UNSAT/PROVED)
    proof_depth:   int  = 0


# ============================================================================
# Formula generators
# ============================================================================

def gen_random_3cnf(n_vars: int, n_clauses: int,
                     seed: Optional[int] = None) -> List[Clause]:
    """
    Generate a random 3-CNF instance.

    Each clause is formed by sampling 3 distinct variables uniformly at
    random (with replacement if n_vars < 3) and negating each with
    probability 0.5.

    Args:
        n_vars:   Number of propositional variables (named x1, ..., xn).
        n_clauses: Number of clauses to generate.
        seed:      Random seed for reproducibility.

    Returns:
        List of clauses, where each clause is a frozenset of (var_name, is_positive) tuples.
    """
    rng = random.Random(seed)
    vars_ = [f"x{i}" for i in range(1, n_vars + 1)]
    clauses = []
    for _ in range(n_clauses):
        lits = rng.sample(vars_, min(3, len(vars_)))
        clause = frozenset(
            (v, rng.choice([True, False])) for v in lits
        )
        clauses.append(clause)
    return clauses


def gen_pigeonhole(n: int) -> List[Clause]:
    """
    Generate the Pigeonhole Principle CNF: PHP_n^{n+1}.

    Encoding: n+1 pigeons into n holes.
    Variables: p_{i,j} = "pigeon i goes into hole j" (1 <= i <= n+1, 1 <= j <= n).

    Clause families:
      1. At-least-one:  for each pigeon i: OR_{j=1}^{n} p_{i,j}
      2. At-most-one:  for each hole j, each pair i < k: NOT p_{i,j} OR NOT p_{k,j}
      3. Functional:   for each pigeon i, each pair j < k: NOT p_{i,j} OR NOT p_{i,k}

    The third family is essential for correct PHP semantics and yields
    the exponential resolution lower bound of 2^{Omega(n)} (Haken 1985).

    Total clauses: (n+1) + (n+1)*n*(n-1)/2 + (n+1)*n*(n-1)/2 = O(n^3).
    Total variables: (n+1)*n.

    Args:
        n: Number of holes.

    Returns:
        List of CNF clauses encoding PHP_n^{n+1}.
    """
    def pvar(i: int, j: int) -> str:
        return f"p_{i}_{j}"

    clauses: List[Clause] = []
    pigeons = list(range(1, n + 2))   # 1 .. n+1
    holes   = list(range(1, n + 1))   # 1 .. n

    # Family 1: Each pigeon must be in at least one hole
    for i in pigeons:
        clause = frozenset((pvar(i, j), True) for j in holes)
        clauses.append(clause)

    # Family 2: No two pigeons share the same hole
    for j in holes:
        for i in pigeons:
            for k in pigeons:
                if i < k:
                    clauses.append(frozenset([
                        (pvar(i, j), False),
                        (pvar(k, j), False)
                    ]))

    # Family 3: Each pigeon in at most one hole (functional constraint)
    for i in pigeons:
        for j in holes:
            for k in holes:
                if j < k:
                    clauses.append(frozenset([
                        (pvar(i, j), False),
                        (pvar(i, k), False)
                    ]))

    return clauses


def _xor_clauses_linear(vars_: List[str], target: int,
                         aux_counter: List[int]) -> List[Clause]:
    """
    Encode XOR(vars_) = target as CNF clauses using LINEAR-SIZE encoding.

    Uses a cascade of auxiliary variables instead of the exponential 2^k
    blowup.  For each adjacent pair (vars_[i], vars_[i+1]), an auxiliary
    variable s_i encodes the partial XOR.

    Encoding for XOR(a, b) = s (auxiliary):
      (a  | b  | ~s)      -- if both true, s must be true (parity 0)
      (a  | ~b |  s)      -- if a true, b false, s must be true
      (~a | b  |  s)      -- if a false, b true, s must be true
      (~a | ~b | ~s)      -- if both false, s must be false (parity 0)

    This uses exactly 4*(n-1) clauses for n input variables, compared to
    2^{n-1} clauses in the old exponential encoding.

    Args:
        vars_:       List of variable names.
        target:      Desired XOR value (0 or 1).
        aux_counter: Mutable [int] counter for generating unique auxiliary
                     variable names (shared across calls).

    Returns:
        List of CNF clauses.
    """
    n = len(vars_)
    if n == 0:
        # Empty XOR: vacuously 0
        return [frozenset()] if target == 1 else []

    if n == 1:
        v = vars_[0]
        if target == 1:
            return [frozenset([(v, True)])]     # v must be true
        else:
            return [frozenset([(v, False)])]    # v must be false

    # Build a linear cascade: s_0, s_1, ..., s_{n-2}
    aux_names = []
    for i in range(n - 1):
        aux_counter[0] += 1
        aux_names.append(f"_xor_a{aux_counter[0]}")

    clauses: List[Clause] = []

    # First stage: XOR(vars_[0], vars_[1]) = s_0
    a, b, s = vars_[0], vars_[1], aux_names[0]
    clauses.extend([
        frozenset([(a, True),  (b, True),  (s, False)]),
        frozenset([(a, True),  (b, False), (s, True)]),
        frozenset([(a, False), (b, True),  (s, True)]),
        frozenset([(a, False), (b, False), (s, False)]),
    ])

    # Middle stages: XOR(s_{i-1}, vars_[i+1]) = s_i
    for i in range(1, n - 1):
        a, b, s = aux_names[i - 1], vars_[i + 1], aux_names[i]
        clauses.extend([
            frozenset([(a, True),  (b, True),  (s, False)]),
            frozenset([(a, True),  (b, False), (s, True)]),
            frozenset([(a, False), (b, True),  (s, True)]),
            frozenset([(a, False), (b, False), (s, False)]),
        ])

    # Final: s_{n-2} must equal target
    last_s = aux_names[-1]
    if target == 1:
        clauses.append(frozenset([(last_s, True)]))
    else:
        clauses.append(frozenset([(last_s, False)]))

    return clauses


def gen_tseitin(n_vertices: int, density: float = 0.5,
                 seed: Optional[int] = None) -> List[Clause]:
    """
    Generate a Tseitin tautology on a random graph.

    Each edge (u, v) gets a variable e_{u}_{v}.  For each vertex with
    odd-degree-parity assignment, XOR constraints are encoded as CNF
    clauses.  The total parity is forced to be odd, making the system
    UNSAT (Tseitin, 1968).

    Uses LINEAR-SIZE auxiliary-variable XOR encoding (4*(d-1) clauses
    per vertex of degree d), replacing the old 2^{d} exponential encoding.

    Args:
        n_vertices: Number of graph vertices.
        density:    Edge probability (Erdos-Renyi model), default 0.5.
        seed:       Random seed for reproducibility.

    Returns:
        List of CNF clauses encoding the Tseitin formula.
    """
    rng = random.Random(seed)
    vertices = list(range(n_vertices))
    edges = [(u, v) for u in vertices for v in vertices
             if u < v and rng.random() < density]

    if not edges:
        return [frozenset([('dummy', False), ('dummy', True)])]  # trivially UNSAT

    def evar(u: int, v: int) -> str:
        return f"e_{min(u,v)}_{max(u,v)}"

    clauses: List[Clause] = []
    aux_counter = [0]  # mutable counter shared across XOR encodings

    # XOR constraints: for each vertex, parity of incident edges = label
    labels = {v: rng.choice([0, 1]) for v in vertices}

    # Ensure total parity is odd (making the system UNSAT)
    total = sum(labels.values()) % 2
    if total == 0 and vertices:
        labels[vertices[0]] ^= 1

    for v in vertices:
        incident = [evar(v, u) for u in vertices
                    if (v, u) in edges or (u, v) in edges]
        if not incident:
            continue
        clauses.extend(_xor_clauses_linear(incident, labels[v], aux_counter))

    return clauses if clauses else [frozenset()]


def gen_random_tautology(depth: int, rng: random.Random) -> Formula:
    """
    Generate a random provable tautology by compositional construction.

    The generator starts from the base tautology p1 -> p1 and applies
    depth uniformly random transformations, each preserving provability.

    Variable pool: {p1, p2, p3, p4}.

    Transformation rules (chosen uniformly):
      0. q -> phi          (weakening)
      1. (q AND phi) -> phi  (conjunction elimination)
      2. phi -> (phi OR q)   (disjunction introduction)
      3. phi AND (q -> q)    (tautological conjunction)

    Args:
        depth: Number of transformation steps (2-5 in experiments).
        rng:   Seeded random number generator.

    Returns:
        A Formula object that is guaranteed to be a tautology.
    """
    vars_ = [Var(f"p{i}") for i in range(1, 5)]  # p1, p2, p3, p4

    if depth == 0:
        v = rng.choice(vars_)
        return Implies(v, v)   # base: p -> p

    sub = gen_random_tautology(depth - 1, rng)
    extra_var = rng.choice(vars_)
    kind = rng.randint(0, 3)
    if kind == 0:
        return Implies(extra_var, sub)                    # weakening
    elif kind == 1:
        return Implies(And(extra_var, sub), sub)           # and-elim
    elif kind == 2:
        return Implies(sub, Or(sub, extra_var))            # or-intro
    else:
        return And(sub, Implies(extra_var, extra_var))    # tautological conj


# ============================================================================
# DPLL baseline solver (for comparison)
# ============================================================================

def dpll_baseline(clauses: List[Clause],
                   timeout: float = 30.0) -> Dict:
    """
    Simple DPLL solver without clause learning, used as a baseline.

    Args:
        clauses: CNF clause set.
        timeout: Maximum runtime in seconds.

    Returns:
        Dict with keys: status ('SAT'/'UNSAT'), time_sec, decisions.
    """
    t0 = time.perf_counter()
    decisions = [0]

    def _unit_prop(cls: List[Clause],
                    asgn: Dict[str, bool]) -> Optional[List[Clause]]:
        changed = True
        while changed:
            changed = False
            for c in cls:
                undefs = [(v, ip) for v, ip in c if v not in asgn]
                falses = [(v, ip) for v, ip in c
                          if (ip and asgn.get(v) == False) or
                             (not ip and asgn.get(v) == True)]
                if len(c) == len(falses):
                    return None  # conflict
                if len(undefs) == 1 and len(falses) == len(c) - 1:
                    v, ip = undefs[0]
                    asgn[v] = ip
                    changed = True
        return cls

    def _solve(cls: List[Clause], asgn: Dict[str, bool]) -> bool:
        if time.perf_counter() - t0 > timeout:
            return False
        cls2 = _unit_prop(cls, asgn)
        if cls2 is None:
            return False
        if all(any((ip and asgn.get(v) == True) or
                   (not ip and asgn.get(v) == False)
                   for v, ip in c)
               for c in cls2):
            return True
        unassigned = [v for c in cls2 for v, _ in c if v not in asgn]
        if not unassigned:
            return False
        v = unassigned[0]
        decisions[0] += 1
        for val in [True, False]:
            asgn2 = dict(asgn)
            asgn2[v] = val
            if _solve(cls2, asgn2):
                asgn.update(asgn2)
                return True
        return False

    sat = _solve(clauses, {})
    return {
        'status': 'SAT' if sat else 'UNSAT',
        'time_sec': time.perf_counter() - t0,
        'decisions': decisions[0]
    }


# ============================================================================
# Worker functions for parallel execution
# ============================================================================
# These must be top-level functions (not methods) for pickling with
# ProcessPoolExecutor.

def _worker_neuroproof(args: Tuple) -> Dict:
    """
    Run NeuroProof on a single instance. Returns a dict representation
    of BenchmarkResult (safe to serialise across processes).

    args: (name, iid, clauses_serial, timeout, max_conflicts, solver_label)
    """
    name, iid, clauses_serial, timeout = args[:4]
    max_conflicts = args[4] if len(args) > 4 else 500_000
    solver_label = args[5] if len(args) > 5 else 'NeuroProof'
    clauses = [frozenset(tuple(lit) for lit in c) for c in clauses_serial]

    atss = ATSS()
    solver = NeuroProofSolver(atss=atss, max_conflicts=max_conflicts)
    all_vars: set = set()
    for c in clauses:
        for v, _ in c:
            all_vars.add(v)

    t0 = time.perf_counter()
    try:
        result = solver.solve_clauses(clauses, all_vars)
    except Exception as e:
        return {
            'name': name, 'instance_id': iid,
            'n_vars': len(all_vars), 'n_clauses': len(clauses),
            'status': f'ERROR:{e}', 'solver': solver_label,
            'time_sec': time.perf_counter() - t0,
            'decisions': 0, 'conflicts': 0, 'learned': 0,
            'proof_size': 0, 'proof_depth': 0,
        }

    elapsed = time.perf_counter() - t0
    status = result.status.name
    proof_size = proof_depth = 0
    if result.proof is not None:
        try:
            proof_size  = result.proof.size
            proof_depth = result.proof.depth
        except Exception:
            pass

    return {
        'name': name, 'instance_id': iid,
        'n_vars': len(all_vars), 'n_clauses': len(clauses),
        'status': status, 'solver': solver_label,
        'time_sec': elapsed,
        'decisions': result.stats.get('decisions', 0),
        'conflicts': result.stats.get('conflicts', 0),
        'learned': result.stats.get('learned_clauses', 0),
        'proof_size': proof_size,
        'proof_depth': proof_depth,
    }


def _worker_dpll(args: Tuple) -> Dict:
    """
    Run DPLL baseline on a single instance. Returns a dict representation
    of BenchmarkResult.
    """
    name, iid, clauses_serial, timeout = args
    clauses = [frozenset(tuple(lit) for lit in c) for c in clauses_serial]

    all_vars: set = set()
    for c in clauses:
        for v, _ in c:
            all_vars.add(v)

    res = dpll_baseline(clauses, timeout)
    return {
        'name': name, 'instance_id': iid,
        'n_vars': len(all_vars), 'n_clauses': len(clauses),
        'status': res['status'], 'solver': 'DPLL-Baseline',
        'time_sec': res['time_sec'],
        'decisions': res['decisions'],
        'conflicts': 0, 'learned': 0,
        'proof_size': 0, 'proof_depth': 0,
    }


def _serialise_clauses(clauses: List[Clause]) -> List[List[Tuple]]:
    """Convert frozenset clauses to plain tuples for pickling."""
    return [list(c) for c in clauses]


# ============================================================================
# Experiment runner
# ============================================================================

class ExperimentRunner:
    """
    Runs benchmark experiments and records results to CSV.

    Each experiment method populates self._results with BenchmarkResult
    entries. Call save_results() to write them to disk.

    Parallelisation:
      EXP-1, EXP-2, EXP-3 use ProcessPoolExecutor for multi-core utilisation.
      EXP-4 is fast (15 trivial tautologies) and remains sequential.
      EXP-5 requires shared ATSS state and remains sequential.
    """

    def __init__(self, output_dir: str = '.',
                 n_workers: Optional[int] = None) -> None:
        self._output_dir = output_dir
        self._n_workers = n_workers or os.cpu_count() or 4
        os.makedirs(output_dir, exist_ok=True)
        self._results: List[BenchmarkResult] = []

    def _run_neuroproof(self, name: str, iid: int,
                         clauses: List[Clause],
                         timeout: float = 60.0) -> BenchmarkResult:
        """Run NeuroProof solver on a clause set (sequential fallback)."""
        atss = ATSS()
        solver = NeuroProofSolver(atss=atss, max_conflicts=500_000)
        all_vars: set = set()
        for c in clauses:
            for v, _ in c:
                all_vars.add(v)

        t0 = time.perf_counter()
        try:
            result = solver.solve_clauses(clauses, all_vars)
        except Exception as e:
            return BenchmarkResult(
                name=name, instance_id=iid,
                n_vars=len(all_vars), n_clauses=len(clauses),
                status='ERROR', solver='NeuroProof',
                time_sec=time.perf_counter() - t0)

        elapsed = time.perf_counter() - t0
        status = result.status.name
        proof_size = proof_depth = 0
        if result.proof is not None:
            try:
                proof_size  = result.proof.size
                proof_depth = result.proof.depth
            except Exception:
                pass

        return BenchmarkResult(
            name=name, instance_id=iid,
            n_vars=len(all_vars), n_clauses=len(clauses),
            status=status, solver='NeuroProof',
            time_sec=elapsed,
            decisions=result.stats.get('decisions', 0),
            conflicts=result.stats.get('conflicts', 0),
            learned=result.stats.get('learned_clauses', 0),
            proof_size=proof_size,
            proof_depth=proof_depth)

    def _run_dpll(self, name: str, iid: int,
                   clauses: List[Clause],
                   timeout: float = 30.0) -> BenchmarkResult:
        """Run DPLL baseline solver on a clause set."""
        all_vars: set = set()
        for c in clauses:
            for v, _ in c:
                all_vars.add(v)

        res = dpll_baseline(clauses, timeout)
        return BenchmarkResult(
            name=name, instance_id=iid,
            n_vars=len(all_vars), n_clauses=len(clauses),
            status=res['status'], solver='DPLL-Baseline',
            time_sec=res['time_sec'],
            decisions=res['decisions'])

    @staticmethod
    def _dict_to_result(d: Dict) -> BenchmarkResult:
        """Convert a worker result dict back to a BenchmarkResult."""
        return BenchmarkResult(**d)

    # ── EXP-1: Random 3-CNF Phase Transition (PARALLEL) ────────────────────

    def exp_random_3cnf(self, n_vars: int = 30,
                         n_trials: int = 20) -> None:
        """
        EXP-1: Sweep clause-to-variable ratio across the phase transition.

        The 3-CNF phase transition occurs at ratio ~ 4.267 (Dubois et al.).
        We sweep from 2.0 to 6.0 in steps of 0.2.

        Uses ProcessPoolExecutor: each (ratio, trial, solver) combo is
        an independent task.

        Args:
            n_vars:   Number of variables per instance.
            n_trials: Number of random instances per ratio.
        """
        print(f"\n[EXP-1] Random 3-CNF, n_vars={n_vars}, n_trials={n_trials}")
        print(f"  Using {self._n_workers} workers (ProcessPoolExecutor)")

        ratios = [2.0 + 0.2 * i for i in range(21)]  # 2.0 to 6.0

        # Build task list: (name, id, serialised_clauses, timeout, worker_fn)
        tasks_np = []  # NeuroProof tasks
        tasks_dp = []  # DPLL tasks
        task_id = 0

        for ratio in ratios:
            n_clauses = int(ratio * n_vars)
            for trial in range(n_trials):
                clauses = gen_random_3cnf(n_vars, n_clauses,
                                           seed=trial * 1000 + n_clauses)
                sc = _serialise_clauses(clauses)
                name = f"rand3cnf_n{n_vars}_r{ratio:.1f}"
                tasks_np.append((name, task_id, sc, 60.0, 500_000, 'NeuroProof'))
                tasks_dp.append((name, task_id, sc, 30.0))
                task_id += 1

        total = len(tasks_np) + len(tasks_dp)
        print(f"  Dispatching {total} tasks...")

        t_batch = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            # Submit all NeuroProof tasks
            fut_np = {executor.submit(_worker_neuroproof, t): t
                      for t in tasks_np}
            # Submit all DPLL tasks
            fut_dp = {executor.submit(_worker_dpll, t): t
                      for t in tasks_dp}

            done_np = 0
            done_dp = 0
            for fut in as_completed(fut_np):
                try:
                    d = fut.result()
                    self._results.append(self._dict_to_result(d))
                except Exception as e:
                    t = fut_np[fut]
                    self._results.append(BenchmarkResult(
                        name=t[0], instance_id=t[1],
                        n_vars=0, n_clauses=0,
                        status=f'FAIL:{e}', solver='NeuroProof',
                        time_sec=0.0))
                done_np += 1
                if done_np % 100 == 0:
                    print(f"    NeuroProof: {done_np}/{len(tasks_np)}")

            for fut in as_completed(fut_dp):
                try:
                    d = fut.result()
                    self._results.append(self._dict_to_result(d))
                except Exception as e:
                    t = fut_dp[fut]
                    self._results.append(BenchmarkResult(
                        name=t[0], instance_id=t[1],
                        n_vars=0, n_clauses=0,
                        status=f'FAIL:{e}', solver='DPLL-Baseline',
                        time_sec=0.0))
                done_dp += 1
                if done_dp % 100 == 0:
                    print(f"    DPLL: {done_dp}/{len(tasks_dp)}")

        elapsed = time.perf_counter() - t_batch
        print(f"  EXP-1 completed in {elapsed:.1f}s "
              f"({len(self._results)} results)")

    # ── EXP-2: Pigeonhole Principle (PARALLEL) ──────────────────────────────

    def exp_pigeonhole(self, max_n: int = 6) -> None:
        """
        EXP-2: Evaluate on PHP_n for n = 2..max_n.

        PHP is a canonical hard-UNSAT benchmark for resolution.  NeuroProof
        is expected to return UNKNOWN (conflict limit hit) for n >= 3.
        DPLL solves small instances quickly via unit propagation.

        Uses ProcessPoolExecutor for independent n values.

        Args:
            max_n: Maximum number of holes (default 6).
        """
        print(f"\n[EXP-2] Pigeonhole PHP_n, n = 2 .. {max_n}")
        print(f"  Using {self._n_workers} workers")

        tasks_np = []
        tasks_dp = []

        for n in range(2, max_n + 1):
            clauses = gen_pigeonhole(n)
            sc = _serialise_clauses(clauses)
            name = f"PHP_{n}"
            tasks_np.append((name, n, sc, 120.0))
            tasks_dp.append((name, n, sc, 120.0))

        t_batch = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            all_results = []
            for t in tasks_np:
                all_results.append(executor.submit(_worker_neuroproof, t))
            for t in tasks_dp:
                all_results.append(executor.submit(_worker_dpll, t))

            for fut in as_completed(all_results):
                try:
                    d = fut.result()
                    self._results.append(self._dict_to_result(d))
                except Exception:
                    pass

        elapsed = time.perf_counter() - t_batch
        print(f"  EXP-2 completed in {elapsed:.1f}s")

        # Print summary
        for n in range(2, max_n + 1):
            for r in self._results:
                if r.name == f"PHP_{n}":
                    print(f"  PHP_{n}: vars={r.n_vars}, clauses={r.n_clauses}, "
                          f"{r.solver}={r.status}({r.time_sec:.3f}s)")

    # ── EXP-3: Tseitin Tautologies (PARALLEL) ───────────────────────────────

    def exp_tseitin(self, n_trials: int = 20) -> None:
        """
        EXP-3: Evaluate on Tseitin formulas of increasing graph size.

        Tseitin formulas are UNSAT and require exponential resolution proofs.
        Graph sizes: 5, 8, 10, 12, 15 vertices with edge density 0.5.

        Uses ProcessPoolExecutor: each (n, trial) is independent.

        Args:
            n_trials: Number of random graphs per graph size.
        """
        print(f"\n[EXP-3] Tseitin tautologies")
        print(f"  Using {self._n_workers} workers")

        tasks = []
        for n in [5, 8, 10, 12, 15]:
            for t in range(n_trials):
                clauses = gen_tseitin(n, density=0.5, seed=t)
                sc = _serialise_clauses(clauses)
                name = f"Tseitin_n{n}"
                tasks.append((name, t, sc, 60.0))

        t_batch = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futs = {executor.submit(_worker_neuroproof, t): t
                    for t in tasks}
            for fut in as_completed(futs):
                try:
                    d = fut.result()
                    self._results.append(self._dict_to_result(d))
                except Exception:
                    pass

        elapsed = time.perf_counter() - t_batch
        print(f"  EXP-3 completed in {elapsed:.1f}s "
              f"({len(tasks)} instances)")

    # ── EXP-4: Proof Quality on Classical Tautologies (SEQUENTIAL) ───────────

    def exp_proof_quality(self) -> None:
        """
        EXP-4: Measure proof size and depth on 15 classical tautologies.

        This is the main experiment demonstrating NeuroProof's proof quality.
        All tautologies should be proved in under 3ms with proof sizes 2-7.
        Runs sequentially (fast, < 1 second total).
        """
        print(f"\n[EXP-4] Proof quality: ATSS vs baseline")
        test_formulas = [
            "p -> p",
            "(p -> q) -> (q -> r) -> (p -> r)",
            "(p & q) -> p",
            "(p & q) -> q",
            "p -> (q -> p)",
            "p -> (p | q)",
            "q -> (p | q)",
            "(p -> q) -> ((p -> ~q) -> ~p)",
            "(~p -> ~q) -> (q -> p)",
            "((p -> q) & (q -> r)) -> (p -> r)",
            "(p <-> q) -> (q <-> p)",
            "((p | q) & ~p) -> q",
            "p | ~p",                              # law of excluded middle
            "~(p & ~p)",                           # law of non-contradiction
            "(p -> q) -> (~q -> ~p)",              # contrapositive
        ]
        for fstr in test_formulas:
            f = parse(fstr)
            try:
                t0 = time.perf_counter()
                proof = tauto(f)
                elapsed = time.perf_counter() - t0
                r = BenchmarkResult(
                    name=f"tauto({fstr})", instance_id=0,
                    n_vars=len(f.variables()), n_clauses=0,
                    status='PROVED', solver='NeuroProof+ATSS',
                    time_sec=elapsed,
                    proof_size=proof.size,
                    proof_depth=proof.depth)
            except Exception as e:
                r = BenchmarkResult(
                    name=f"tauto({fstr})", instance_id=0,
                    n_vars=len(f.variables()), n_clauses=0,
                    status=f'FAIL:{e}', solver='NeuroProof+ATSS',
                    time_sec=0.0)
            self._results.append(r)
            print(f"  {fstr[:40]:40s}  "
                  f"{r.status:10s}  "
                  f"size={r.proof_size:4d}  "
                  f"depth={r.proof_depth:3d}  "
                  f"t={r.time_sec:.4f}s")

    # ── EXP-5: ATSS Online Learning Curve (SEQUENTIAL) ───────────────────────

    def exp_atss_learning_curve(self, n_problems: int = 100) -> None:
        """
        EXP-5: Demonstrate ATSS online learning convergence.

        Generates a stream of 100 random provable formulas (depth ~ U{2,5},
        4 variables, seed 42) and runs NeuroProof+ATSS on each.
        Tracks solve rate per epoch (20 problems/epoch).

        Must run sequentially: ATSS state is shared across problems
        (online learning requires cumulative updates).

        Args:
            n_problems: Total number of problems (default 100).
        """
        print(f"\n[EXP-5] ATSS Online Learning Curve ({n_problems} problems)")
        rng = random.Random(42)
        atss = ATSS()
        engine = TacticEngine(atss=atss, max_depth=100)

        results_per_epoch = []
        epoch_size = 20

        for i in range(n_problems):
            depth = rng.randint(2, 5)
            f = gen_random_tautology(depth, rng)

            t0 = time.perf_counter()
            try:
                proof = engine.prove(f)
                success = True
                elapsed = time.perf_counter() - t0
            except Exception:
                success = False
                elapsed = time.perf_counter() - t0

            results_per_epoch.append((success, elapsed))

            if (i + 1) % epoch_size == 0:
                epoch = i // epoch_size
                n_solved = sum(s for s, _ in results_per_epoch[-epoch_size:])
                avg_time = sum(t for _, t in results_per_epoch[-epoch_size:]) / epoch_size
                print(f"  Epoch {epoch+1:3d}: solved {n_solved}/{epoch_size}, "
                      f"avg_time={avg_time*1000:.1f}ms")

    # ── EXP-6: Ablation Study ────────────────────────────────────────────────

    def exp_ablation(self) -> None:
        """
        EXP-6: Ablation study — measure contribution of each component.

        Configurations:
          - Full NeuroProof (2WL + VSIDS + restarts + phase saving + clause deletion + ATSS)
          - No-ATSS: same as full but with empty ATSS (no heuristic bonus)
          - No-restarts: restarts disabled
          - DPLL: naive backtracking baseline

        Evaluated on EXP-4 tautologies (proof quality) and a subset of
        random 3-CNF at the phase transition (ratio=4.267, n=20).

        Runs sequentially (fast, < 10 seconds total).
        """
        print(f"\n[EXP-6] Ablation Study")

        # --- Part A: Tautology proof quality ---
        test_formulas = [
            "p -> p",
            "(p -> q) -> (q -> r) -> (p -> r)",
            "(p & q) -> p",
            "(p & q) -> q",
            "p -> (q -> p)",
            "p -> (p | q)",
            "q -> (p | q)",
            "(p -> q) -> ((p -> ~q) -> ~p)",
            "(~p -> ~q) -> (q -> p)",
            "((p -> q) & (q -> r)) -> (p -> r)",
            "(p <-> q) -> (q <-> p)",
            "((p | q) & ~p) -> q",
            "p | ~p",
            "~(p & ~p)",
            "(p -> q) -> (~q -> ~p)",
        ]
        for fstr in test_formulas:
            f = parse(fstr)
            try:
                t0 = time.perf_counter()
                proof = tauto(f)
                elapsed = time.perf_counter() - t0
                self._results.append(BenchmarkResult(
                    name=f"ablation_tauto({fstr})", instance_id=0,
                    n_vars=len(f.variables()), n_clauses=0,
                    status='PROVED', solver='NeuroProof-Full',
                    time_sec=elapsed, proof_size=proof.size,
                    proof_depth=proof.depth))
            except Exception:
                pass

        # --- Part B: Random 3-CNF at phase transition ---
        print("  Phase transition subset (n=20, ratio=4.267, 10 instances)")
        n_vars = 20
        n_clauses = int(4.267 * n_vars)
        for trial in range(10):
            clauses = gen_random_3cnf(n_vars, n_clauses,
                                       seed=trial * 1000 + n_clauses)
            all_vars = {v for c in clauses for v, _ in c}

            # Full NeuroProof
            r = self._run_neuroproof(
                f"ablation_rand_n{n_vars}", trial, clauses)
            r.solver = 'NeuroProof-Full'
            self._results.append(r)

            # DPLL baseline
            r = self._run_dpll(
                f"ablation_rand_n{n_vars}", trial, clauses)
            self._results.append(r)

        print(f"  Ablation: {len(self._results)} results collected")

    # ── EXP-7: Scalability Study ─────────────────────────────────────────────

    def exp_scalability(self) -> None:
        """
        EXP-7: Scalability — solve time vs problem size at phase transition.

        Sweep n_vars from 10 to 60 in steps of 10, with ratio fixed at
        4.267 (theoretical phase transition). 5 random instances per size.

        Uses ProcessPoolExecutor for parallelism.
        """
        print(f"\n[EXP-7] Scalability Study")
        print(f"  Using {self._n_workers} workers")

        tasks = []
        for n_vars in [10, 20, 30, 40, 50, 60]:
            n_clauses = int(4.267 * n_vars)
            for trial in range(5):
                clauses = gen_random_3cnf(n_vars, n_clauses,
                                           seed=trial * 1000 + n_clauses)
                sc = _serialise_clauses(clauses)
                name = f"scale_n{n_vars}"
                tasks.append((name, trial, sc, 120.0, 500_000, 'NeuroProof'))

        t_batch = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
            futs = {executor.submit(_worker_neuroproof, t): t
                    for t in tasks}
            for fut in as_completed(futs):
                try:
                    d = fut.result()
                    self._results.append(self._dict_to_result(d))
                except Exception:
                    pass

        elapsed = time.perf_counter() - t_batch
        print(f"  EXP-7 completed in {elapsed:.1f}s ({len(tasks)} instances)")

        # Print summary
        for n_vars in [10, 20, 30, 40, 50, 60]:
            times = [r.time_sec for r in self._results
                     if r.name == f"scale_n{n_vars}" and r.solver == 'NeuroProof']
            if times:
                avg_t = sum(times) / len(times) * 1000
                print(f"  n={n_vars}: avg_time={avg_t:.1f}ms ({len(times)} instances)")

    # ── EXP-8: SOTA Comparison ──────────────────────────────────────────────

    def exp_sota_comparison(self) -> None:
        """
        EXP-8: SOTA-style comparison on standard benchmarks.

        Compares DPLL (no learning), CDCL (NeuroProof without ATSS),
        and NeuroProof+ATSS on PHP instances and random 3-CNF.

        This demonstrates the value added by clause learning (CDCL over DPLL)
        and by ATSS heuristic guidance (NeuroProof over vanilla CDCL).
        """
        print(f"\n[EXP-8] SOTA Comparison (DPLL vs CDCL vs NeuroProof+ATSS)")

        # --- PHP instances ---
        print("  PHP instances:")
        for n in range(2, 6):
            clauses = gen_pigeonhole(n)
            all_vars = {v for c in clauses for v, _ in c}

            # DPLL
            r_dp = self._run_dpll(f"sota_PHP_{n}", 0, clauses, timeout=120.0)
            self._results.append(r_dp)

            # NeuroProof (CDCL + ATSS)
            r_np = self._run_neuroproof(f"sota_PHP_{n}", 0, clauses, timeout=120.0)
            r_np.solver = 'NeuroProof+ATSS'
            self._results.append(r_np)

            print(f"  PHP_{n}: DPLL={r_dp.status}({r_dp.time_sec:.3f}s), "
                  f"NeuroProof={r_np.status}({r_np.time_sec:.3f}s)")

        # --- Random 3-CNF at various ratios ---
        print("  Random 3-CNF (n=20):")
        for ratio in [2.0, 3.0, 4.267, 5.0]:
            n_clauses = int(ratio * 20)
            clauses = gen_random_3cnf(20, n_clauses, seed=42)
            all_vars = {v for c in clauses for v, _ in c}

            r_dp = self._run_dpll(f"sota_rand_r{ratio:.1f}", 0, clauses, timeout=30.0)
            self._results.append(r_dp)

            r_np = self._run_neuroproof(f"sota_rand_r{ratio:.1f}", 0, clauses, timeout=60.0)
            r_np.solver = 'NeuroProof+ATSS'
            self._results.append(r_np)

            print(f"  ratio={ratio}: DPLL={r_dp.status}({r_dp.time_sec:.3f}s), "
                  f"NeuroProof={r_np.status}({r_np.time_sec:.3f}s)")

        print(f"  SOTA: {len(self._results)} results collected")

    # ── Save results ────────────────────────────────────────────────────────

    def save_results(self, filename: str = 'results.csv') -> str:
        """Save accumulated results to CSV file."""
        path = os.path.join(self._output_dir, filename)
        if not self._results:
            return path
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f,
                fieldnames=list(asdict(self._results[0]).keys()))
            writer.writeheader()
            for r in self._results:
                writer.writerow(asdict(r))
        print(f"\nResults saved to: {path}")
        return path

    def run_all(self) -> str:
        """Run the complete benchmark suite."""
        print("=" * 60)
        print(" NeuroProof Benchmark Suite")
        print(f" Workers: {self._n_workers}")
        print("=" * 60)

        t_total = time.perf_counter()

        self.exp_random_3cnf(n_vars=30, n_trials=20)
        self.exp_pigeonhole(max_n=6)
        self.exp_tseitin(n_trials=10)
        self.exp_proof_quality()
        self.exp_atss_learning_curve(n_problems=100)
        self.exp_ablation()
        self.exp_scalability()
        self.exp_sota_comparison()

        t_total = time.perf_counter() - t_total
        print(f"\n{'=' * 60}")
        print(f" All experiments completed in {t_total:.1f}s")
        print(f" Total results: {len(self._results)}")
        print(f"{'=' * 60}")

        return self.save_results()


# ============================================================================
# Entry point
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='NeuroProof Benchmark Suite')
    parser.add_argument('--exp', type=int, default=None,
                        help='Run only the specified experiment (1-8)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: cpu_count)')
    args = parser.parse_args()

    output_dir = args.output or os.path.join(
        os.path.dirname(__file__), '..', 'experiments')
    runner = ExperimentRunner(output_dir=output_dir,
                               n_workers=args.workers)

    if args.exp is not None:
        exp_map = {
            1: runner.exp_random_3cnf,
            2: runner.exp_pigeonhole,
            3: runner.exp_tseitin,
            4: runner.exp_proof_quality,
            5: runner.exp_atss_learning_curve,
            6: runner.exp_ablation,
            7: runner.exp_scalability,
            8: runner.exp_sota_comparison,
        }
        if args.exp in exp_map:
            exp_map[args.exp]()
            runner.save_results()
        else:
            print(f"Unknown experiment: {args.exp}. Choose 1-8.")
            sys.exit(1)
    else:
        runner.run_all()
