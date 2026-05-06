# NeuroProof

**A Hybrid Propositional Proof System with Adaptive Tactic Synthesis and Certified Proof Checking**

NeuroProof is a hybrid propositional proof system that combines natural deduction, sequent calculus, and resolution with three novel rules: ADAPTIVE_CUT, LEMMA_REUSE, and INTERPOLANT. It features ATSS (Adaptive Tactic Synthesis System), an online bandit-style learning component that guides proof search without pre-training.

## Key Features

- **Hybrid proof calculus**: Natural deduction + sequent calculus + resolution rules
- **Novel rules**: ADAPTIVE_CUT (learned cut formula selection), LEMMA_REUSE (proof DAG edge reuse), INTERPOLANT (Craig interpolation via CDCL)
- **ATSS**: Online tactic synthesis with zero pre-training (EMA updates, softmax policy)
- **Certified checking**: Dual Python/Rocq verification chain following the de Bruijn criterion
- **DAG proof compression**: Proof size reduction of s - Omega(log s)

## Project Structure

```
NeuroProof/
├── src/                          # Core library (pure Python, no dependencies)
│   ├── __init__.py               # Public API exports
│   ├── formula.py                # Formula AST, parser, NNF/CNF transformations
│   ├── proof.py                  # Proof steps, ProofBuilder, Rule enum
│   ├── kernel.py                 # Trusted verification kernel (TCB, 287 lines)
│   ├── solver.py                 # CDCL solver + ATSS + Craig interpolation
│   ├── tactic.py                 # Tactic engine (9 tactics)
│   └── tseitin.py                # Tseitin CNF encoding
├── experiments/
│   ├── __init__.py
│   ├── benchmark_suite.py        # Full benchmark suite (all 5 experiments)
│   ├── plot_results.py           # Publication-quality plot generation
│   ├── results.csv               # Experiment output data
│   └── figures/                  # Generated plots (PDF)
├── scripts/
│   ├── run_exp1_phase_transition.py
│   ├── run_exp2_pigeonhole.py
│   ├── run_exp3_tseitin.py
│   ├── run_exp4_tautologies.py
│   ├── run_exp5_atss_learning.py
│   └── run_all_experiments.py
├── coq/
│   └── NeuroProof.v              # Rocq/Coq formalisation (soundness + ADAPTIVE_CUT)
├── LICENSE
└── README.md
```

## Requirements

- **Python**: 3.10+ (tested on 3.12.4)
- **Dependencies**: None for core library (pure Python standard library)
- **Optional**: `matplotlib`, `numpy`, `pandas` for plot generation
- **Optional**: Rocq/Coq 8.19+ for formal verification

## Quick Start

### 1. Verify Installation

```bash
cd NeuroProof
python -c "from src import tauto, parse; p = tauto(parse('p -> p')); print(f'OK: size={p.size}')"
```

Expected output: `OK: size=2`

### 2. Run Experiments

Run individual experiments:

```bash
# EXP-4: Classical tautology proofs (fastest, ~1 second)
python scripts/run_exp4_tautologies.py

# EXP-2: Pigeonhole Principle (~2 minutes)
python scripts/run_exp2_pigeonhole.py

# EXP-5: ATSS online learning (~30 seconds)
python scripts/run_exp5_atss_learning.py

# EXP-1: Random 3-CNF phase transition (~5 minutes)
python scripts/run_exp1_phase_transition.py

# EXP-3: Tseitin tautologies (~2 minutes)
python scripts/run_exp3_tseitin.py
```

Run all experiments at once:

```bash
python scripts/run_all_experiments.py
```

Results are saved to `experiments/results.csv`.

### 3. Generate Plots (optional)

```bash
pip install matplotlib numpy pandas
python experiments/plot_results.py
```

Plots are saved to `experiments/figures/`.

### 4. Rocq Formal Verification (optional)

```bash
cd coq
coqc NeuroProof.v
```

## Public API

```python
from src import Var, Not, And, Or, Implies, parse, tauto, decide, NeuroProofSolver

# Parse and prove a tautology
f = parse("(p -> q) -> ((not q) -> (not p))")  # contrapositive
proof = tauto(f)
print(f"Proof size: {proof.size}, depth: {proof.depth}")

# SAT solving
result = decide(parse("p | q"))
print(f"Status: {result}")  # SAT

# CNF solving via CDCL
from src.solver import NeuroProofSolver, Clause
solver = NeuroProofSolver(max_conflicts=10000)
clauses = [
    frozenset([('x1', True), ('x2', True)]),
    frozenset([('x1', False), ('x2', True)]),
    frozenset([('x1', True), ('x2', False)]),
]
result = solver.solve_clauses(clauses, {'x1', 'x2'})
print(f"Status: {result.status}")
```

## Architecture

### Trusted Computing Base (TCB)

The verification kernel (`kernel.py`, 287 lines) is intentionally minimal:
- Each proof step is verified by pattern-matching against the rule definition
- All other modules (ATSS, interpolation, tactic engine) produce `ProofStep` objects that pass through the kernel
- A bug in untrusted components cannot produce a false proof that passes verification

### ATSS (Adaptive Tactic Synthesis System)

- Maintains a tactic embedding table: formula hash -> (success_count, attempt_count)
- Updates via exponential moving average (decay=0.95)
- Cut formula selection by maximizing cosine similarity (sparse bag-of-subformulas)
- Learns online during proof search, no pre-training required

### CDCL Solver

- Standard CDCL with 1-UIP conflict analysis and clause learning
- ATSS-enriched VSIDS heuristic for variable selection
- Craig interpolant extraction via Pudlak's algorithm
- Proof logging via `ProofStep` DAG construction

## Experiments

| Experiment | Description | Duration |
|-----------|-------------|----------|
| EXP-1 | Random 3-CNF phase transition (n=30, 20 trials/ratio) | ~5 min |
| EXP-2 | Pigeonhole Principle PHP_n (n=2..6) | ~2 min |
| EXP-3 | Tseitin tautologies (graph sizes 5-15) | ~2 min |
| EXP-4 | 15 classical tautologies (proof quality) | ~1 sec |
| EXP-5 | ATSS online learning curve (100 problems) | ~30 sec |

## Citation

```bibtex
@article{qu2026neuroproof,
  title={NeuroProof: A Hybrid Propositional Proof System with Adaptive Tactic Synthesis and Certified Proof Checking},
  author={Qu, Guanheng and Zhang, Chunxiao and Liu, Jiangming},
  journal={},
  year={2026}
}
```
