"""
run_exp1_phase_transition.py
============================
EXP-1: Random 3-CNF phase transition benchmark.

Sweeps the clause-to-variable ratio from 2.0 to 6.0 around the
3-CNF phase transition (~4.267) and compares NeuroProof vs DPLL.

Usage:
    python scripts/run_exp1_phase_transition.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_suite import ExperimentRunner

if __name__ == '__main__':
    runner = ExperimentRunner(output_dir=os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    runner.exp_random_3cnf(n_vars=30, n_trials=20)
    runner.save_results()
