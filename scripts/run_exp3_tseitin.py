"""
run_exp3_tseitin.py
===================
EXP-3: Tseitin tautology benchmark.

Tests NeuroProof on Tseitin formulas derived from random graphs
(5, 8, 10, 12, 15 vertices). Tseitin formulas are UNSAT with
exponential resolution proof complexity.

Usage:
    python scripts/run_exp3_tseitin.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_suite import ExperimentRunner

if __name__ == '__main__':
    runner = ExperimentRunner(output_dir=os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    runner.exp_tseitin(n_trials=20)
    runner.save_results()
