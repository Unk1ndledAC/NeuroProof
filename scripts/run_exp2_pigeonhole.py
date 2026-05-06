"""
run_exp2_pigeonhole.py
=======================
EXP-2: Pigeonhole Principle (PHP_n^{n+1}) benchmark.

Tests NeuroProof and DPLL on the canonical resolution-hard benchmark.
Expected: DPLL returns UNSAT for all n; NeuroProof returns UNKNOWN for n >= 3.

Usage:
    python scripts/run_exp2_pigeonhole.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_suite import ExperimentRunner

if __name__ == '__main__':
    runner = ExperimentRunner(output_dir=os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    runner.exp_pigeonhole(max_n=6)
    runner.save_results()
