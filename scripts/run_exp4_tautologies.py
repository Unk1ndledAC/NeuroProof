"""
run_exp4_tautologies.py
=======================
EXP-4: Classical tautology proof quality benchmark.

Measures proof size, depth, and timing for 15 classical tautologies.
This is the main experiment demonstrating NeuroProof's proof quality.

Usage:
    python scripts/run_exp4_tautologies.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.benchmark_suite import ExperimentRunner

if __name__ == '__main__':
    runner = ExperimentRunner(output_dir=os.path.join(os.path.dirname(__file__), '..', 'experiments'))
    runner.exp_proof_quality()
    runner.save_results()
