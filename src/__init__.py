"""
src/__init__.py
===============
NeuroProof: A Hybrid Propositional Proof System with Adaptive Tactic
Synthesis and Certified Proof Checking.

Public API:
  - Formula types: Var, Not, And, Or, Implies, Iff, Xor, Top, Bot
  - parse(s): parse an infix formula string
  - ProofBuilder: fluent proof construction API
  - Proof: immutable certified proof object
  - NeuroProofSolver: CDCL solver with ATSS and interpolation
  - TacticEngine: high-level tactic-based prover
"""

from .formula import (
    Formula, Var, Not, And, Or, Implies, Iff, Xor, Top, Bot,
    Unary, Binary, Connective, parse,
    to_nnf, to_cnf, eliminate_iff
)
from .proof import Proof, ProofStep, ProofBuilder, Rule
from .kernel import verify_step, verify_step_strict, KernelError
from .solver import NeuroProofSolver, ATSS, SolverStatus, SolverResult
from .tactic import TacticEngine, TacticResult, decide, tauto, refute

__version__ = "1.0.0"
__all__ = [
    # Formulas
    "Formula", "Var", "Not", "And", "Or", "Implies", "Iff", "Xor",
    "Top", "Bot", "Unary", "Binary", "Connective",
    "parse", "to_nnf", "to_cnf", "eliminate_iff",
    # Proofs
    "Proof", "ProofStep", "ProofBuilder", "Rule",
    # Kernel
    "verify_step", "verify_step_strict", "KernelError",
    # Solver
    "NeuroProofSolver", "ATSS", "SolverStatus", "SolverResult",
    # Tactics
    "TacticEngine", "TacticResult", "decide", "tauto", "refute",
]
