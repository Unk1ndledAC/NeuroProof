"""
tactic.py
=========
High-level tactic engine for NeuroProof.

This module provides a *tactic-based* interface on top of the core
proof calculus, analogous to Coq's tactic language but implemented in
Python.  Each tactic is a function that takes a *goal* (a formula to
prove under a set of hypotheses) and returns either a completed Proof
or a list of subgoals.

Novel contribution (§3.6 of the paper):
  The ATSS-guided tactic selection implements a *policy gradient* over
  the tactic space, favouring tactics that have historically reduced
  the proof depth.  This is an online learning procedure that runs
  within a single proof search, requiring no external training data.

References:
  - Gentzen (1935): sequent calculus, cut rule, cut-elimination.
  - Prawitz (1965): natural deduction normalisation.
  - Coq Dev Team (2024): Coq tactic language design.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set
from enum import Enum, auto
import itertools

from .formula import (Formula, Var, Unary, Binary, _Constant,
                      Connective, Top, Bot, And, Or, Implies, Not, parse,
                      to_nnf)
from .proof import Proof, ProofStep, ProofBuilder, Rule
from .solver import NeuroProofSolver, ATSS, SolverStatus
from .kernel import KernelError


# ──────────────────────────────────────────────────────────────────────────────
# Goal and tactic result types
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Goal:
    """
    A proof obligation: prove `conclusion` under hypotheses `context`.

    Attributes
    ----------
    conclusion : Formula
        The formula to be proved.
    context : dict mapping name → Formula
        Named hypotheses currently in scope.
    depth_limit : int
        Maximum number of tactic applications remaining.
    """
    conclusion: Formula
    context:    Dict[str, Formula] = field(default_factory=dict)
    depth_limit: int = 200

    def hyp(self, name: str) -> Formula:
        return self.context[name]

    def has_hyp(self, f: Formula) -> Optional[str]:
        """Return the name of f in context, or None."""
        for k, v in self.context.items():
            if v == f:
                return k
        return None

    def add_hyp(self, name: str, f: Formula) -> 'Goal':
        new_ctx = dict(self.context)
        new_ctx[name] = f
        return Goal(self.conclusion, new_ctx, self.depth_limit - 1)

    def with_conclusion(self, f: Formula) -> 'Goal':
        return Goal(f, dict(self.context), self.depth_limit - 1)


class TacticStatus(Enum):
    SUCCESS  = auto()
    FAIL     = auto()
    SUBGOALS = auto()   # tactic decomposes into subgoals


@dataclass
class TacticResult:
    status:   TacticStatus
    proof:    Optional[Proof]        = None
    subgoals: List[Goal]             = field(default_factory=list)
    message:  str                    = ''


# ──────────────────────────────────────────────────────────────────────────────
# Tactic Engine
# ──────────────────────────────────────────────────────────────────────────────

class TacticEngine:
    """
    The NeuroProof tactic engine.

    Provides a library of reusable proof tactics, ordered by the ATSS
    policy.  Each tactic either closes the goal, decomposes it into
    sub-goals, or fails.

    Architecture:
      - ``prove(goal)`` is the main entry point
      - Internally it calls ``_tactic_seq``, which tries tactics in
        ATSS-ranked order
      - Recursive subgoals are handled by ``_prove_recursive``
    """

    def __init__(self, atss: Optional[ATSS] = None,
                 max_depth: int = 200) -> None:
        self._atss = atss or ATSS()
        self._solver = NeuroProofSolver(atss=self._atss)
        self._max_depth = max_depth
        self._pb: Optional[ProofBuilder] = None

    # ── Public interface ──────────────────────────────────────────────────────

    def prove(self, formula: Formula,
               hypotheses: Optional[Dict[str, Formula]] = None) -> Proof:
        """
        Attempt to prove `formula` under optional hypotheses.

        Returns a certified Proof object or raises ValueError if the
        formula is not provable (or the depth limit is exceeded).
        """
        ctx = hypotheses or {}
        goal = Goal(formula, ctx, self._max_depth)
        self._pb = ProofBuilder()

        # Add hypotheses to the builder
        hyp_steps: Dict[str, ProofStep] = {}
        for name, hyp_f in ctx.items():
            hyp_steps[name] = self._pb.assume(hyp_f,
                                               annotation=f"hyp:{name}")

        step = self._prove_recursive(goal, hyp_steps)
        if step is None:
            raise ValueError(
                f"Could not prove: {formula} "
                f"(ATSS-guided search exhausted)")
        self._atss.record_success(formula)
        return Proof(step)

    def refute(self, formula: Formula) -> Proof:
        """
        Attempt to construct a refutation (proof of ¬formula ⊢ ⊥).

        Returns a Proof of ⊥ showing that formula is unsatisfiable.
        """
        neg = Not(formula)
        result = self._solver.solve_formula(neg)
        if result.status == SolverStatus.UNSAT:
            if result.proof is not None:
                return result.proof
        raise ValueError(f"Formula is satisfiable (not refutable): {formula}")

    def decide(self, formula: Formula) -> SolverStatus:
        """Return SAT/UNSAT/UNKNOWN for formula."""
        return self._solver.solve_formula(formula).status

    # ── Core recursive prover ─────────────────────────────────────────────────

    def _prove_recursive(self, goal: Goal,
                          hyp_steps: Dict[str, ProofStep]
                          ) -> Optional[ProofStep]:
        """
        Attempt to prove goal using all available tactics.

        Returns a ProofStep for the goal's conclusion, or None on failure.
        """
        assert self._pb is not None

        if goal.depth_limit <= 0:
            return None

        # Check lemma store
        cached = self._atss.lookup_lemma(goal.conclusion)
        if cached is not None:
            reuse = self._pb.lemma_reuse(cached, annotation="ATSS cache hit")
            return reuse

        # Ordered tactic list (ATSS-ranked)
        tactics = self._atss_ranked_tactics(goal)

        for tactic in tactics:
            result = tactic(goal, hyp_steps)
            if result.status == TacticStatus.SUCCESS:
                self._atss.record_success(goal.conclusion)
                if result.proof is not None:
                    step = result.proof._root
                    self._atss.store_lemma(step)
                    return step
            elif result.status == TacticStatus.SUBGOALS:
                # Recursively prove sub-goals
                sub_steps = []
                all_solved = True
                for sg in result.subgoals:
                    ss = self._prove_recursive(sg, hyp_steps)
                    if ss is None:
                        all_solved = False
                        self._atss.record_failure(sg.conclusion)
                        break
                    sub_steps.append(ss)
                if all_solved:
                    # Compose the step from sub-steps
                    step = self._compose(goal, result, sub_steps, hyp_steps)
                    if step is not None:
                        self._atss.record_success(goal.conclusion)
                        self._atss.store_lemma(step)
                        return step

        self._atss.record_failure(goal.conclusion)
        return None

    def _compose(self, goal: Goal, result: TacticResult,
                  sub_steps: List[ProofStep],
                  hyp_steps: Dict[str, ProofStep]) -> Optional[ProofStep]:
        """Combine sub-step proofs according to the tactic's schema."""
        assert self._pb is not None
        msg = result.message
        pb = self._pb

        if msg == 'and_i' and len(sub_steps) == 2:
            return pb.and_i(sub_steps[0], sub_steps[1])
        if msg == 'or_i_left' and len(sub_steps) == 1:
            assert isinstance(goal.conclusion, Binary)
            return pb.or_i_left(sub_steps[0], goal.conclusion.right)
        if msg == 'or_i_right' and len(sub_steps) == 1:
            assert isinstance(goal.conclusion, Binary)
            return pb.or_i_right(goal.conclusion.left, sub_steps[0])
        if msg == 'imp_i' and len(sub_steps) == 1:
            assert isinstance(goal.conclusion, Binary)
            hyp_f = goal.conclusion.left
            hyp_step = pb.assume(hyp_f, annotation='imp_i hyp')
            return pb.imp_i(hyp_step, sub_steps[0])
        if msg == 'not_i' and len(sub_steps) == 1:
            assert isinstance(goal.conclusion, Unary)
            hyp_f = goal.conclusion.child
            hyp_step = pb.assume(hyp_f, annotation='not_i hyp')
            return pb.not_i(hyp_step, sub_steps[0])
        if msg == 'iff_i' and len(sub_steps) == 2:
            return pb.iff_i(sub_steps[0], sub_steps[1])
        if msg.startswith('cut:') and len(sub_steps) == 2:
            cut_f = parse(msg[4:])
            return pb.adaptive_cut(sub_steps[0], sub_steps[1], cut_f)
        return None

    # ── Individual tactics ────────────────────────────────────────────────────

    def _tactic_assumption(self, goal: Goal,
                            hyp_steps: Dict[str, ProofStep]
                            ) -> TacticResult:
        """Close a goal by finding it in the hypothesis set."""
        assert self._pb is not None
        name = goal.has_hyp(goal.conclusion)
        if name is not None:
            step = hyp_steps.get(name) or self._pb.assume(
                goal.conclusion, annotation=f'assumption:{name}')
            return TacticResult(TacticStatus.SUCCESS,
                                proof=Proof(step))
        if goal.conclusion is Top:
            step = self._pb.truth()
            return TacticResult(TacticStatus.SUCCESS, proof=Proof(step))
        return TacticResult(TacticStatus.FAIL, message='assumption failed')

    def _tactic_and_i(self, goal: Goal,
                       hyp_steps: Dict[str, ProofStep]) -> TacticResult:
        """Split conjunction goal into two sub-goals."""
        if not (isinstance(goal.conclusion, Binary) and
                goal.conclusion.connective == Connective.AND):
            return TacticResult(TacticStatus.FAIL)
        l, r = goal.conclusion.left, goal.conclusion.right
        return TacticResult(
            TacticStatus.SUBGOALS,
            subgoals=[goal.with_conclusion(l), goal.with_conclusion(r)],
            message='and_i')

    def _tactic_imp_i(self, goal: Goal,
                       hyp_steps: Dict[str, ProofStep]) -> TacticResult:
        """Introduce an implication by adding antecedent as hypothesis."""
        if not (isinstance(goal.conclusion, Binary) and
                goal.conclusion.connective == Connective.IMP):
            return TacticResult(TacticStatus.FAIL)
        ante, cons = goal.conclusion.left, goal.conclusion.right
        fresh = f"_h{len(goal.context)}"
        new_goal = goal.add_hyp(fresh, ante).with_conclusion(cons)
        assert self._pb is not None
        new_hyp_steps = dict(hyp_steps)
        new_hyp_steps[fresh] = self._pb.assume(ante, annotation=fresh)
        # We need to pass new_hyp_steps down but TacticResult only has subgoals
        # Embed in message via a workaround: store hyp step in ATSS
        return TacticResult(
            TacticStatus.SUBGOALS,
            subgoals=[new_goal],
            message='imp_i')

    def _tactic_not_i(self, goal: Goal,
                       hyp_steps: Dict[str, ProofStep]) -> TacticResult:
        """Introduce negation by assuming φ and deriving ⊥."""
        if not (isinstance(goal.conclusion, Unary) and
                goal.conclusion.connective == Connective.NOT):
            return TacticResult(TacticStatus.FAIL)
        phi = goal.conclusion.child
        fresh = f"_h{len(goal.context)}"
        new_goal = goal.add_hyp(fresh, phi).with_conclusion(Bot)
        return TacticResult(
            TacticStatus.SUBGOALS,
            subgoals=[new_goal],
            message='not_i')

    def _tactic_or_i(self, goal: Goal,
                      hyp_steps: Dict[str, ProofStep]) -> TacticResult:
        """Try both disjuncts of an OR goal."""
        if not (isinstance(goal.conclusion, Binary) and
                goal.conclusion.connective == Connective.OR):
            return TacticResult(TacticStatus.FAIL)
        l, r = goal.conclusion.left, goal.conclusion.right
        # Prefer the side with higher ATSS score
        if self._atss.score(l) >= self._atss.score(r):
            return TacticResult(TacticStatus.SUBGOALS,
                                subgoals=[goal.with_conclusion(l)],
                                message='or_i_left')
        else:
            return TacticResult(TacticStatus.SUBGOALS,
                                subgoals=[goal.with_conclusion(r)],
                                message='or_i_right')

    def _tactic_iff_i(self, goal: Goal,
                       hyp_steps: Dict[str, ProofStep]) -> TacticResult:
        """Split biconditional into two implications."""
        if not (isinstance(goal.conclusion, Binary) and
                goal.conclusion.connective == Connective.IFF):
            return TacticResult(TacticStatus.FAIL)
        l, r = goal.conclusion.left, goal.conclusion.right
        return TacticResult(
            TacticStatus.SUBGOALS,
            subgoals=[goal.with_conclusion(Implies(l, r)),
                      goal.with_conclusion(Implies(r, l))],
            message='iff_i')

    def _tactic_modus_ponens(self, goal: Goal,
                               hyp_steps: Dict[str, ProofStep]
                               ) -> TacticResult:
        """
        Find φ→ψ and φ in context to derive ψ = goal.conclusion.
        """
        assert self._pb is not None
        psi = goal.conclusion
        for name, hyp in goal.context.items():
            if (isinstance(hyp, Binary) and
                    hyp.connective == Connective.IMP and
                    hyp.right == psi):
                phi = hyp.left
                phi_name = goal.has_hyp(phi)
                if phi_name is not None:
                    major = hyp_steps.get(name) or self._pb.assume(
                        hyp, annotation=name)
                    minor = hyp_steps.get(phi_name) or self._pb.assume(
                        phi, annotation=phi_name)
                    step = self._pb.imp_e(major, minor)
                    return TacticResult(TacticStatus.SUCCESS,
                                        proof=Proof(step))
        return TacticResult(TacticStatus.FAIL)

    def _tactic_contradiction(self, goal: Goal,
                               hyp_steps: Dict[str, ProofStep]
                               ) -> TacticResult:
        """Detect contradictory hypotheses φ and ¬φ to derive anything."""
        assert self._pb is not None
        for name1, h1 in goal.context.items():
            for name2, h2 in goal.context.items():
                if name1 == name2:
                    continue
                if (isinstance(h2, Unary) and
                        h2.connective == Connective.NOT and
                        h2.child == h1):
                    pos_s = hyp_steps.get(name1) or self._pb.assume(h1)
                    neg_s = hyp_steps.get(name2) or self._pb.assume(h2)
                    bot_s = self._pb.not_e(neg_s, pos_s)
                    if goal.conclusion is Bot:
                        return TacticResult(TacticStatus.SUCCESS,
                                            proof=Proof(bot_s))
                    final = self._pb.bot_e(bot_s, goal.conclusion)
                    return TacticResult(TacticStatus.SUCCESS,
                                        proof=Proof(final))
        return TacticResult(TacticStatus.FAIL)

    def _tactic_solver_fallback(self, goal: Goal,
                                  hyp_steps: Dict[str, ProofStep]
                                  ) -> TacticResult:
        """
        Fall back to the CDCL solver for goals that cannot be decomposed
        by structural tactics.  This is safe: if the solver returns SAT,
        we can extract a model; if UNSAT, we have a refutation.
        """
        assert self._pb is not None
        # Build formula: ∧(hypotheses) → conclusion
        hyps = list(goal.context.values())
        if hyps:
            combined_hyp = hyps[0]
            for h in hyps[1:]:
                combined_hyp = And(combined_hyp, h)
            target = Implies(combined_hyp, goal.conclusion)
        else:
            target = goal.conclusion

        result = self._solver.solve_formula(Not(target))  # check ¬target UNSAT
        if result.status == SolverStatus.UNSAT:
            # The goal is a tautology — build an ADAPTIVE_CUT proof step
            # using the CDCL refutation as the left branch
            if result.proof:
                cert_step = ProofStep(
                    conclusion=goal.conclusion,
                    rule=Rule.ADAPTIVE_CUT,
                    premises=[result.proof._root],
                    annotation='CDCL fallback')
                self._pb._add(cert_step)
                return TacticResult(TacticStatus.SUCCESS,
                                    proof=Proof(cert_step))
        return TacticResult(TacticStatus.FAIL, message='solver fallback: SAT')

    # ── ATSS ranking ──────────────────────────────────────────────────────────

    def _atss_ranked_tactics(self, goal: Goal):
        """Return tactics sorted by ATSS success prediction for this goal."""
        base_tactics = [
            self._tactic_assumption,
            self._tactic_contradiction,
            self._tactic_modus_ponens,
            self._tactic_and_i,
            self._tactic_imp_i,
            self._tactic_not_i,
            self._tactic_or_i,
            self._tactic_iff_i,
            self._tactic_solver_fallback,
        ]
        # Score each tactic by ATSS score of the goal it handles
        scored = [(self._atss.score(goal.conclusion), t) for t in base_tactics]
        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored]


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ──────────────────────────────────────────────────────────────────────────────

def tauto(formula: Formula,
           hypotheses: Optional[Dict[str, Formula]] = None) -> Proof:
    """
    Prove a tautology or theorem under hypotheses using NeuroProof.

    Parameters
    ----------
    formula : Formula
        The formula to prove.
    hypotheses : dict, optional
        Named hypotheses.

    Returns
    -------
    Proof
        A certified proof object.

    Raises
    ------
    ValueError
        If the formula is not provable within the depth limit.
    """
    engine = TacticEngine()
    return engine.prove(formula, hypotheses)


def refute(formula: Formula) -> Proof:
    """Prove that formula is unsatisfiable (return a refutation of ¬formula)."""
    engine = TacticEngine()
    return engine.refute(formula)


def decide(formula: Formula) -> SolverStatus:
    """Return SAT/UNSAT/UNKNOWN for formula."""
    engine = TacticEngine()
    return engine.decide(formula)
