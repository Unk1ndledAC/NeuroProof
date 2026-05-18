"""
proof.py
========
Proof objects and formal derivation rules for NeuroProof.

This module implements the *proof calculus* of NeuroProof, a hybrid system
combining three complementary rule sets:

  1. Natural Deduction (ND) — Prawitz-style introduction/elimination rules
  2. Sequent Calculus (SC) — Gentzen LK rules with explicit cut
  3. Resolution Calculus (RC) — Robinson-style resolution with compactness

Each proof step is a *certified* object: it carries the formula it derives,
a list of premises (previous steps), the rule applied, and an optional
*Rocq witness* (a string encoding the corresponding Coq proof term, to be
verified by the external Rocq kernel via the verify.v adapter).

Theoretical background:
  - Gentzen (1935): sequent calculus LK, cut-elimination theorem.
  - Prawitz (1965): natural deduction, normalisation.
  - Robinson (1965): resolution principle.
  - Cook & Reckhow (1979): propositional proof systems framework.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, FrozenSet, Dict
import hashlib

from .formula import Formula, Var, Not, And, Or, Implies, Iff, Top, Bot
from .formula import Unary, Binary, Connective


# ──────────────────────────────────────────────────────────────────────────────
# Rule identifiers
# ──────────────────────────────────────────────────────────────────────────────

class Rule(Enum):
    # Axiom-like
    AXIOM        = auto()   # Assumption (hypothesis)
    REFL         = auto()   # φ ⊢ φ
    TRUTH        = auto()   # ⊢ ⊤

    # Natural Deduction — Introduction
    AND_I        = auto()   # φ, ψ ⊢ φ ∧ ψ
    OR_I_LEFT    = auto()   # φ ⊢ φ ∨ ψ
    OR_I_RIGHT   = auto()   # ψ ⊢ φ ∨ ψ
    IMP_I        = auto()   # [φ] ψ ⊢ φ → ψ  (assumption discharged)
    NOT_I        = auto()   # [φ] ⊥ ⊢ ¬φ
    IFF_I        = auto()   # φ→ψ, ψ→φ ⊢ φ↔ψ

    # Natural Deduction — Elimination
    AND_E_LEFT   = auto()   # φ ∧ ψ ⊢ φ
    AND_E_RIGHT  = auto()   # φ ∧ ψ ⊢ ψ
    OR_E         = auto()   # φ∨ψ, [φ]χ, [ψ]χ ⊢ χ
    IMP_E        = auto()   # φ→ψ, φ ⊢ ψ  (modus ponens)
    NOT_E        = auto()   # ¬φ, φ ⊢ ⊥
    BOT_E        = auto()   # ⊥ ⊢ φ         (ex falso quodlibet)
    IFF_E_LEFT   = auto()   # φ↔ψ, φ ⊢ ψ
    IFF_E_RIGHT  = auto()   # φ↔ψ, ψ ⊢ φ
    DNE          = auto()   # ¬¬φ ⊢ φ       (double negation elimination, classical)

    # Sequent Calculus rules
    SC_AX        = auto()   # Γ,φ ⊢ φ,Δ
    SC_CUT       = auto()   # Γ⊢φ,Δ  Γ',φ⊢Δ'  /  Γ,Γ'⊢Δ,Δ'
    SC_WEAKL     = auto()   # Γ⊢Δ  /  Γ,φ⊢Δ
    SC_WEAKR     = auto()   # Γ⊢Δ  /  Γ⊢φ,Δ
    SC_CONTRL    = auto()   # Γ,φ,φ⊢Δ  /  Γ,φ⊢Δ
    SC_AND_L1    = auto()   # Γ,φ⊢Δ  /  Γ,φ∧ψ⊢Δ
    SC_AND_R     = auto()   # Γ⊢φ,Δ  Γ⊢ψ,Δ  /  Γ⊢φ∧ψ,Δ
    SC_OR_L      = auto()   # Γ,φ⊢Δ  Γ,ψ⊢Δ  /  Γ,φ∨ψ⊢Δ
    SC_OR_R1     = auto()   # Γ⊢φ,Δ  /  Γ⊢φ∨ψ,Δ
    SC_IMP_L     = auto()   # Γ⊢φ,Δ  Γ,ψ⊢Δ  /  Γ,φ→ψ⊢Δ
    SC_IMP_R     = auto()   # Γ,φ⊢ψ,Δ  /  Γ⊢φ→ψ,Δ
    SC_NOT_L     = auto()   # Γ⊢φ,Δ  /  Γ,¬φ⊢Δ
    SC_NOT_R     = auto()   # Γ,φ⊢Δ  /  Γ⊢¬φ,Δ

    # Resolution Calculus
    RES_UNIT     = auto()   # unit resolution
    RES_FULL     = auto()   # full resolution step
    RES_FACTOR   = auto()   # factoring

    # NeuroProof — novel rules (contributions of this paper)
    ADAPTIVE_CUT   = auto()  # learned cut formula selection (§3.3)
    INTERPOLANT    = auto()  # Craig interpolation step (§3.4)
    LEMMA_REUSE    = auto()  # proof graph edge reuse (§3.5)


# ──────────────────────────────────────────────────────────────────────────────
# Proof step (node in the proof DAG)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProofStep:
    """
    A single certified step in a formal proof derivation.

    Attributes
    ----------
    conclusion : Formula
        The formula proved by this step.
    rule : Rule
        The inference rule applied.
    premises : list of ProofStep
        Antecedent steps (empty for axioms).
    discharged : list of Formula
        Hypotheses discharged by this step (used in IMP_I, NOT_I, OR_E).
    annotation : str
        Human-readable annotation for the step.
    rocq_term : str or None
        Optional Rocq/Coq proof term string for external certification.
    _id : int
        Unique identifier based on the step's content hash.
    """
    conclusion:  Formula
    rule:        Rule
    premises:    List['ProofStep']         = field(default_factory=list)
    discharged:  List[Formula]             = field(default_factory=list)
    annotation:  str                       = field(default='')
    rocq_term:   Optional[str]             = field(default=None)
    _id:         int                       = field(init=False)

    def __post_init__(self) -> None:
        content = (str(self.conclusion) + self.rule.name
                   + ''.join(str(p._id) for p in self.premises))
        self._id = int(hashlib.sha256(content.encode()).hexdigest()[:8], 16)

    def __repr__(self) -> str:
        return (f"ProofStep(rule={self.rule.name}, "
                f"conclusion={self.conclusion}, id={self._id:#010x})")

    def assumptions(self) -> FrozenSet[Formula]:
        """Return the set of undischarged assumptions of this step."""
        own = frozenset(self.discharged)
        from_prems = frozenset().union(
            *(p.assumptions() for p in self.premises)
        )
        return from_prems - own


# ──────────────────────────────────────────────────────────────────────────────
# Proof object: a verified derivation
# ──────────────────────────────────────────────────────────────────────────────

class Proof:
    """
    An immutable, verified proof derivation.

    A Proof wraps the top-level ProofStep and provides:
      - ``conclusion``: the proved formula
      - ``assumptions``: the undischarged hypotheses
      - ``size``: number of rule applications
      - ``depth``: length of the longest proof path
      - ``is_theorem``: True iff there are no undischarged assumptions
      - ``to_dag()``: convert the proof to a directed acyclic graph (for
        lemma-reuse analysis in NeuroProof's ADAPTIVE_CUT procedure)
      - ``check()``: structural re-verification of the proof
    """

    def __init__(self, root: ProofStep) -> None:
        self._root = root

    @property
    def conclusion(self) -> Formula:
        return self._root.conclusion

    @property
    def assumptions(self) -> FrozenSet[Formula]:
        return self._root.assumptions()

    @property
    def is_theorem(self) -> bool:
        return len(self.assumptions) == 0

    @property
    def size(self) -> int:
        """Number of distinct proof steps (DAG nodes)."""
        return len(self._all_steps())

    @property
    def depth(self) -> int:
        """Longest chain from a leaf to the root."""
        return self._depth(self._root, {})

    def _depth(self, step: ProofStep, memo: Dict[int, int]) -> int:
        if step._id in memo:
            return memo[step._id]
        d = 0 if not step.premises else (
            1 + max(self._depth(p, memo) for p in step.premises))
        memo[step._id] = d
        return d

    def _all_steps(self) -> List[ProofStep]:
        """BFS traversal returning all unique ProofStep nodes."""
        visited: Dict[int, ProofStep] = {}
        queue = [self._root]
        while queue:
            step = queue.pop()
            if step._id not in visited:
                visited[step._id] = step
                queue.extend(step.premises)
        return list(visited.values())

    def check(self) -> bool:
        """
        Structural kernel verification.

        Traverses the proof DAG and re-checks each rule application by calling
        the rule-specific verification function in ``kernel.py``.

        Returns True iff the proof is structurally valid.

        Note: This does NOT call the external Rocq kernel; for that,
        see ``certify.py``.
        """
        from .kernel import verify_step
        for step in self._all_steps():
            if not verify_step(step):
                return False
        return True

    def to_dag(self) -> Tuple[List[ProofStep], List[Tuple[int, int]]]:
        """
        Return (nodes, edges) where edges are (premise._id, conclusion._id).
        Used by the ADAPTIVE_CUT heuristic.
        """
        nodes = self._all_steps()
        edges: List[Tuple[int, int]] = []
        for step in nodes:
            for prem in step.premises:
                edges.append((prem._id, step._id))
        return nodes, edges

    def __repr__(self) -> str:
        return (f"Proof(conclusion={self.conclusion}, "
                f"size={self.size}, theorem={self.is_theorem})")


# ──────────────────────────────────────────────────────────────────────────────
# Proof builder: fluent API for constructing proofs
# ──────────────────────────────────────────────────────────────────────────────

class ProofBuilder:
    """
    A stateful builder for constructing formal proofs step by step.

    Example usage::

        pb = ProofBuilder()
        h1 = pb.assume(parse('p -> q'))
        h2 = pb.assume(parse('p'))
        s  = pb.imp_e(h1, h2)          # derives q by modus ponens
        pf = pb.build()                # close and verify
    """

    def __init__(self) -> None:
        self._steps: List[ProofStep] = []
        self._last: Optional[ProofStep] = None

    def _add(self, step: ProofStep) -> ProofStep:
        self._steps.append(step)
        self._last = step
        return step

    # ── Axioms ────────────────────────────────────────────────────────────────

    def assume(self, formula: Formula, annotation: str = '') -> ProofStep:
        """Introduce an assumption (undischarged hypothesis)."""
        return self._add(ProofStep(
            conclusion=formula, rule=Rule.AXIOM,
            premises=[], annotation=annotation or f"[{formula}]"))

    def truth(self) -> ProofStep:
        """Derive ⊤ by the TRUTH axiom."""
        return self._add(ProofStep(
            conclusion=Top, rule=Rule.TRUTH, annotation='⊢ ⊤'))

    # ── Natural Deduction introduction rules ──────────────────────────────────

    def and_i(self, left: ProofStep, right: ProofStep) -> ProofStep:
        """AND-I: derive φ ∧ ψ from proofs of φ and ψ."""
        return self._add(ProofStep(
            conclusion=And(left.conclusion, right.conclusion),
            rule=Rule.AND_I, premises=[left, right],
            annotation=f"∧I"))

    def or_i_left(self, step: ProofStep, rhs: Formula) -> ProofStep:
        """OR-I-L: derive φ ∨ ψ from a proof of φ."""
        return self._add(ProofStep(
            conclusion=Or(step.conclusion, rhs),
            rule=Rule.OR_I_LEFT, premises=[step],
            annotation=f"∨I-L"))

    def or_i_right(self, lhs: Formula, step: ProofStep) -> ProofStep:
        """OR-I-R: derive φ ∨ ψ from a proof of ψ."""
        return self._add(ProofStep(
            conclusion=Or(lhs, step.conclusion),
            rule=Rule.OR_I_RIGHT, premises=[step],
            annotation=f"∨I-R"))

    def imp_i(self, hyp: ProofStep, body: ProofStep) -> ProofStep:
        """IMP-I: derive φ → ψ by discharging the assumption φ."""
        return self._add(ProofStep(
            conclusion=Implies(hyp.conclusion, body.conclusion),
            rule=Rule.IMP_I, premises=[body],
            discharged=[hyp.conclusion],
            annotation=f"→I [{hyp.conclusion}]"))

    def not_i(self, hyp: ProofStep, bot: ProofStep) -> ProofStep:
        """NOT-I: derive ¬φ by discharging assumption φ from a proof of ⊥."""
        assert bot.conclusion is Bot, "not_i requires a proof of ⊥"
        return self._add(ProofStep(
            conclusion=Not(hyp.conclusion),
            rule=Rule.NOT_I, premises=[bot],
            discharged=[hyp.conclusion],
            annotation=f"¬I [{hyp.conclusion}]"))

    def iff_i(self, fwd: ProofStep, bwd: ProofStep) -> ProofStep:
        """IFF-I: derive φ ↔ ψ from φ→ψ and ψ→φ."""
        assert isinstance(fwd.conclusion, Binary)
        assert isinstance(bwd.conclusion, Binary)
        assert fwd.conclusion.connective == Connective.IMP
        assert bwd.conclusion.connective == Connective.IMP
        phi, psi = fwd.conclusion.left, fwd.conclusion.right
        return self._add(ProofStep(
            conclusion=Iff(phi, psi),
            rule=Rule.IFF_I, premises=[fwd, bwd],
            annotation='↔I'))

    # ── Natural Deduction elimination rules ───────────────────────────────────

    def and_e_left(self, step: ProofStep) -> ProofStep:
        """AND-E-L: derive φ from φ ∧ ψ."""
        assert isinstance(step.conclusion, Binary)
        assert step.conclusion.connective == Connective.AND
        return self._add(ProofStep(
            conclusion=step.conclusion.left,
            rule=Rule.AND_E_LEFT, premises=[step], annotation='∧E-L'))

    def and_e_right(self, step: ProofStep) -> ProofStep:
        """AND-E-R: derive ψ from φ ∧ ψ."""
        assert isinstance(step.conclusion, Binary)
        assert step.conclusion.connective == Connective.AND
        return self._add(ProofStep(
            conclusion=step.conclusion.right,
            rule=Rule.AND_E_RIGHT, premises=[step], annotation='∧E-R'))

    def imp_e(self, major: ProofStep, minor: ProofStep) -> ProofStep:
        """IMP-E (modus ponens): derive ψ from φ→ψ and φ."""
        assert isinstance(major.conclusion, Binary)
        assert major.conclusion.connective == Connective.IMP
        assert major.conclusion.left == minor.conclusion, (
            f"MP antecedent mismatch: expected {major.conclusion.left}, "
            f"got {minor.conclusion}")
        return self._add(ProofStep(
            conclusion=major.conclusion.right,
            rule=Rule.IMP_E, premises=[major, minor], annotation='→E'))

    def not_e(self, neg: ProofStep, pos: ProofStep) -> ProofStep:
        """NOT-E: derive ⊥ from ¬φ and φ."""
        assert isinstance(neg.conclusion, Unary)
        assert neg.conclusion.connective == Connective.NOT
        assert neg.conclusion.child == pos.conclusion, (
            f"NOT-E mismatch: ¬{neg.conclusion.child} vs {pos.conclusion}")
        return self._add(ProofStep(
            conclusion=Bot,
            rule=Rule.NOT_E, premises=[neg, pos], annotation='¬E'))

    def bot_e(self, bot: ProofStep, phi: Formula) -> ProofStep:
        """BOT-E (ex falso): derive any φ from ⊥."""
        assert bot.conclusion is Bot, "bot_e requires a proof of ⊥"
        return self._add(ProofStep(
            conclusion=phi,
            rule=Rule.BOT_E, premises=[bot], annotation=f'⊥E → {phi}'))

    def dne(self, step: ProofStep) -> ProofStep:
        """DNE: derive φ from ¬¬φ (classical logic only)."""
        assert isinstance(step.conclusion, Unary)
        assert step.conclusion.connective == Connective.NOT
        inner = step.conclusion.child
        assert isinstance(inner, Unary)
        assert inner.connective == Connective.NOT
        return self._add(ProofStep(
            conclusion=inner.child,
            rule=Rule.DNE, premises=[step], annotation='DNE'))

    def iff_e_left(self, iff: ProofStep, phi: ProofStep) -> ProofStep:
        """IFF-E-L: derive ψ from φ↔ψ and φ."""
        assert isinstance(iff.conclusion, Binary)
        assert iff.conclusion.connective == Connective.IFF
        assert iff.conclusion.left == phi.conclusion
        return self._add(ProofStep(
            conclusion=iff.conclusion.right,
            rule=Rule.IFF_E_LEFT, premises=[iff, phi], annotation='↔E-L'))

    def or_e(self, disj: ProofStep, left_case: ProofStep,
              right_case: ProofStep,
              hyp_left: ProofStep, hyp_right: ProofStep) -> ProofStep:
        """
        OR-E: derive χ from φ∨ψ, [φ]→χ, [ψ]→χ.

        The hypotheses hyp_left and hyp_right will be discharged.
        """
        assert isinstance(disj.conclusion, Binary)
        assert disj.conclusion.connective == Connective.OR
        assert left_case.conclusion == right_case.conclusion, (
            "OR-E: both branches must conclude the same formula")
        return self._add(ProofStep(
            conclusion=left_case.conclusion,
            rule=Rule.OR_E,
            premises=[disj, left_case, right_case],
            discharged=[hyp_left.conclusion, hyp_right.conclusion],
            annotation='∨E'))

    # ── Novel NeuroProof rules ─────────────────────────────────────────────────

    def adaptive_cut(self, left: ProofStep, right: ProofStep,
                     cut_formula: Formula) -> ProofStep:
        """
        ADAPTIVE_CUT (NeuroProof §3.3):
        A learned variant of the sequent calculus cut rule where the cut
        formula is selected by the ATSS (Adaptive Tactic Synthesis System)
        based on subgoal embedding similarity.

        Semantics: from Γ⊢φ,Δ and Γ',φ⊢Δ' derive Γ,Γ'⊢Δ,Δ'.
        """
        return self._add(ProofStep(
            conclusion=right.conclusion,
            rule=Rule.ADAPTIVE_CUT,
            premises=[left, right],
            annotation=f'ADAPTIVE-CUT on [{cut_formula}]'))

    def interpolant(self, phi: ProofStep, psi: ProofStep,
                    itp: Formula) -> ProofStep:
        """
        INTERPOLANT (NeuroProof §3.4):
        Craig interpolation step — synthesise a formula I such that
          φ ⊢ I   and   I ⊢ ψ,
        witnessing φ ⊢ ψ via the intermediate I.
        """
        return self._add(ProofStep(
            conclusion=psi.conclusion,
            rule=Rule.INTERPOLANT,
            premises=[phi, psi],
            annotation=f'INTERPOLANT: [{itp}]'))

    def lemma_reuse(self, lemma: ProofStep,
                    annotation: str = '') -> ProofStep:
        """
        LEMMA_REUSE (NeuroProof §3.5):
        Explicit proof-DAG edge reuse — reference a previously proved
        lemma to avoid duplication in the proof graph.
        """
        return self._add(ProofStep(
            conclusion=lemma.conclusion,
            rule=Rule.LEMMA_REUSE,
            premises=[lemma],
            annotation=annotation or f'Lemma: {lemma.conclusion}'))

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self) -> Proof:
        """
        Close the proof and return a verified Proof object.

        Raises
        ------
        ValueError
            If no steps have been added or if the proof fails structural
            verification.
        """
        if self._last is None:
            raise ValueError("ProofBuilder has no steps")
        proof = Proof(self._last)
        if not proof.check():
            raise ValueError(
                f"Proof verification failed for conclusion: {proof.conclusion}")
        return proof
