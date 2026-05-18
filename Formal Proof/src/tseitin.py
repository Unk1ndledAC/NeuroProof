"""
tseitin.py
==========
Tseitin Transformation: linear-size CNF encoding for propositional formulas.

Reference:
  Tseitin (1968): On the Complexity of Derivation in Propositional Calculus.
  In: Automation of Reasoning 2, Springer (1983), pp. 466-483.
  DOI: 10.1007/978-3-642-81955-1_28
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from .formula import (Formula, Var, Unary, Binary, _Constant,
                      Connective, Top, Bot, And, Or, Implies, Not)


class TseitinEncoder:
    """
    Equisatisfiable CNF encoding via Tseitin transformation.

    For each subformula φ, we introduce a fresh variable t_φ and add
    clauses that encode the biconditional  t_φ ↔ (definition of φ).

    The output formula is a conjunction of clauses (in CNF) that is
    satisfiable iff the original formula is satisfiable.  Additionally,
    a unit clause {t_root} asserts that the original formula is true.
    """

    def __init__(self) -> None:
        self._counter: int = 0
        self._cache: Dict[int, Var] = {}   # formula id → auxiliary variable

    def fresh(self) -> Var:
        """Generate a fresh auxiliary variable."""
        self._counter += 1
        return Var(f"_t{self._counter}")

    def _aux(self, f: Formula) -> Var:
        """Return the auxiliary variable for subformula f, creating if needed."""
        fid = id(f)
        if fid not in self._cache:
            self._cache[fid] = self.fresh()
        return self._cache[fid]

    def encode(self, f: Formula) -> Formula:
        """
        Encode formula f into an equisatisfiable CNF formula.

        Returns a conjunction of clauses.
        """
        clauses: List[Formula] = []
        t_root = self._encode_rec(f, clauses)
        # Assert that the root auxiliary variable is true
        clauses.append(t_root)
        return self._big_and(clauses)

    def _encode_rec(self, f: Formula, clauses: List[Formula]) -> Formula:
        """
        Recursively encode subformula f, appending clauses and returning
        the auxiliary variable representing f.
        """
        if isinstance(f, _Constant):
            return f
        if isinstance(f, Var):
            return f  # atoms are their own representatives

        t = self._aux(f)

        if isinstance(f, Unary):
            assert f.connective == Connective.NOT
            t_child = self._encode_rec(f.child, clauses)
            # t ↔ ¬c  ≡  (t ∨ c) ∧ (¬t ∨ ¬c)
            clauses.append(Or(t, t_child))
            clauses.append(Or(Not(t), Not(t_child)))
            return t

        assert isinstance(f, Binary)
        t_l = self._encode_rec(f.left,  clauses)
        t_r = self._encode_rec(f.right, clauses)

        if f.connective == Connective.AND:
            # t ↔ (l ∧ r)  ≡  (t→l) ∧ (t→r) ∧ (l∧r→t)
            #              ≡  (¬t∨l) ∧ (¬t∨r) ∧ (¬l∨¬r∨t)
            clauses.append(Or(Not(t), t_l))
            clauses.append(Or(Not(t), t_r))
            clauses.append(Or(Or(Not(t_l), Not(t_r)), t))

        elif f.connective == Connective.OR:
            # t ↔ (l ∨ r)  ≡  (¬t∨l∨r) ∧ (t∨¬l) ∧ (t∨¬r)
            clauses.append(Or(Or(Not(t), t_l), t_r))
            clauses.append(Or(t, Not(t_l)))
            clauses.append(Or(t, Not(t_r)))

        elif f.connective == Connective.IMP:
            # t ↔ (l→r)  ≡  t ↔ (¬l ∨ r)
            # (¬t ∨ ¬l ∨ r) ∧ (t ∨ l) ∧ (t ∨ ¬r)
            clauses.append(Or(Or(Not(t), Not(t_l)), t_r))
            clauses.append(Or(t, t_l))
            clauses.append(Or(t, Not(t_r)))

        elif f.connective == Connective.IFF:
            # t ↔ (l ↔ r)
            # (¬t ∨ ¬l ∨ r) ∧ (¬t ∨ l ∨ ¬r) ∧ (t ∨ l ∨ r) ∧ (t ∨ ¬l ∨ ¬r)
            clauses.append(Or(Or(Not(t), Not(t_l)), t_r))
            clauses.append(Or(Or(Not(t), t_l), Not(t_r)))
            clauses.append(Or(Or(t, t_l), t_r))
            clauses.append(Or(Or(t, Not(t_l)), Not(t_r)))

        elif f.connective == Connective.XOR:
            # t ↔ (l ⊕ r)  ≡  t ↔ (¬(l ↔ r))
            # (¬t ∨ l ∨ r) ∧ (¬t ∨ ¬l ∨ ¬r) ∧ (t ∨ ¬l ∨ r) ∧ (t ∨ l ∨ ¬r)
            clauses.append(Or(Or(Not(t), t_l), t_r))
            clauses.append(Or(Or(Not(t), Not(t_l)), Not(t_r)))
            clauses.append(Or(Or(t, Not(t_l)), t_r))
            clauses.append(Or(Or(t, t_l), Not(t_r)))

        return t

    @staticmethod
    def _big_and(clauses: List[Formula]) -> Formula:
        """Fold a list of clauses into a big conjunction."""
        if not clauses:
            return Top
        result = clauses[-1]
        for c in reversed(clauses[:-1]):
            result = And(c, result)
        return result
