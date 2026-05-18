"""
formula.py
==========
Abstract Syntax Tree (AST) for classical propositional logic formulas.

This module defines the core data structures for representing propositional
formulas, including:
  - Atomic propositions (variables)
  - Logical connectives: negation (NOT), conjunction (AND), disjunction (OR),
    implication (IMP), biconditional (IFF), and the truth constants (TOP/BOT).

Design goals:
  1. Immutability: all Formula objects are frozen dataclasses.
  2. Hashability: formulas can be used as dictionary keys and set members.
  3. Structural equality: two formulas are equal iff they have identical ASTs.
  4. Pretty printing: standard mathematical notation for readability.

Reference:
  Cook & Reckhow (1979), Section 2 – Definitions of propositional proof systems.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Set, Iterator
from enum import Enum, auto


# ──────────────────────────────────────────────────────────────────────────────
# Connector tags
# ──────────────────────────────────────────────────────────────────────────────

class Connective(Enum):
    VAR  = auto()   # atomic proposition
    TOP  = auto()   # verum  (⊤)
    BOT  = auto()   # falsum (⊥)
    NOT  = auto()   # ¬φ
    AND  = auto()   # φ ∧ ψ
    OR   = auto()   # φ ∨ ψ
    IMP  = auto()   # φ → ψ
    IFF  = auto()   # φ ↔ ψ
    XOR  = auto()   # φ ⊕ ψ  (exclusive or, used in parity reasoning)


# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Formula:
    """Base class for all propositional formulas."""

    connective: Connective

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def is_literal(self) -> bool:
        """Return True if the formula is a literal (atom or negated atom)."""
        if self.connective == Connective.VAR:
            return True
        if self.connective == Connective.NOT:
            assert isinstance(self, Unary)
            return self.child.connective == Connective.VAR
        return False

    @property
    def is_clause(self) -> bool:
        """Return True if the formula is a clause (disjunction of literals)."""
        if self.is_literal:
            return True
        if self.connective == Connective.OR:
            assert isinstance(self, Binary)
            return self.left.is_clause and self.right.is_clause
        return False

    @property
    def size(self) -> int:
        """Return the number of connectives (not counting variables)."""
        raise NotImplementedError

    @property
    def depth(self) -> int:
        """Return the nesting depth of the formula tree."""
        raise NotImplementedError

    def variables(self) -> FrozenSet[str]:
        """Return the set of all variable names occurring in the formula."""
        raise NotImplementedError

    def subformulas(self) -> FrozenSet['Formula']:
        """Return the set of all subformulas (including self)."""
        raise NotImplementedError

    # ── Operator overloads for convenient formula construction ────────────────

    def __and__(self, other: 'Formula') -> 'Formula':
        return And(self, other)

    def __or__(self, other: 'Formula') -> 'Formula':
        return Or(self, other)

    def __invert__(self) -> 'Formula':
        return Not(self)

    def __rshift__(self, other: 'Formula') -> 'Formula':
        """φ >> ψ  stands for φ → ψ."""
        return Implies(self, other)

    def __eq__(self, other: object) -> bool:  # must be explicitly re-declared
        if not isinstance(other, Formula):
            return NotImplemented
        return object.__eq__(self, other)     # rely on frozen dataclass __eq__

    def __hash__(self) -> int:
        return object.__hash__(self)


# ──────────────────────────────────────────────────────────────────────────────
# Atomic formulas
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Var(Formula):
    """Atomic proposition (propositional variable)."""
    name: str

    def __init__(self, name: str) -> None:
        object.__setattr__(self, 'connective', Connective.VAR)
        object.__setattr__(self, 'name', name)

    @property
    def size(self) -> int:
        return 0

    @property
    def depth(self) -> int:
        return 0

    def variables(self) -> FrozenSet[str]:
        return frozenset({self.name})

    def subformulas(self) -> FrozenSet['Formula']:
        return frozenset({self})

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Var({self.name!r})"


@dataclass(frozen=True)
class _Constant(Formula):
    """Truth constant (⊤ or ⊥)."""

    def __init__(self, conn: Connective) -> None:
        object.__setattr__(self, 'connective', conn)

    @property
    def size(self) -> int:
        return 0

    @property
    def depth(self) -> int:
        return 0

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    def subformulas(self) -> FrozenSet['Formula']:
        return frozenset({self})

    def __str__(self) -> str:
        return '⊤' if self.connective == Connective.TOP else '⊥'

    def __repr__(self) -> str:
        return 'Top' if self.connective == Connective.TOP else 'Bot'


# Singletons for truth constants
Top: Formula = _Constant(Connective.TOP)
Bot: Formula = _Constant(Connective.BOT)


# ──────────────────────────────────────────────────────────────────────────────
# Unary connective: negation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Unary(Formula):
    """Unary formula: ¬φ."""
    child: Formula

    def __init__(self, conn: Connective, child: Formula) -> None:
        object.__setattr__(self, 'connective', conn)
        object.__setattr__(self, 'child', child)

    @property
    def size(self) -> int:
        return 1 + self.child.size

    @property
    def depth(self) -> int:
        return 1 + self.child.depth

    def variables(self) -> FrozenSet[str]:
        return self.child.variables()

    def subformulas(self) -> FrozenSet['Formula']:
        return self.child.subformulas() | frozenset({self})

    def __str__(self) -> str:
        inner = str(self.child)
        if self.child.connective not in (Connective.VAR, Connective.TOP,
                                         Connective.BOT, Connective.NOT):
            inner = f"({inner})"
        return f"¬{inner}"

    def __repr__(self) -> str:
        return f"Not({self.child!r})"


def Not(child: Formula) -> Formula:
    """Construct a negation, applying double-negation elimination eagerly."""
    if isinstance(child, Unary) and child.connective == Connective.NOT:
        return child.child   # ¬¬φ = φ
    if child is Top:
        return Bot
    if child is Bot:
        return Top
    return Unary(Connective.NOT, child)


# ──────────────────────────────────────────────────────────────────────────────
# Binary connectives
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Binary(Formula):
    """Binary formula: φ ◇ ψ."""
    left:  Formula
    right: Formula

    def __init__(self, conn: Connective, left: Formula,
                 right: Formula) -> None:
        object.__setattr__(self, 'connective', conn)
        object.__setattr__(self, 'left', left)
        object.__setattr__(self, 'right', right)

    @property
    def size(self) -> int:
        return 1 + self.left.size + self.right.size

    @property
    def depth(self) -> int:
        return 1 + max(self.left.depth, self.right.depth)

    def variables(self) -> FrozenSet[str]:
        return self.left.variables() | self.right.variables()

    def subformulas(self) -> FrozenSet['Formula']:
        return (self.left.subformulas()
                | self.right.subformulas()
                | frozenset({self}))

    def __str__(self) -> str:
        sym = {Connective.AND: '∧', Connective.OR: '∨',
               Connective.IMP: '→', Connective.IFF: '↔',
               Connective.XOR: '⊕'}[self.connective]
        def _wrap(f: Formula) -> str:
            if f.connective in (Connective.VAR, Connective.TOP,
                                Connective.BOT):
                return str(f)
            if isinstance(f, Unary):
                return str(f)
            return f"({f})"
        return f"{_wrap(self.left)} {sym} {_wrap(self.right)}"

    def __repr__(self) -> str:
        name = {Connective.AND: 'And', Connective.OR: 'Or',
                Connective.IMP: 'Implies', Connective.IFF: 'Iff',
                Connective.XOR: 'Xor'}[self.connective]
        return f"{name}({self.left!r}, {self.right!r})"


def And(left: Formula, right: Formula) -> Formula:
    """Construct φ ∧ ψ with simplification on truth constants."""
    if left is Bot or right is Bot:
        return Bot
    if left is Top:
        return right
    if right is Top:
        return left
    return Binary(Connective.AND, left, right)


def Or(left: Formula, right: Formula) -> Formula:
    """Construct φ ∨ ψ with simplification on truth constants."""
    if left is Top or right is Top:
        return Top
    if left is Bot:
        return right
    if right is Bot:
        return left
    return Binary(Connective.OR, left, right)


def Implies(left: Formula, right: Formula) -> Formula:
    """Construct φ → ψ with simplification."""
    if left is Bot:
        return Top       # ex falso quodlibet
    if left is Top:
        return right
    if right is Top:
        return Top
    return Binary(Connective.IMP, left, right)


def Iff(left: Formula, right: Formula) -> Formula:
    """Construct φ ↔ ψ."""
    return Binary(Connective.IFF, left, right)


def Xor(left: Formula, right: Formula) -> Formula:
    """Construct φ ⊕ ψ (exclusive or)."""
    return Binary(Connective.XOR, left, right)


# ──────────────────────────────────────────────────────────────────────────────
# Formula parsing (infix string → AST)
# ──────────────────────────────────────────────────────────────────────────────

def parse(s: str) -> Formula:
    """
    Parse a propositional formula from an infix string.

    Supported syntax:
      - Variables:    single identifier [a-zA-Z][a-zA-Z0-9_]*
      - Constants:    T (⊤), F (⊥)
      - Not:          ~, !, ¬
      - And:          &, &&, /\\, ∧
      - Or:           |, ||, \\/, ∨
      - Implies:      ->, =>, →
      - Iff:          <->, <=>, ↔
      - Parentheses:  (φ)

    Precedence (lowest to highest): IFF, IMP, OR, AND, NOT.
    """
    tokens = _tokenize(s)
    pos = [0]

    def peek() -> str:
        return tokens[pos[0]] if pos[0] < len(tokens) else 'EOF'

    def consume() -> str:
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_iff() -> Formula:
        lhs = parse_imp()
        if peek() in ('<->', '<=>', '↔'):
            consume()
            rhs = parse_iff()
            return Iff(lhs, rhs)
        return lhs

    def parse_imp() -> Formula:
        lhs = parse_or()
        if peek() in ('->', '=>', '→'):
            consume()
            rhs = parse_imp()
            return Implies(lhs, rhs)
        return lhs

    def parse_or() -> Formula:
        lhs = parse_and()
        while peek() in ('|', '||', '\\/', '∨'):
            consume()
            rhs = parse_and()
            lhs = Or(lhs, rhs)
        return lhs

    def parse_and() -> Formula:
        lhs = parse_not()
        while peek() in ('&', '&&', '/\\', '∧'):
            consume()
            rhs = parse_not()
            lhs = And(lhs, rhs)
        return lhs

    def parse_not() -> Formula:
        if peek() in ('~', '!', '¬'):
            consume()
            return Not(parse_not())
        return parse_atom()

    def parse_atom() -> Formula:
        t = peek()
        if t == '(':
            consume()
            f = parse_iff()
            if peek() != ')':
                raise SyntaxError(f"Expected ')' at position {pos[0]}, got '{peek()}'")
            consume()
            return f
        if t in ('T', 'true', '⊤'):
            consume()
            return Top
        if t in ('F', 'false', '⊥'):
            consume()
            return Bot
        if t != 'EOF' and t[0].isalpha():
            consume()
            return Var(t)
        raise SyntaxError(f"Unexpected token '{t}' at position {pos[0]}")

    formula = parse_iff()
    if pos[0] != len(tokens):
        raise SyntaxError(
            f"Unexpected tokens after formula: {tokens[pos[0]:]}")
    return formula


def _tokenize(s: str) -> list[str]:
    """Tokenize an infix formula string."""
    import re
    token_re = re.compile(
        r'<->|<=>|↔|->|=>|→|/\\|∧|\\/|∨|&&|&|\|\||'
        r'\||~|!|¬|\(|\)|[A-Za-z][A-Za-z0-9_]*|\S'
    )
    return token_re.findall(s)


# ──────────────────────────────────────────────────────────────────────────────
# Normal-form transformations
# ──────────────────────────────────────────────────────────────────────────────

def eliminate_iff(f: Formula) -> Formula:
    """Replace ↔ and ⊕ with equivalent formulas using ∧, ∨, ¬, →."""
    if isinstance(f, _Constant) or isinstance(f, Var):
        return f
    if isinstance(f, Unary):
        return Not(eliminate_iff(f.child))
    assert isinstance(f, Binary)
    l, r = eliminate_iff(f.left), eliminate_iff(f.right)
    if f.connective == Connective.IFF:
        return And(Implies(l, r), Implies(r, l))
    if f.connective == Connective.XOR:
        return Or(And(l, Not(r)), And(Not(l), r))
    constructors = {
        Connective.AND: And, Connective.OR: Or, Connective.IMP: Implies,
    }
    return constructors[f.connective](l, r)


def to_nnf(f: Formula) -> Formula:
    """
    Convert formula to Negation Normal Form (NNF).

    Post-condition: negations occur only directly before variables.
    """
    f = eliminate_iff(f)
    return _push_negation(f, False)


def _push_negation(f: Formula, negated: bool) -> Formula:
    """Recursively push negations inward (helper for NNF conversion)."""
    if isinstance(f, _Constant):
        if not negated:
            return f
        return Bot if f is Top else Top
    if isinstance(f, Var):
        return Unary(Connective.NOT, f) if negated else f
    if isinstance(f, Unary):
        # Remove double negation by flipping the negated flag
        return _push_negation(f.child, not negated)
    assert isinstance(f, Binary)
    conn = f.connective
    if conn == Connective.IMP:
        # φ → ψ  ≡  ¬φ ∨ ψ
        f = Or(Not(f.left), f.right)
        return _push_negation(f, negated)
    # De Morgan for AND and OR
    if negated:
        if conn == Connective.AND:
            return Or(_push_negation(f.left, True),
                      _push_negation(f.right, True))
        if conn == Connective.OR:
            return And(_push_negation(f.left, True),
                       _push_negation(f.right, True))
    return Binary(conn,
                  _push_negation(f.left, False),
                  _push_negation(f.right, False))


def to_cnf(f: Formula) -> Formula:
    """
    Convert formula to Conjunctive Normal Form (CNF) via Tseitin encoding.

    Instead of the exponential distributive encoding, we use the Tseitin
    transformation (Tseitin, 1968) which introduces auxiliary variables and
    produces an equisatisfiable CNF formula of linear size.
    """
    from .tseitin import TseitinEncoder
    encoder = TseitinEncoder()
    return encoder.encode(f)
