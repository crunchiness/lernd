#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from typing import NamedTuple, Tuple


class Constant(NamedTuple):
    name: str


class Predicate(NamedTuple):
    name: str
    arity: int


class RuleTemplate(NamedTuple):
    v: int  # number of exist. quantified vars allowed in the clause
    int: bool  # whether intensional predicates are allowed


class Variable(NamedTuple):
    name: str


class Atom(NamedTuple):
    pred: Predicate
    vars: Tuple[Variable, ...]


class GroundAtom(NamedTuple):
    pred: Predicate
    consts: Tuple[Constant, ...]
