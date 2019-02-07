#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from typing import NewType, Tuple

RuleTemplate = NewType('RuleTemplate', Tuple[int, bool])  # (v, int)
Predicate = NewType('Predicate', Tuple[str, int])  # (name, arity)
Variable = NewType('Variable', str)
Constant = NewType('Constant', str)
Atom = NewType('Atom', Tuple[Predicate, Tuple[Variable, ...]])
GroundAtom = NewType('GroundAtom', Tuple[Predicate, Tuple[Constant, ...]])
