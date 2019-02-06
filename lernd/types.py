#!/usr/bin/env python3

from typing import Tuple, NewType

RuleTemplate = NewType('RuleTemplate', Tuple[int, bool])  # (v, int)
Predicate = NewType('Predicate', Tuple[str, int])  # (name, arity)
Variable = NewType('Variable', str)
Atom = NewType('Atom', Tuple[Predicate, Tuple[Variable, ...]])
