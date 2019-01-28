#!/usr/bin/env python3

from typing import Tuple, NewType, List

RuleTemplate = NewType('RuleTemplate', Tuple[int, bool])  # (v, int)
Predicate = NewType('Predicate', Tuple[str, int])  # (name, arity)
Variable = NewType('Variable', str)
FullPredicate = NewType('PredicateWithArgs', Tuple[Predicate, List[Variable]])
