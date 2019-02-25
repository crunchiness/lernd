#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from typing import Tuple, List, Dict

from lernd import util as u
from .types import Atom, Constant, GroundAtom, Predicate, RuleTemplate


class Clause:
    def __init__(self, head: Atom, body: Tuple[Atom, ...]):
        self._head = head
        self._body = body

    def __eq__(self, other):
        return self.head == other.head and self.body == other.body

    def __hash__(self):
        return hash(self.head) ^ hash(self.body)

    def __str__(self):
        return '{0}<-{1}'.format(u.atom2str(self._head), ', '.join(map(u.atom2str, self._body)))

    @property
    def head(self) -> Atom:
        return self._head

    @property
    def body(self) -> Tuple[Atom, ...]:
        return self._body

    @classmethod
    def from_str(cls, s: str):
        clause_list = s.split('<-')
        head_str = clause_list[0]
        body_strs = clause_list[1].split(', ')
        head = u.str2atom(head_str)
        body = tuple(map(u.str2atom, body_strs))
        return cls(head, body)


class LanguageModel:
    def __init__(self, target: Predicate, preds_ext: List[Predicate], constants: List[Constant]):
        self._target = target
        self._preds_ext = preds_ext
        self._constants = constants

    @property
    def target(self) -> Predicate:
        return self._target

    @property
    def preds_ext(self) -> List[Predicate]:
        return self._preds_ext

    @property
    def constants(self) -> List[Constant]:
        return self._constants


class ProgramTemplate:
    def __init__(self,
                 preds_aux: List[Predicate],                                 # Pa
                 rules: Dict[Predicate, Tuple[RuleTemplate, RuleTemplate]],  # rules
                 forward_chaining_steps: int                                 # T
                 ):
        self._preds_aux = preds_aux
        self._rules = rules
        self._forward_chaining_steps = forward_chaining_steps

    @property
    def preds_aux(self) -> List[Predicate]:
        return self._preds_aux

    @property
    def rules(self) -> Dict[Predicate, Tuple[RuleTemplate, RuleTemplate]]:
        return self._rules

    @property
    def forward_chaining_steps(self):
        return self._forward_chaining_steps


class ILP:
    def __init__(self,
                 language_model: LanguageModel,        # L
                 background_axioms: List[GroundAtom],  # B
                 positive_examples: List[GroundAtom],  # P
                 negative_examples: List[GroundAtom]   # N
                 ):
        self._language_model = language_model
        self._background_axioms = background_axioms
        self._positive_examples = positive_examples
        self._negative_examples = negative_examples

    @property
    def language_model(self) -> LanguageModel:
        return self._language_model

    @property
    def background_axioms(self) -> List[GroundAtom]:
        return self._background_axioms

    @property
    def positive_examples(self) -> List[GroundAtom]:
        return self._positive_examples

    @property
    def negative_examples(self) -> List[GroundAtom]:
        return self._negative_examples
