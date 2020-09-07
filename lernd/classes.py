#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import itertools
from typing import Dict, Iterable, List, Optional, Tuple, Union

from ordered_set import OrderedSet

from lernd import util as u
from .lernd_types import Atom, Constant, GroundAtom, Predicate, RuleTemplate, Variable


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

    def __repr__(self):
        return f'"{self.__str__()}"'

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

    def to_latex(self) -> str:
        return '${0}\\leftarrow {1}$'.format(u.atom2str(self._head), ', '.join(map(u.atom2str, self._body)))


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
                 preds_aux: List[Predicate],                                           # Pa
                 rules: Dict[Predicate, Tuple[RuleTemplate, Optional[RuleTemplate]]],  # rules
                 forward_chaining_steps: int                                           # T
                 ):
        self._preds_aux = preds_aux
        self._rules = rules
        self._forward_chaining_steps = forward_chaining_steps

    @property
    def preds_aux(self) -> List[Predicate]:
        return self._preds_aux

    @property
    def rules(self) -> Dict[Predicate, Tuple[RuleTemplate, Optional[RuleTemplate]]]:
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


class MaybeGroundAtom:
    def __init__(self, pred: Predicate, args, groundedness: Iterable[bool]):
        self._pred = pred
        self._groundedness = groundedness
        self._len = len(args)
        self._arg0 = (args[0], groundedness[0]) if self._len >= 1 else None
        self._arg1 = (args[1], groundedness[1]) if self._len >= 2 else None
        self._args = [
            self._arg0,
            self._arg1
        ]

    def arg_at(self, index: int) -> Union[Constant, Variable]:
        return self._args[index][0]

    def const_at(self, index: int) -> bool:
        return self._args[index][1]

    def is_ground(self) -> bool:
        return (self._args[0] is None or self._args[0][1]) and (self._args[1] is None or self._args[1][1])

    def to_ground_atom(self) -> GroundAtom:
        if self.is_ground():
            return GroundAtom(self._pred, tuple((self.arg_at(i) for i in range(self._len))))
        else:
            raise Exception()  # TODO: something better

    def apply_substitutions(self, substitutions: Dict[Variable, Constant]):
        for i in range(self._len):
            if not self.const_at(i) and self.arg_at(i) in substitutions:
                self._args[i] = (substitutions[self.arg_at(i)], True)

    @property
    def pred(self) -> Predicate:
        return self._pred

    @property
    def arity(self) -> int:
        return u.arity(self._pred)

    @classmethod
    def from_ground_atom(cls, ground_atom: GroundAtom):
        pred, args = ground_atom
        groundedness = [True] * u.arity(pred)
        return cls(pred, args, groundedness)

    @classmethod
    def from_atom(cls, atom: Atom):
        pred, args = atom
        groundedness = [False] * u.arity(pred)
        return cls(pred, args, groundedness)

    @classmethod
    def from_pred(cls, pred: Predicate):
        args = [Variable(f'tmp{i}') for i in range(u.arity(pred))]
        groundedness = [False] * u.arity(pred)
        return cls(pred, args, groundedness)

    def __str__(self):
        string = ''
        string += self._pred[0] + '('
        for i in range(self.arity):
            string += self._args[i][0]
        string += ')'
        return string

    def copy(self):
        return type(self)(self._pred, [self.arg_at(i) for i in range(self._len)], [self.const_at(i) for i in range(self._len)])


class GroundAtoms:
    def __init__(self, language_model: LanguageModel, program_template: ProgramTemplate):
        self._constants = OrderedSet(language_model.constants)
        self._number_of_constants = len(language_model.constants)
        self._preds = language_model.preds_ext + program_template.preds_aux + [language_model.target]
        preds = self._preds

        # First element (index 0) is falsum
        # key: predicate,
        # value: index of predicate's first ground atom (amongst all ground atoms)
        self._ground_atom_base_index = {preds[0]: 1}

        for i in range(1, len(preds)):
            prev_pred = preds[i - 1]
            pred = preds[i]
            self._ground_atom_base_index[pred] = self._ground_atom_base_index[prev_pred] + len(self._constants) ** u.arity(prev_pred)
        self._len = self._ground_atom_base_index[preds[-1]] + len(self._constants) ** u.arity(preds[-1])

    @property
    def len(self):
        return self._len

    def ground_atom_generator(self, maybe_ground_atom: MaybeGroundAtom) -> Iterable[Tuple[GroundAtom, Dict[Variable, Constant]]]:
        # TODO: maybe doesn't need to return ground_atoms at all?
        if maybe_ground_atom.is_ground():
            return [(maybe_ground_atom.to_ground_atom(), {})]
        pred = maybe_ground_atom.pred
        arity = maybe_ground_atom.arity
        if arity == 1:
            return ((GroundAtom(pred, (c,)), {maybe_ground_atom.arg_at(0): c}) for c in self._constants)
        elif arity == 2:
            if maybe_ground_atom.const_at(0):
                return ((GroundAtom(pred, (maybe_ground_atom.arg_at(0), c)), {maybe_ground_atom.arg_at(1): c}) for c in self._constants)
            elif maybe_ground_atom.const_at(1):
                return ((GroundAtom(pred, (c, maybe_ground_atom.arg_at(1))), {maybe_ground_atom.arg_at(0): c}) for c in self._constants)
            else:
                if not maybe_ground_atom.const_at(0) and not maybe_ground_atom.const_at(1) and maybe_ground_atom.arg_at(0) == maybe_ground_atom.arg_at(1):
                    return ((GroundAtom(pred, (c, c)), {maybe_ground_atom.arg_at(0): c}) for c in self._constants)
                else:
                    return ((GroundAtom(pred, (c1, c2)), {maybe_ground_atom.arg_at(0): c1, maybe_ground_atom.arg_at(1): c2}) for c1, c2 in itertools.product(self._constants, repeat=arity))
        else:
            raise Exception()  # TODO: something better

    def all_ground_atom_generator(self) -> Iterable[GroundAtom]:
        for pred in self._preds:
            arity = u.arity(pred)
            if arity == 0:
                yield GroundAtom(pred, ())
            elif arity == 1:
                for c in self._constants:
                    yield GroundAtom(pred, (c,))
            elif arity == 2:
                for c1, c2 in itertools.product(self._constants, repeat=2):
                    yield GroundAtom(pred, (c1, c2))

    def get_ground_atom_index(self, ground_atom: GroundAtom) -> int:
        pred, consts = ground_atom
        if u.arity(pred) == 0:
            return self._ground_atom_base_index[pred]
        elif u.arity(pred) == 1:
            return self._ground_atom_base_index[pred] + self._constants.map[consts[0]]
        elif u.arity(pred) == 2:
            return self._ground_atom_base_index[pred] + self._constants.map[consts[0]] * self._number_of_constants + self._constants.map[consts[1]]
        else:
            raise Exception()  # TODO: something better
