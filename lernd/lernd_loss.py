#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import itertools
from typing import Dict, List, Tuple

import numpy as np
from ordered_set import OrderedSet

import lernd.util as u
from lernd.classes import LanguageModel, ProgramTemplate
from lernd.generator import f_generate
from lernd.inferrer import f_infer
from lernd.types import GroundAtom, Predicate, RuleTemplate


def f_convert(background_axioms: List[GroundAtom], ground_atoms: List[GroundAtom]) -> np.ndarray:
    # non-differentiable operation
    # order must be the same as in ground_atoms
    return np.array([1 if gamma in background_axioms else 0 for gamma in ground_atoms])


def f_extract(valuation: np.ndarray, gamma: GroundAtom, ground_atoms: List[GroundAtom]) -> float:
    # differentiable operation
    # extracts valuation value of a particular atom gamma
    return valuation[ground_atoms.index(gamma)]


def get_ground_atoms(language_model: LanguageModel, program_template: ProgramTemplate) -> List[GroundAtom]:
    preds_ext = language_model.preds_ext
    preds_aux = program_template.preds_aux
    preds = preds_ext + preds_aux + [language_model.target]
    ground_atoms = []
    for pred in preds:
        for constant_combination in itertools.product(language_model.constants, repeat=u.arity(pred)):
            ground_atoms.append(GroundAtom((pred, constant_combination)))
    return ground_atoms


class Lernd:
    def __init__(self, forward_chaining_steps: int, language_model: LanguageModel, program_template: ProgramTemplate,
                 background_axioms: List[GroundAtom]):
        self._forward_chaining_steps = forward_chaining_steps
        self._language_model = language_model
        self._program_template = program_template

        print('Generating clauses...')
        self._clauses = f_generate(program_template, language_model)

        print('Generating ground atoms...')
        self._ground_atoms = get_ground_atoms(language_model, program_template)

        print('Generating initial valuation...')
        self._initial_valuation = f_convert(background_axioms, self._ground_atoms)

    @property
    def forward_chaining_steps(self) -> int:
        return self._forward_chaining_steps

    @property
    def language_model(self) -> LanguageModel:
        return self._language_model

    @property
    def program_template(self) -> ProgramTemplate:
        return self._program_template

    @property
    def clauses(self) -> Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]]:
        return self._clauses

    @property
    def ground_atoms(self) -> List[GroundAtom]:
        return self._ground_atoms

    # loss is cross-entropy
    def loss(self, big_lambda: Dict[GroundAtom, int],
             weights: Dict[Predicate, np.matrix]
             ):
        return - np.mean((
            small_lambda * np.log(self._p(alpha, weights, self._initial_valuation)) +
            (1 - small_lambda) * np.log(1 - self._p(alpha, weights, self._initial_valuation))
            for (alpha, small_lambda) in big_lambda.items()
        ))

    # p(lambda|alpha,W,Pi,L,B) - given a particular atom, weights, program template, language model, and background
    # assumptions gives the probability of label of alpha (which is 0 or 1).
    def _p(self, alpha: GroundAtom, weights: Dict[Predicate, np.matrix], initial_valuation: np.ndarray) -> float:
        valuation = f_infer(
            initial_valuation,
            self._clauses,
            weights,
            self._forward_chaining_steps,
            self._language_model,
            self._ground_atoms
        )
        return f_extract(valuation, alpha, self._ground_atoms)
