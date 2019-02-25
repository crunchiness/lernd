#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import itertools
from typing import List, Dict

import numpy as np

import lernd.util as u
from .classes import ILP, LanguageModel, ProgramTemplate
from .generator import f_generate
from .inferrer import f_infer
from .types import GroundAtom, Predicate

target = Predicate(('q', 2))
preds_ext = []  # Set of extensional predicates
constants = []  # Set of constants
language_model = LanguageModel(target, preds_ext, constants)
background_axioms = []  # Background assumptions
positive_examples = []  # Positive examples
negative_examples = []  # Negative examples
ilp_problem = ILP(language_model, background_axioms, positive_examples, negative_examples)

preds_aux = []
rules = {}  # Dict (predicate p: tuple of rule templates (tau1, tau2))
forward_chaining_steps = 0
program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)

weights = {}  # type: Dict[Predicate, np.matrix] # set of clause weights


def make_lambda(positive_examples: List[GroundAtom], negative_examples: List[GroundAtom]) -> Dict[GroundAtom, int]:
    result = {}
    for ground_atom in positive_examples:
        result[ground_atom] = 1
    for ground_atom in negative_examples:
        result[ground_atom] = 0
    return result


big_lambda = make_lambda(positive_examples, negative_examples)


def get_ground_atoms(l: LanguageModel, pi: ProgramTemplate) -> List[GroundAtom]:
    preds_ext = l.preds_ext
    preds_aux = pi.preds_aux
    preds = preds_ext + preds_aux + [l.target]
    ground_atoms = []  # type: List[GroundAtom]
    for pred in preds:
        for constant_combination in itertools.product(l.constants, repeat=u.arity(pred)):
            ground_atoms.append(GroundAtom((pred, constant_combination)))
    return ground_atoms


print('Generating ground atoms...')
ground_atoms = get_ground_atoms(language_model, program_template)


# loss is cross-entropy
def loss(big_lambda: Dict[GroundAtom, int],
         weights: Dict[Predicate, np.matrix],
         l: LanguageModel,
         background_axioms: List[GroundAtom],
         ground_atoms: List[GroundAtom],
         program_template: ProgramTemplate
         ):
    return - np.mean((
        small_lambda * np.log(p1(alpha, weights, l, background_axioms, ground_atoms, program_template)) +
        (1 - small_lambda) * np.log(1 - p1(alpha, weights, l, background_axioms, ground_atoms, program_template))
        for (alpha, small_lambda) in big_lambda.items()
    ))


# p(lambda|alpha,W,Pi,L,B) - p1 - given particular atom, weights, program template, language model, and background
# assumptions gives the probability of label of alpha (which is 0 or 1).
def p1(alpha: GroundAtom,
       weights: Dict[Predicate, np.matrix],
       language_model: LanguageModel,
       background_axioms: List[GroundAtom],
       ground_atoms: List[GroundAtom],
       program_template: ProgramTemplate
       ) -> float:
    return f_extract(f_infer(f_convert(background_axioms, ground_atoms), f_generate(program_template, language_model), weights, forward_chaining_steps, language_model, ground_atoms), alpha, ground_atoms)


def f_extract(valuation: np.ndarray, gamma: GroundAtom, ground_atoms: List[GroundAtom]) -> float:
    # differentiable operation
    # extracts valuation value of a particular atom gamma
    return valuation[ground_atoms.index(gamma)]


def f_convert(background_axioms: List[GroundAtom], ground_atoms: List[GroundAtom]) -> np.ndarray:
    # non-differentiable operation
    # order must be the same as in ground_atoms
    return np.array([1 if gamma in background_axioms else 0 for gamma in ground_atoms])


# def generate_weights(clauses: Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]]
#                      ) -> Dict[Predicate, np.matrix]:
#     weights_dict = {}
#     for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
#         weights_dict[pred] = np.matrix(np.zeros(shape=(len(clauses_1), len(clauses_2))))  # TODO: initialize randomly
#     return weights_dict
#
#
# print('Initializing weights...')
# weights = generate_weights(clauses)
