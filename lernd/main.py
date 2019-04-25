#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from typing import Dict, List, Tuple

import numpy as np
from ordered_set import OrderedSet

from lernd.classes import Clause
from lernd.lernd_loss import Lernd
from .classes import ILP, LanguageModel, ProgramTemplate
from .types import GroundAtom, Predicate, RuleTemplate


def make_lambda(positive_examples: List[GroundAtom], negative_examples: List[GroundAtom]) -> Dict[GroundAtom, int]:
    result = {}
    for ground_atom in positive_examples:
        result[ground_atom] = 1
    for ground_atom in negative_examples:
        result[ground_atom] = 0
    return result


def generate_weight_matrices(
        clauses: Dict[Predicate, Tuple[Tuple['OrderedSet[Clause]', RuleTemplate], Tuple['OrderedSet[Clause]', RuleTemplate]]],
        standard_deviation: float,
        mean: float = 0
        ) -> Dict[Predicate, np.matrix]:
    weights_dict = {}
    for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
        weights_dict[pred] = np.matrix(np.random.normal(mean, standard_deviation, (len(clauses_1), len(clauses_2))))

    return weights_dict


if __name__ == '__main__':
    target = Predicate(('q', 2))
    preds_ext = []  # Set of extensional predicates
    constants = []  # Set of constants
    language_model = LanguageModel(target, preds_ext, constants)
    background_axioms = []  # Background assumptions
    positive_examples = []  # Positive examples
    negative_examples = []  # Negative examples
    ilp_problem = ILP(language_model, background_axioms, positive_examples, negative_examples)

    preds_aux = []

    # Dict (predicate p: tuple of rule templates (tau1, tau2))
    rules = {Predicate(('q', 2)): (RuleTemplate((0, False)), RuleTemplate((1, True)))}
    forward_chaining_steps = 0
    program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)

    big_lambda = make_lambda(positive_examples, negative_examples)

    lernd_model = Lernd(forward_chaining_steps, language_model, program_template)

    print('Generating weight matrices')
    weights = generate_weight_matrices(lernd_model.clauses, standard_deviation=0.5)  # type: Dict[Predicate, np.matrix]
