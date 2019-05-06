#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from typing import Dict, List, Tuple

import numpy as np
from ordered_set import OrderedSet

from lernd.classes import Clause
from lernd.lernd_loss import Lernd
from .classes import ILP, LanguageModel, ProgramTemplate
from .types import Constant, GroundAtom, Predicate, RuleTemplate


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


def main_loop(
        ilp_problem: ILP,
        program_template: ProgramTemplate
):
    lernd_model = Lernd(ilp_problem, program_template)

    print('Generating weight matrices...')
    weights = generate_weight_matrices(lernd_model.clauses, standard_deviation=0.5)  # type: Dict[Predicate, np.matrix]

    print('Making big lambda...')
    big_lambda = make_lambda(ilp_problem.positive_examples, ilp_problem.negative_examples)

    # TODO: put in loop
    loss, loss_grad = lernd_model.loss_and_grad(big_lambda, weights)
