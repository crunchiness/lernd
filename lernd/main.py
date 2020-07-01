#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import typing
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet

from .classes import Clause, ILP, ProgramTemplate
from .lernd_loss import Lernd
from .lernd_types import Predicate, RuleTemplate, GroundAtom
from .util import ground_atom2str, get_ground_atom_probs


def generate_weight_matrices(
        clauses: Dict[
            Predicate,
            Tuple[Tuple[OrderedSet[Clause], RuleTemplate], Tuple[OrderedSet[Clause], RuleTemplate]]
        ],
        stddev: float = 0.05
) -> typing.OrderedDict[Predicate, tf.Variable]:
    rule_weights = OrderedDict()
    initializer = tf.random_normal_initializer(mean=0, stddev=stddev)
    for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
        rule_weights[pred] = tf.Variable(initializer(shape=[len(clauses_1), len(clauses_2) or 1]),
                                         trainable=True,
                                         dtype=tf.float32)
    return rule_weights


def extract_definitions(
        clauses: Dict[
            Predicate,
            Tuple[Tuple[OrderedSet[Clause], RuleTemplate], Tuple[OrderedSet[Clause], RuleTemplate]]
        ],
        weights: typing.OrderedDict[Predicate, tf.Variable],
        clause_prob_threshold: float = 0.1
):
    for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
        shape = weights[pred].shape
        pred_weights = tf.reshape(weights[pred], [-1])
        pred_probs_flat = tf.nn.softmax(pred_weights)
        max_value = np.max(pred_probs_flat)
        clause_prob_threshold = min(max_value, clause_prob_threshold)
        pred_probs = tf.reshape(pred_probs_flat[:, np.newaxis], shape)
        print(f'clause_prob_threshold: {clause_prob_threshold}\n')  # debug
        print('Clause learnt:')
        indices = np.nonzero(pred_probs >= clause_prob_threshold)
        if tau2 is not None:
            for index_tuple in zip(indices[0], indices[1]):
                print(f'With probability (confidence): {pred_probs[index_tuple]}')
                print(clauses_1[index_tuple[0]])
                print(clauses_2[index_tuple[1]])
        else:
            for index in indices[0]:
                print(f'With probability (confidence): {pred_probs[index][0]}')
                print(clauses_1[index])
        print()


def print_valuations(ground_atom_probs: typing.OrderedDict[GroundAtom, float], threshold: float = 0.01):
    print(f'Valuations of ground atoms (only those >{threshold} for readability):')
    for ground_atom, p in ground_atom_probs.items():
        if p > threshold:
            print(f'{ground_atom2str(ground_atom)} - {p}')


def main_loop(
        ilp_problem: ILP,
        program_template: ProgramTemplate,
        learning_rate: float = 0.5,
        steps: int = 6000,
        mini_batch: float = 1.0,
        weight_stddev: float = 0.05,
        clause_prob_threshold: float = 0.1
):
    lernd_model = Lernd(ilp_problem, program_template, mini_batch=mini_batch)

    print('Generating weight matrices...')
    weights = generate_weight_matrices(lernd_model.clauses, stddev=weight_stddev)

    losses = []
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    for i in range(1, steps + 1):
        loss_grad, loss, valuation = lernd_model.grad(weights)
        optimizer.apply_gradients(zip(loss_grad, list(weights.values())))
        loss_float = float(loss.numpy())
        losses.append(loss_float)
        if i % 10 == 0:
            print(f'Step {i} loss: {loss_float}\n')
        if i == steps:
            extract_definitions(lernd_model.clauses, weights, clause_prob_threshold=clause_prob_threshold)
            ground_atom_probs = get_ground_atom_probs(valuation, lernd_model.ground_atoms)
            print_valuations(ground_atom_probs)
