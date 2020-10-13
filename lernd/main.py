#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import datetime
import json
import os
import pickle
import typing
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from ordered_set import OrderedSet

from .classes import Clause, ILP, ProgramTemplate
from .lernd_loss import Lernd
from .lernd_types import Predicate, RuleTemplate, GroundAtom
from .util import ground_atom2str, get_ground_atom_probs


def output_to_files(
        task_id: str,
        definitions: str,
        ground_atoms: str,
        losses: typing.List[float],
        weights: typing.OrderedDict[Predicate, tf.Variable],
        folder: str = ''
):
    with open(os.path.join(folder, f'{task_id}_definitions.json'), 'w') as f:
        f.write(definitions)
    with open(os.path.join(folder, f'{task_id}_ground_atoms.txt'), 'w') as f:
        f.write(ground_atoms)
    with open(os.path.join(folder, f'{task_id}_losses.txt'), 'w') as f:
        [f.write(str(loss) + '\n') for loss in losses]
    pickle.dump(weights, open(os.path.join(folder, f'{task_id}_weights.pickle'), 'wb'))


def timestamp_str():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')


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
) -> str:
    """Extracts definitions of target and auxiliary predicates from weights.
    :return JSON string
    """
    output = []
    for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
        shape = weights[pred].shape
        pred_weights = tf.reshape(weights[pred], [-1])
        pred_probs_flat = tf.nn.softmax(pred_weights)
        max_value = np.max(pred_probs_flat)
        clause_prob_threshold = min(max_value, clause_prob_threshold)
        pred_probs = tf.reshape(pred_probs_flat[:, np.newaxis], shape)
        item = {'clause_prob_threshold': clause_prob_threshold}
        indices = np.nonzero(pred_probs >= clause_prob_threshold)
        if tau2 is not None:
            for index_tuple in zip(indices[0], indices[1]):
                item['confidence'] = pred_probs[index_tuple].numpy().astype(float)
                item['definition'] = [str(clauses_1[index_tuple[0]]), str(clauses_2[index_tuple[1]])]
        else:
            for index in indices[0]:
                item['confidence'] = pred_probs[index][0].numpy().astype(float)
                item['definition'] = [str(clauses_1[index])]
        output.append(item)
    return json.dumps(output, indent=4)


def get_valuations(ground_atom_probs: typing.OrderedDict[GroundAtom, float], threshold: float = 0.01) -> str:
    output = f'Valuations of ground atoms (only those >{threshold} for readability):\n'
    for ground_atom, p in ground_atom_probs.items():
        if p > threshold:
            output += f'{ground_atom2str(ground_atom)} - {p}\n'
    return output


def main_loop(
        ilp_problem: ILP,
        program_template: ProgramTemplate,
        learning_rate: float = 0.5,
        steps: int = 6000,
        mini_batch: float = 1.0,
        weight_stddev: float = 0.05,
        clause_prob_threshold: float = 0.1,
        plot_loss: bool = False,
        save_output: bool = False
):
    task_id = f'{ilp_problem.name}_{timestamp_str()}'

    # mini-batch flag
    mb = mini_batch < 1.0

    lernd_model = Lernd(ilp_problem, program_template, mini_batch=mini_batch)

    print('Generating weight matrices...')
    weights = generate_weight_matrices(lernd_model.clauses, stddev=weight_stddev)

    losses = []
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    for i in range(1, steps + 1):
        loss_grad, loss, valuation, full_loss = lernd_model.grad(weights)
        optimizer.apply_gradients(zip(loss_grad, list(weights.values())))
        full_loss_float = None if full_loss is None else float(full_loss.numpy())
        loss_float = float(loss.numpy())

        # Loss for plotting
        if mb and full_loss is not None:
            losses.append(full_loss_float)
        elif not mb:
            losses.append(loss_float)

        # Printing info
        if i % 10 == 0:
            if mb and full_loss is not None:
                print(f'Step {i} loss: {full_loss_float}, mini_batch loss: {loss_float}\n')
            elif mb and full_loss is None:
                print(f'Step {i} mini_batch loss: {loss_float}\n')
            elif not mb:
                print(f'Step {i} loss: {loss_float}\n')

        if i == steps:
            definitions = extract_definitions(lernd_model.clauses, weights, clause_prob_threshold=clause_prob_threshold)
            print('Definitions:', definitions)
            ground_atom_probs = get_ground_atom_probs(valuation, lernd_model.ground_atoms)
            ground_atom_probs_str = get_valuations(ground_atom_probs)
            print(ground_atom_probs_str)
            if save_output:
                output_to_files(task_id, definitions, ground_atom_probs_str, losses, weights)

    if (plot_loss or save_output) and len(losses) > 0:
        fig, ax = plt.subplots()
        ax.plot(losses)
        ax.set_title('Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Value')
        if save_output:
            plt.savefig(task_id + '.png')
        if plot_loss:
            plt.show()
