#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import argparse
import os
import tensorflow as tf

from lernd.classes import GroundAtoms, ILP, LanguageModel, MaybeGroundAtom, ProgramTemplate
from lernd.main import main_loop
from lernd.lernd_types import Constant, RuleTemplate
from lernd.util import ground_atom2str, str2ground_atom, str2pred


def print_BPN(ilp_problem: ILP):
    background = ilp_problem.background_axioms
    positive = ilp_problem.positive_examples
    negative = ilp_problem.negative_examples
    print('background_axioms\n', list(map(ground_atom2str, background)), '\n')
    print('positive_examples\n', list(map(ground_atom2str, positive)), '\n')
    print('negative_examples\n', list(map(ground_atom2str, negative)), '\n')


def setup_even():
    print('Setting up "Even" problem...')
    # Language model
    target_pred = str2pred('even/1')
    zero_pred = str2pred('zero/1')
    succ_pred = str2pred('succ/2')
    preds_ext = [zero_pred, succ_pred]
    constants = [Constant(str(i)) for i in range(11)]
    language_model = LanguageModel(target_pred, preds_ext, constants)

    # Program template
    aux_pred = str2pred('pred/2')
    aux_preds = [aux_pred]
    rules = {
        target_pred: (RuleTemplate(0, False), RuleTemplate(1, True)),
        aux_pred: (RuleTemplate(1, False), None)
    }
    forward_chaining_steps = 6
    program_template = ProgramTemplate(aux_preds, rules, forward_chaining_steps)

    # ILP problem
    ground_zero = str2ground_atom('zero(0)')
    background = [ground_zero] + [str2ground_atom(f'succ({i},{i + 1})') for i in range(10)]
    positive = [str2ground_atom(f'even({i})') for i in range(0, 11, 2)]
    negative = [str2ground_atom(f'even({i})') for i in range(1, 10, 2)]
    ilp_problem = ILP('even', language_model, background, positive, negative)
    return ilp_problem, program_template


def setup_predecessor():
    print('Setting up "Predecessor" problem...')

    # Language model
    target_pred = str2pred('predecessor/2')
    zero_pred = str2pred('zero/1')
    succ_pred = str2pred('succ/2')
    preds_ext = [zero_pred, succ_pred]
    constants = [Constant(str(i)) for i in range(10)]
    language_model = LanguageModel(target_pred, preds_ext, constants)

    # Program template
    preds_aux = []
    rules = {target_pred: (RuleTemplate(0, False), None)}
    forward_chaining_steps = 1
    program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)

    # ILP problem
    ground_zero = str2ground_atom('zero(0)')
    background_axioms = [ground_zero] + [str2ground_atom(f'succ({i},{i + 1})') for i in range(9)]
    positive_examples = [str2ground_atom(f'predecessor({i + 1},{i})') for i in range(9)]

    ground_atoms = GroundAtoms(language_model, program_template)
    negative_examples = []
    for ground_atom, _ in ground_atoms.ground_atom_generator(MaybeGroundAtom.from_pred(target_pred)):
        if ground_atom not in positive_examples:
            negative_examples.append(ground_atom)

    ilp_problem = ILP('predecessor', language_model, background_axioms, positive_examples, negative_examples)
    return ilp_problem, program_template


if __name__ == '__main__':
    # Disable Tensorflow logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=str, choices=['predecessor', 'even'], help='Problem to solve')
    args = parser.parse_args()

    with tf.device('/CPU:0'):
        if args.problem == 'predecessor':
            ilp_problem, program_template = setup_predecessor()
            steps = 100
            mini_batch = 1.0  # no mini batching
        elif args.problem == 'even':
            ilp_problem, program_template = setup_even()
            steps = 300
            mini_batch = 0.3  # loss is based on 30% of random given examples
        print_BPN(ilp_problem)
        main_loop(ilp_problem, program_template, steps=steps, mini_batch=mini_batch, plot_loss=True, save_output=True)
