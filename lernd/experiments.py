#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from itertools import product
from lernd.classes import ILP, LanguageModel, ProgramTemplate
from lernd.main import main_loop
from lernd.types import Constant, GroundAtom, Predicate, RuleTemplate
from lernd.util import ground_atom2str


def empty():
    # Language model
    target_pred = Predicate(('q', 2))
    preds_ext = []  # Set of extensional predicates
    constants = []  # Set of constants
    language_model = LanguageModel(target_pred, preds_ext, constants)

    # ILP problem
    background_axioms = []  # Background assumptions
    positive_examples = []  # Positive examples
    negative_examples = []  # Negative examples
    ilp_problem = ILP(language_model, background_axioms, positive_examples, negative_examples)

    # Program template
    preds_aux = []
    # Dict (predicate p: tuple of rule templates (tau1, tau2))
    rules = {Predicate(('q', 2)): (RuleTemplate((0, False)), RuleTemplate((1, True)))}
    forward_chaining_steps = 0
    program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)

    main_loop(ilp_problem, program_template)


def predecessor(do_print=False):
    # Language model
    target_pred = Predicate(('target', 2))
    zero_pred = Predicate(('zero', 1))
    succ_pred = Predicate(('succ', 2))
    preds_ext = [zero_pred, succ_pred]
    constants = [Constant(str(i)) for i in range(0, 10)]
    language_model = LanguageModel(target_pred, preds_ext, constants)

    # ILP problem
    ground_zero = GroundAtom((zero_pred, (constants[0],)))
    background_axioms = [ground_zero] + list(map(lambda i: GroundAtom((succ_pred, (constants[i - 1], constants[i]))), range(1, 10)))
    positive_examples = list(map(lambda i: GroundAtom((target_pred, (constants[i], constants[i - 1]))), range(1, 10)))
    negative_examples = []  # Negative examples
    for const1, const2 in product(constants, constants):
        ground_atom = GroundAtom((target_pred, (const1, const2)))
        if ground_atom not in positive_examples:
            negative_examples.append(ground_atom)
    if do_print:
        print('background_axioms\n', list(map(ground_atom2str, background_axioms)), '\n')
        print('positive_examples\n', list(map(ground_atom2str, positive_examples)), '\n')
        print('negative_examples\n', list(map(ground_atom2str, negative_examples)), '\n')
    ilp_problem = ILP(language_model, background_axioms, positive_examples, negative_examples)

    # Program template
    preds_aux = []
    # rules = {}  # TODO: ?
    rules = {target_pred: (RuleTemplate((0, False)), RuleTemplate((1, True)))}
    forward_chaining_steps = 10  # TODO: ??
    program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)

    main_loop(ilp_problem, program_template)

    # TODO: finish


# empty()
predecessor()
