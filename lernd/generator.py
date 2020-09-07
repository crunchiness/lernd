#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import string
from functools import reduce
from itertools import product
from operator import add
from typing import Dict, Iterable, List, Tuple, Optional

from ordered_set import OrderedSet

from .classes import Clause, LanguageModel, ProgramTemplate
from .lernd_types import Atom, Predicate, RuleTemplate, Variable


def f_generate(
        program_template: ProgramTemplate,
        language_model: LanguageModel
) -> Dict[Predicate, Tuple[Tuple[OrderedSet[Clause], RuleTemplate], Tuple[OrderedSet[Clause], RuleTemplate]]]:
    # non-differentiable operation
    preds_int = program_template.preds_aux + [language_model.target]  # type: List[Predicate]
    clauses = {}
    for pred in preds_int:
        tau1, tau2 = program_template.rules[pred]
        clauses[pred] = (
            (cl(preds_int, language_model.preds_ext, pred, tau1), tau1),
            (cl(preds_int, language_model.preds_ext, pred, tau2), tau2)
        )
    return clauses


def cl(
        preds_int: List[Predicate],
        preds_ext: List[Predicate],
        pred: Predicate,
        tau: Optional[RuleTemplate]
) -> OrderedSet[Clause]:
    """Generates all possible clauses adhering the restrictions.
    Restrictions:
    1. Only clauses of atoms involving free variables. No constants in any of the clauses.
    2. Only predicates of arity 0-2.
    3. Exactly 2 atoms in the body.
    4. No unsafe (which have a variable used in the head but not the body)
    5. No circular (head atom appears in the body)
    6. No duplicate (same but different order of body atoms)
    7. No those that contain an intensional predicate in the clause body, even though int flag was set to 0, false.
    """
    if tau is None:
        return OrderedSet()

    v, int_ = tau  # number of exist. quantified variables allowed, whether intensional predicates allowed in the body

    pred_arity = pred[1]
    total_vars = pred_arity + v

    assert total_vars <= len(string.ascii_uppercase), 'Handling of more than 26 variables not implemented!'

    variables = [Variable(string.ascii_uppercase[i]) for i in range(total_vars)]
    head = Atom(pred, tuple([variables[i] for i in range(pred_arity)]))

    possible_preds = list(preds_ext) + preds_int if int_ else list(preds_ext)  # type: List[Predicate]

    clauses = OrderedSet()
    for pred1, pred2 in product(possible_preds, possible_preds):
        for pred1_full in pred_with_vars_generator(pred1, variables):
            for pred2_full in pred_with_vars_generator(pred2, variables):
                clause = Clause(head, tuple(sorted([pred1_full, pred2_full])))
                if check_clause_unsafe(clause):
                    continue
                if check_circular(clause):
                    continue
                if not check_int_flag_satisfied(clause, int_, preds_int):
                    continue
                clauses.add(clause)
    return clauses


def pred_with_vars_generator(predicate: Predicate, variables: List[Variable]) -> Iterable[Atom]:
    for combination in product(variables, repeat=predicate[1]):
        yield Atom(predicate, tuple(combination))


def check_clause_unsafe(clause: Clause) -> bool:
    """
    Returns True if clause is unsafe (has a variable used in the head but not the body)
    """
    head_vars = clause.head[1]  # type: List[Variable]
    preds = clause.body  # type: Tuple[Atom, ...]
    preds_list = list(preds)
    body_vars = reduce(add, map(lambda x: list(x[1]), preds_list))
    for head_var in head_vars:
        if head_var not in body_vars:
            return True
    return False


def check_circular(clause: Clause) -> bool:
    """
    Returns True if the clause is circular (head atom appears in the body)
    """
    head = clause.head
    atoms = clause.body
    if head in atoms:
        return True
    return False


def check_int_flag_satisfied(clause: Clause, int_: bool, preds_int: List[Predicate]) -> bool:
    # if intensional predicate required:
    if int_:
        for atom in clause.body:
            if atom[0] in preds_int:
                return True
        return False
    # otherwise intensional predicates not in possible_preds
    return True
