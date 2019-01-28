#!/usr/bin/env python3

import string
from itertools import product
from typing import List, Set, Iterable

import numpy as np

from .types import FullPredicate, Predicate, RuleTemplate, Variable
from .util import fpred2str


class Clause:
    def __init__(self, head: FullPredicate, body: List[FullPredicate]):
        self.head = head
        self.body = body

    def __str__(self):
        return '{0}->{1}'.format(fpred2str(self.head), ','.join(map(fpred2str, self.body)))


G = []  # All ground atoms

# ILP problem
# Language model
target = Predicate(('q', 2))
P_e = set([])  # Set of extensional predicates
arity_e = None
C = set([])  # Set of constants
L = (target, P_e, arity_e, C)  # Language model

B = None  # Background assumptions
P = []  # Positive examples
N = []  # Negative examples

ILP = (L, B, P, N)  # ILP problem

# Program template
P_a = None
arity_a = None
rules = None  # Dict (predicate p: tuple of rule templates (tau1, tau2))
T = None
Pi = (P_a, arity_a, rules, T)  # Program template

W = None  # set of clause weights

Lambda = [(gamma, 1) for gamma in P] + [(gamma, 0) for gamma in N]


# p(lambda|alpha,W,Pi,L,B) - p1 - given particular atom, weights, program template, language model, and background
# assumptions gives the probability of label of alpha (which is 0 or 1).
def f_extract(valuation, gamma):
    # differentiable operation
    # extracts valuation value of a particular atom gamma
    pass


def f_infer(param, param1, W, T):
    # differentiable operation
    pass


def f_convert(B):
    # non-differentiable operation
    # creates a valuation mapping B to 1 and other to 0
    return [1 if gamma in B else 0 for gamma in G]


def cl(Pi, L, p: Predicate, tau: RuleTemplate) -> Set[Clause]:
    # L = (target, P_e, arity_e, C)  # Language model
    """
    Restrictions:
    1. No constants in any of the clauses.
    2. Only clauses of atoms involving free variables.
    3. Only predicates of arity 0-2.
    4. Exactly 2 atoms in the body.

    5. No unsafe (which have a variable used in the head but not the body)
    6. No circular (head atom appears in the body)
    7. No duplicate (same but different order of body atoms)
    8. No those that contain an intensional predicate in the clause body, even though int flag was set to 0, false.
    """
    # set of clauses that satisfy the rule template
    v, int_ = tau  # number of exist. quantified variables allowed, whether intensional predicates allowed in the body
    target = L[0]  # type: Predicate
    P_e = L[1]  # type: Set[Predicate]

    target_arity = target[1]
    total_vars = target_arity + v

    assert total_vars <= len(string.ascii_uppercase), 'Handling of more than 26 variables not implemented!'

    vars = [Variable(string.ascii_uppercase[i]) for i in range(total_vars)]
    head = FullPredicate((target, [vars[i] for i in range(target_arity)]))

    # TODO: now do all combinations and check all the rules...

    possible_preds = list(P_e) + [target] if int_ else P_e  # TODO: creating auxiliary predicates

    clauses = set([])
    for pred1, pred2 in product(possible_preds, possible_preds):
        for pred1_full in pred_with_vars_generator(pred1, vars):
            for pred2_full in pred_with_vars_generator(pred2, vars):
                clauses.add(Clause(head, [pred1_full, pred2_full]))
                # print(Clause(head, [pred1_full, pred2_full]))
                # print(predicate_to_str(pred1_full) + ', ' + predicate_to_str(pred2_full))
    return clauses


def generate_predicate(possible_preds: List[Predicate], vars: List[Variable]):
    for pred in possible_preds:
        for pred_full in pred_with_vars_generator(pred, vars):
            print(fpred2str(pred_full))


def generate_2_predicates(possible_preds: list, vars: list):
    for pred1, pred2 in product(possible_preds, possible_preds):
        for pred1_full, pred2_full in product(pred_with_vars_generator(pred1, vars), pred_with_vars_generator(pred2, vars)):
            print(fpred2str(pred1_full) + ', ' + fpred2str(pred2_full))


def pred_with_vars_generator(predicate: Predicate, vars: List[Variable]) -> Iterable[FullPredicate]:
    for combination in product(vars, repeat=predicate[1]):
        yield FullPredicate((predicate, list(combination)))


p = ['a/2', 'b/1', 'c/0', 'd/2']
vars = ['X', 'Y', 'Z']
# generate_predicate(list(map(predicate_from_str, p)), list(map(lambda x: Variable(x), vars)))
# generate_2_predicates(list(map(str2pred, p)), list(map(lambda x: Variable(x), vars)))


def arity(p: Predicate) -> int:
    return p[1]


def f_generate(Pi, L):
    # non-differentiable operation
    # returns a set of clauses
    target = L[0]
    P_a = Pi[0]
    P_i = P_a + [target]
    rules = Pi[2]
    clauses = []
    for p in P_i:
        tau1, tau2 = rules[p]
        clauses.append(cl(Pi, L, p, tau1))
        clauses.append(cl(Pi, L, p, tau2))
    return clauses


def p1(lambda_, alpha, W, Pi, L, B):
    return f_extract(f_infer(f_convert(B), f_generate(Pi, L), W, T), alpha)


# loss is cross-entropy
def loss(Lambda, W, Pi, L, B):
    return - np.mean([
        lambda_ * np.log(p1(lambda_, alpha, W, Pi, L, B)) + (1 - lambda_) * np.log(1 - p1(lambda_, alpha, W, Pi, L, B))
        for (alpha, lambda_) in Lambda
    ])
