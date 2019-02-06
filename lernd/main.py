#!/usr/bin/env python3

import string
from functools import reduce
from itertools import product
from operator import add
from typing import List, Set, Iterable, Tuple

import numpy as np
from ordered_set import OrderedSet

import lernd.util as u
from .types import Atom, Predicate, RuleTemplate, Variable


class Clause:
    def __init__(self, head: Atom, body: Tuple[Atom, ...]):
        self._head = head
        self._body = body

    def __eq__(self, other):
        return self.head == other.head and self.body == other.body

    def __hash__(self):
        return hash(self.head) ^ hash(self.body)

    def __str__(self):
        return '{0}<-{1}'.format(u.atom2str(self._head), ', '.join(map(u.atom2str, self._body)))

    @property
    def head(self):
        return self._head

    @property
    def body(self):
        return self._body

    @classmethod
    def from_str(cls, s: str):
        clause_list = s.split('<-')
        head_str = clause_list[0]
        body_strs = clause_list[1].split(', ')
        head = u.str2atom(head_str)
        body = tuple(map(u.str2atom, body_strs))
        return cls(head, body)


G = []  # All ground atoms

# ILP problem
# Language model
target = Predicate(('q', 2))
P_e = set([])  # Set of extensional predicates
arity_e = None
C = set([])  # Set of constants
L = (target, P_e, arity_e, C)  # type: Tuple[Predicate, Set[Predicate], None, Set] # Language model

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


def cl(preds_int: List[Predicate], preds_ext: Set[Predicate], pred: Predicate, tau: RuleTemplate) -> OrderedSet:
    """
    Restrictions:
    1. Only clauses of atoms involving free variables. No constants in any of the clauses. - implemented
    2. Only predicates of arity 0-2. - TODO: implement elsewhere
    3. Exactly 2 atoms in the body. - implemented

    4. No unsafe (which have a variable used in the head but not the body) - implemented
    5. No circular (head atom appears in the body) - implemented
    6. No duplicate (same but different order of body atoms) - implemented
    7. No those that contain an intensional predicate in the clause body, even though int flag was set to 0, false. -
       implemented
       TODO: test the entire cl
    """
    # set of clauses that satisfy the rule template
    v, int_ = tau  # number of exist. quantified variables allowed, whether intensional predicates allowed in the body

    pred_arity = pred[1]
    total_vars = pred_arity + v

    assert total_vars <= len(string.ascii_uppercase), 'Handling of more than 26 variables not implemented!'

    vars = [Variable(string.ascii_uppercase[i]) for i in range(total_vars)]
    head = Atom((pred, tuple([vars[i] for i in range(pred_arity)])))

    possible_preds = list(preds_ext) + preds_int if int_ else list(preds_ext)  # type: List[Predicate]

    clauses = OrderedSet()
    for pred1, pred2 in product(possible_preds, possible_preds):
        for pred1_full in pred_with_vars_generator(pred1, vars):
            for pred2_full in pred_with_vars_generator(pred2, vars):
                clause = Clause(head, tuple(sorted([pred1_full, pred2_full])))
                if check_clause_unsafe(clause):
                    continue
                if check_circular(clause):
                    continue
                if not check_int_flag_satisfied(clause, int_, preds_int):
                    continue
                clauses.add(clause)
    return clauses


def check_int_flag_satisfied(clause: Clause, int_: bool, preds_int: List[Predicate]) -> bool:
    # if intensional predicate required:
    if int_:
        for atom in clause.body:
            if atom[0] in preds_int:
                return True
        return False
    # otherwise intensional predicates not in possible_preds
    return True


def check_circular(clause: Clause) -> bool:
    """
    Returns True if the clause is circular (head atom appears in the body)
    """
    head = clause.head  # type: Atom
    atoms = clause.body  # type: List[Atom]
    if head in atoms:
        return True
    return False


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


def generate_predicate(possible_preds: List[Predicate], vars: List[Variable]):
    for pred in possible_preds:
        for pred_full in pred_with_vars_generator(pred, vars):
            print(u.atom2str(pred_full))


def generate_2_predicates(possible_preds: list, vars: list):
    for pred1, pred2 in product(possible_preds, possible_preds):
        for pred1_full, pred2_full in product(pred_with_vars_generator(pred1, vars), pred_with_vars_generator(pred2, vars)):
            print(u.atom2str(pred1_full) + ', ' + u.atom2str(pred2_full))


def pred_with_vars_generator(predicate: Predicate, vars: List[Variable]) -> Iterable[Atom]:
    for combination in product(vars, repeat=predicate[1]):
        yield Atom((predicate, tuple(combination)))


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
    P_e = L[1]  # type: Set[Predicate]
    P_a = Pi[0]
    P_i = P_a + [target]  # type: List[Predicate]
    rules = Pi[2]
    clauses = []
    for p in P_i:
        tau1, tau2 = rules[p]
        clauses.append(cl(P_i, P_e, p, tau1))
        clauses.append(cl(P_i, P_e, p, tau2))
    return clauses


def p1(lambda_, alpha, W, Pi, L, B):
    return f_extract(f_infer(f_convert(B), f_generate(Pi, L), W, T), alpha)


# loss is cross-entropy
def loss(Lambda, W, Pi, L, B):
    return - np.mean([
        lambda_ * np.log(p1(lambda_, alpha, W, Pi, L, B)) + (1 - lambda_) * np.log(1 - p1(lambda_, alpha, W, Pi, L, B))
        for (alpha, lambda_) in Lambda
    ])
