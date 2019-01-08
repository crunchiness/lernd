#!/usr/bin/env python3
from typing import Tuple, NewType, List
import string
import numpy as np

# Types
RuleTemplate = NewType('RuleTemplate', Tuple[int, bool])  # (v, int)
Predicate = NewType('Predicate', Tuple[str, int])  # (name, arity)
Variable = NewType('Variable', str)
PredicateWithArgs = NewType('PredicateWithArgs', Tuple[Predicate, List[Variable]])


def predicate_to_str(p: PredicateWithArgs) -> str:
    pred, args = p
    pred_name, pred_arity = pred
    assert pred_arity == len(args), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join(args))


class Clause:
    def __init__(self, head: PredicateWithArgs, body: List[PredicateWithArgs]):
        self.head = head
        self.body = body

    def __str__(self):
        return '{0}->{1}'.format(predicate_to_str(self.head), ','.join(map(predicate_to_str, self.body)))


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


def cl(Pi, L, p: Predicate, tau: RuleTemplate):
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
    # set of clauses that satisfy rule template
    v, int_ = tau  # number of exist. qualified variables allowed, whether intensional predicates allowed in the body
    assert v < 23, 'Handling of v > 22 not implemented!'
    var1 = Variable('X')
    var2 = Variable('Y')
    head = PredicateWithArgs((p, [var1, var2]))
    vars = {var1, var2}
    for i in range(v):
        vars.add(string.ascii_uppercase[i])
    pass


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
