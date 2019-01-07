#!/usr/bin/env python3

import numpy as np

G = []  # All ground atoms

# ILP problem
# Language model
target = None
P_e = None
arity_e = None
C = None
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


def cl(tau):
    # set of clauses that satisfy rule template
    pass


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
        clauses.append(cl(tau1))
        clauses.append(cl(tau2))
    return clauses


def p1(lambda_, alpha, W, Pi, L, B):
    return f_extract(f_infer(f_convert(B), f_generate(Pi, L), W, T), alpha)


# loss is cross-entropy
def loss(Lambda, W, Pi, L, B):
    return - np.mean([
        lambda_ * np.log(p1(lambda_, alpha, W, Pi, L, B)) + (1 - lambda_) * np.log(1 - p1(lambda_, alpha, W, Pi, L, B))
        for (alpha, lambda_) in Lambda
    ])
