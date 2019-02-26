#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from typing import Dict, List, Tuple

import numpy as np
from ordered_set import OrderedSet
from scipy.special import softmax

from lernd.classes import Clause, LanguageModel
from lernd.types import Atom, Constant, GroundAtom, Predicate, RuleTemplate, Variable


def f_infer(initial_valuation: np.ndarray,  # 1D array of ground atom valuations
            clauses: Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]],
            weights: Dict[Predicate, np.matrix],
            forward_chaining_steps: int,
            language_model: LanguageModel,
            ground_atoms: List[GroundAtom]) -> np.ndarray:
    # differentiable operation
    a = initial_valuation
    for t in range(forward_chaining_steps):
        print('Inference step:', t)
        bt = np.zeros(np.shape(initial_valuation))
        # lists of clauses are of different sizes
        for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
            sm = softmax(weights[pred])
            bt = np.zeros(np.shape(initial_valuation))
            for j in range(len(clauses_1)):
                f1jp = fc(a, clauses_1[j], ground_atoms, language_model.constants, tau1)
                for k in range(len(clauses_2)):
                    f2kp = fc(a, clauses_2[k], ground_atoms, language_model.constants, tau2)
                    bt += g(f1jp, f2kp) * sm[j, k]
        a = f_amalgamate(a, bt)
    return a


def fc(a, c: Clause, ground_atoms: List[GroundAtom], constants, tau: RuleTemplate) -> np.ndarray:
    def gather2(a: np.ndarray, x: np.ndarray):
        # a is a vector, x is a matrix
        return a[x]

    def fuzzy_and(y1: np.ndarray, y2: np.ndarray):
        # product t-norm, element-wise multiplication
        return np.multiply(y1, y2)
    xc = make_xc(c, ground_atoms)
    xc_tensor = make_xc_tensor(xc, constants, tau, ground_atoms)
    x1 = xc_tensor[:, :, 0]
    x2 = xc_tensor[:, :, 1]
    y1 = gather2(a, x1)
    y2 = gather2(a, x2)
    z = fuzzy_and(y1, y2)
    return np.max(z, axis=1)


def make_xc(c: Clause, ground_atoms: List[GroundAtom]) -> List[Tuple[GroundAtom, List[Tuple[int, ...]]]]:
    """Creates a Xc - a set of [sets of [pairs of [indices of ground atoms]]] for clause c
    """
    xc = []
    head_pred, head_vars = c.head

    # for ground_atom that matches the head
    for ground_atom in ground_atoms:
        if ground_atom[0] == head_pred:
            ga_consts = ground_atom[1]

            # create substitution based on the head atom
            substitution = {}
            for var, const in zip(head_vars, ga_consts):
                substitution[var] = const

            # find all pairs of ground atoms for the body of this clause satisfying the substitution
            pairs = xc_rec(c.body, ground_atoms, substitution)
            xc.append((ground_atom, pairs))
        else:
            xc.append((ground_atom, []))
    return xc


def xc_rec(clause_body: Tuple[Atom, ...],
           ground_atoms: List[GroundAtom],
           substitution: Dict[Variable, Constant],
           indices: Tuple[int, ...] = tuple(),
           call_number: int = 0
           ) -> List[Tuple[int, ...]]:

    # base case
    if len(clause_body) == call_number:
        return [indices]

    atom_pred, atom_vars = clause_body[call_number]
    result = []

    # for ground_atom that matches this atom
    for i, ground_atom in enumerate(ground_atoms):
        if ground_atom[0] != atom_pred:
            continue
        ga_consts = ground_atom[1]

        # additional substitutions
        new_substitution = []
        for var, const in zip(atom_vars, ga_consts):

            # if incompatible with current substitution, discard ground_atom
            if var in substitution and substitution[var] != const:
                new_substitution = []
                break

            # otherwise add to substitution
            else:
                new_substitution.append((var, const))

        # if new substitutions added, merge them and recurse
        if len(new_substitution) > 0:
            substitution_ = substitution.copy()
            for key, value in new_substitution:
                substitution_[key] = value
            result += xc_rec(clause_body, ground_atoms, substitution_, tuple(list(indices) + [i + 1]), call_number + 1)
    return result


def make_xc_tensor(xc: List[Tuple[GroundAtom, List[Tuple[int, ...]]]],
                   constants: List[Constant],
                   tau: RuleTemplate,
                   ground_atoms: List[GroundAtom]
                   ) -> np.ndarray:
    """Returns tensor of indices
    """
    n = len(ground_atoms) + 1  # plus falsum
    v = tau[0]
    w = len(constants) ** v

    xc_tensor = np.empty((n, w, 2), dtype=int)
    xc_tensor[0] = np.zeros((w, 2))
    for k, (_, xk_indices) in enumerate(xc):
        for m in range(w):
            if m < len(xk_indices):
                xc_tensor[k + 1][m] = xk_indices[m]
            else:
                xc_tensor[k + 1][m] = (0, 0)
    return xc_tensor


def g(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    return np.maximum(f1, f2)


def f_amalgamate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # probabilistic sum (t-conorm)
    return x + y - np.multiply(x, y)
