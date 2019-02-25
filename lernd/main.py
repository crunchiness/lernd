#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import itertools
from typing import List, Tuple, Dict

import numpy as np
from ordered_set import OrderedSet
from scipy.special import softmax

import lernd.util as u
from .classes import Clause, ILP, LanguageModel, ProgramTemplate
from .generator import f_generate
from .types import Atom, Constant, GroundAtom, Predicate, RuleTemplate, Variable

target = Predicate(('q', 2))
preds_ext = []  # Set of extensional predicates
constants = []  # Set of constants
language_model = LanguageModel(target, preds_ext, constants)
background_axioms = []  # Background assumptions
positive_examples = []  # Positive examples
negative_examples = []  # Negative examples
ilp_problem = ILP(language_model, background_axioms, positive_examples, negative_examples)

preds_aux = []
rules = {}  # Dict (predicate p: tuple of rule templates (tau1, tau2))
forward_chaining_steps = 0
program_template = ProgramTemplate(preds_aux, rules, forward_chaining_steps)

weights = {}  # type: Dict[Predicate, np.matrix] # set of clause weights


def make_lambda(positive_examples: List[GroundAtom], negative_examples: List[GroundAtom]) -> Dict[GroundAtom, int]:
    result = {}
    for ground_atom in positive_examples:
        result[ground_atom] = 1
    for ground_atom in negative_examples:
        result[ground_atom] = 0
    return result


big_lambda = make_lambda(positive_examples, negative_examples)


def get_ground_atoms(l: LanguageModel, pi: ProgramTemplate) -> List[GroundAtom]:
    preds_ext = l.preds_ext
    preds_aux = pi.preds_aux
    preds = preds_ext + preds_aux + [l.target]
    ground_atoms = []  # type: List[GroundAtom]
    for pred in preds:
        for constant_combination in itertools.product(l.constants, repeat=u.arity(pred)):
            ground_atoms.append(GroundAtom((pred, constant_combination)))
    return ground_atoms


print('Generating ground atoms...')
ground_atoms = get_ground_atoms(language_model, program_template)


# loss is cross-entropy
def loss(big_lambda: Dict[GroundAtom, int],
         weights: Dict[Predicate, np.matrix],
         l: LanguageModel,
         background_axioms: List[GroundAtom],
         ground_atoms: List[GroundAtom],
         program_template: ProgramTemplate
         ):
    return - np.mean((
        small_lambda * np.log(p1(alpha, weights, l, background_axioms, ground_atoms, program_template)) +
        (1 - small_lambda) * np.log(1 - p1(alpha, weights, l, background_axioms, ground_atoms, program_template))
        for (alpha, small_lambda) in big_lambda.items()
    ))


# p(lambda|alpha,W,Pi,L,B) - p1 - given particular atom, weights, program template, language model, and background
# assumptions gives the probability of label of alpha (which is 0 or 1).
def p1(alpha: GroundAtom,
       weights: Dict[Predicate, np.matrix],
       language_model: LanguageModel,
       background_axioms: List[GroundAtom],
       ground_atoms: List[GroundAtom],
       program_template: ProgramTemplate
       ) -> float:
    return f_extract(f_infer(f_convert(background_axioms, ground_atoms), f_generate(program_template, language_model), weights, forward_chaining_steps, language_model, ground_atoms), alpha, ground_atoms)


def f_extract(valuation: np.ndarray, gamma: GroundAtom, ground_atoms: List[GroundAtom]) -> float:
    # differentiable operation
    # extracts valuation value of a particular atom gamma
    return valuation[ground_atoms.index(gamma)]


def f_infer(initial_valuations: np.ndarray,  # 1D array of ground atom valuations
            clauses: Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]],  #
            # indexed set of generated clauses
            weights: Dict[Predicate, np.matrix],
            T: int,
            l: LanguageModel,
            ground_atoms: List[GroundAtom]) -> np.ndarray:
    # differentiable operation
    print('f_inferring...')
    a = initial_valuations
    for t in range(T):
        bt = np.zeros(np.shape(initial_valuations))
        # lists of clauses are of different sizes
        for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
            sm = softmax(weights[pred])
            bt = np.zeros(np.shape(initial_valuations))
            for j in range(len(clauses_1)):
                f1jp = fc(a, clauses_1[j], ground_atoms, l.constants, tau1)
                for k in range(len(clauses_2)):
                    f2kp = fc(a, clauses_2[k], ground_atoms, l.constants, tau2)
                    bt += G(f1jp, f2kp) * sm[j, k]
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


def make_xc_tensor(xc: List[Tuple[GroundAtom, List[Tuple[int, ...]]]], constants, tau: RuleTemplate, ground_atoms) -> np.ndarray:
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


def G(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    return np.maximum(f1, f2)


def f_amalgamate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # probabilistic sum (t-conorm)
    return x + y - np.multiply(x, y)


def f_convert(background_axioms: List[GroundAtom], ground_atoms: List[GroundAtom]) -> np.ndarray:
    # non-differentiable operation
    # order must be the same as in ground_atoms
    return np.array([1 if gamma in background_axioms else 0 for gamma in ground_atoms])





# def generate_weights(clauses: Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]]
#                      ) -> Dict[Predicate, np.matrix]:
#     weights_dict = {}
#     for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
#         weights_dict[pred] = np.matrix(np.zeros(shape=(len(clauses_1), len(clauses_2))))  # TODO: initialize randomly
#     return weights_dict
#
#
# print('Initializing weights...')
# weights = generate_weights(clauses)
