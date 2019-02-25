#!/usr/bin/env python3
import itertools

__author__ = "Ingvaras Merkys"

import string
from functools import reduce
from itertools import product
from operator import add
from typing import List, Iterable, Tuple, Dict

import numpy as np
from ordered_set import OrderedSet
from scipy.special import softmax

import lernd.util as u
from .classes import Clause, LanguageModel, ProgramTemplate
from .types import Atom, Constant, GroundAtom, Predicate, RuleTemplate, Variable

# ILP problem
# Language model
target = Predicate(('q', 2))
P_e = set([])  # Set of extensional predicates
arity_e = None
C = set([])  # Set of constants

# L = (target, P_e, arity_e, C)  # type: Tuple[Predicate, Set[Predicate], None, Set] # Language model
l = LanguageModel(target, P_e, C)


B = None  # Background assumptions
P = []  # Positive examples
N = []  # Negative examples

# ILP = (L, B, P, N)  # ILP problem

# Program template
P_a = None
arity_a = None
rules = None  # Dict (predicate p: tuple of rule templates (tau1, tau2))
T = None

# pi = (P_a, arity_a, rules, T)  # Program template
pi = ProgramTemplate(P_a, rules)


W = {}  # type: Dict[Predicate, np.matrix] # set of clause weights

Lambda = [(gamma, 1) for gamma in P] + [(gamma, 0) for gamma in N]


def get_ground_atoms(l: LanguageModel, pi: ProgramTemplate) -> List[GroundAtom]:
    preds_ext = l.preds_ext
    preds_aux = pi.preds_aux
    preds = preds_ext + preds_aux + [l.target]
    ground_atoms = []  # type: List[GroundAtom]
    for pred in preds:
        for constant_combination in itertools.product(l.constants, repeat=u.arity(pred)):
            ground_atoms.append(GroundAtom((pred, constant_combination)))
    return ground_atoms


# p(lambda|alpha,W,Pi,L,B) - p1 - given particular atom, weights, program template, language model, and background
# assumptions gives the probability of label of alpha (which is 0 or 1).
def f_extract(valuation, gamma):
    # differentiable operation
    # extracts valuation value of a particular atom gamma
    pass


def f_generate(pi: ProgramTemplate, l: LanguageModel) -> Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]]:
    # non-differentiable operation
    # returns a set of clauses
    preds_int = pi.preds_aux + [target]  # type: List[Predicate]
    clauses = {}
    for pred in preds_int:
        tau1, tau2 = rules[pred]
        clauses[pred] = ((cl(preds_int, l.preds_ext, pred, tau1), tau1), (cl(preds_int, l.preds_ext, pred, tau2), tau2))
    return clauses


def p1(lambda_, alpha, W, pi: ProgramTemplate, l: LanguageModel, background_axioms: List[GroundAtom]):
    print('Generating ground atoms...')
    ground_atoms = get_ground_atoms(l, pi)

    print('Generating clauses for each predicate...')
    clauses = f_generate(pi, l)

    print('Initializing weights...')
    weights = generate_weights(clauses)

    print('f_inferring...')
    f_infer_result = f_infer(f_convert(background_axioms, ground_atoms), clauses, weights, T, l, ground_atoms)

    return f_extract(f_infer_result, alpha)


def generate_weights(clauses: Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]]
                     ) -> Dict[Predicate, np.matrix]:
    weights_dict = {}
    for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
        weights_dict[pred] = np.matrix(np.zeros(shape=(len(clauses_1), len(clauses_2))))  # TODO: initialize randomly
    return weights_dict


def f_infer(initial_valuations: np.ndarray,  # 1D array of ground atom valuations
            clauses: Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]],  # indexed set of generated clauses
            weights: Dict[Predicate, np.matrix],
            T: int,
            l: LanguageModel,
            ground_atoms: List[GroundAtom]) -> np.ndarray:
    # differentiable operation
    a = initial_valuations  #
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


def G(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    return np.maximum(f1, f2)


def f_amalgamate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # probabilistic sum (t-conorm)
    return x + y - np.multiply(x, y)


def f_convert(background_axioms: List[GroundAtom], ground_atoms: List[GroundAtom]) -> np.ndarray:
    # non-differentiable operation
    # order must be the same as in ground_atoms
    return np.array([1 if gamma in background_axioms else 0 for gamma in ground_atoms])


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


def cl(preds_int: List[Predicate], preds_ext: List[Predicate], pred: Predicate, tau: RuleTemplate) -> OrderedSet:
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


# loss is cross-entropy
def loss(Lambda, W, Pi, L, B):
    return - np.mean([
        lambda_ * np.log(p1(lambda_, alpha, W, Pi, L, B)) + (1 - lambda_) * np.log(1 - p1(lambda_, alpha, W, Pi, L, B))
        for (alpha, lambda_) in Lambda
    ])
