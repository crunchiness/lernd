#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

from collections import defaultdict
from typing import Dict, List, OrderedDict, Tuple

import numpy as np
import tensorflow as tf
from ordered_set import OrderedSet

from lernd.classes import Clause, GroundAtoms, LanguageModel, MaybeGroundAtom, ProgramTemplate
from lernd.lernd_types import Constant, GroundAtom, Predicate, RuleTemplate


class Inferrer:
    def __init__(self,
                 ground_atoms: GroundAtoms,
                 language_model: LanguageModel,
                 clauses: Dict[
                     Predicate,
                     Tuple[Tuple[OrderedSet[Clause], RuleTemplate], Tuple[OrderedSet[Clause], RuleTemplate]]
                 ],
                 program_template: ProgramTemplate):
        self.xc_tensors = {}
        self.ground_atoms = ground_atoms
        self.language_model = language_model
        self.clauses = clauses
        self.forward_chaining_steps = program_template.forward_chaining_steps
        self.xc_tensors = self._init_tensors()

    def _init_tensors(self):
        print('Inferrer initializing xc tensors...')
        tensors = defaultdict(list)
        for pred, clauses in self.clauses.items():
            for clauses_, tau in clauses:
                tensors[pred].append([
                    make_xc_tensor(xc, self.language_model.constants, tau, self.ground_atoms)
                    for xc in [make_xc(c, self.ground_atoms) for c in clauses_]
                ])
        return tensors

    def f_infer(self, a: tf.Tensor, weights: OrderedDict[Predicate, tf.Variable]) -> tf.Tensor:
        # differentiable operation
        for t in range(self.forward_chaining_steps):
            bt = tf.zeros(shape=np.shape(a))
            # print('Inference step:', t)
            for pred, (tensors1, tensors2) in self.xc_tensors.items():
                c_p = []
                f1jps = [fc(a, tensor) for tensor in tensors1]
                f2kps = [fc(a, tensor) for tensor in tensors2]
                for f1jp in f1jps:
                    for f2kp in f2kps:
                        c_p.append(tf.math.maximum(f1jp, f2kp))
                pred_weights = tf.reshape(weights[pred], [-1])
                sm = tf.nn.softmax(pred_weights)[:, np.newaxis]
                bt += tf.reduce_sum(input_tensor=tf.math.multiply(tf.stack(c_p), sm), axis=0)
            # f_amalgamate - probabilistic sum (t-conorm)
            a = a + bt - tf.math.multiply(a, bt)
        return a


def make_xc(c: Clause, ground_atoms: GroundAtoms) -> List[Tuple[GroundAtom, List[Tuple[int, int]]]]:
    """Creates an Xc - a set of [sets of [pairs of [indices of ground atoms]]] for clause c
    """
    xc = []
    head_pred, head_vars = c.head
    atom1, atom2 = c.body

    # for ground_atom that matches the head
    for ground_head, _ in ground_atoms.ground_atom_generator(MaybeGroundAtom.from_atom(c.head)):
        pairs = []
        ground_head_consts = ground_head[1]
        substitutions = {var: const for var, const in zip(head_vars, ground_head_consts)}
        a1 = MaybeGroundAtom.from_atom(atom1)
        a1.apply_substitutions(substitutions)
        a2 = MaybeGroundAtom.from_atom(atom2)
        a2.apply_substitutions(substitutions)
        for ground_atom1, new_subst1 in ground_atoms.ground_atom_generator(a1):
            a2_ = a2.copy()
            a2_.apply_substitutions(new_subst1)
            for ground_atom2, new_subst2 in ground_atoms.ground_atom_generator(a2_):
                i1 = ground_atoms.get_ground_atom_index(ground_atom1)
                i2 = ground_atoms.get_ground_atom_index(ground_atom2)
                pairs.append((i1, i2))
        xc.append((ground_head, pairs))
    return xc


def make_xc_tensor(
        xc: List[Tuple[GroundAtom, List[Tuple[int, int]]]],
        constants: List[Constant],
        tau: RuleTemplate,
        ground_atoms: GroundAtoms
) -> tf.Tensor:
    """Returns a tensor of indices
    """
    n = ground_atoms.len
    v = tau[0]
    w = len(constants) ** v
    xc_tensor = np.zeros((n, w, 2), dtype=np.int32)

    for ground_atom, xk_indices in xc:
        index = ground_atoms.get_ground_atom_index(ground_atom)
        if xk_indices:
            xc_tensor[index] = pad_indices(xk_indices, w)
    return tf.convert_to_tensor(xc_tensor)


def pad_indices(indices: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    m = len(indices)
    if m == n:
        return indices
    new_indices = []
    for i in range(n):
        if i < m:
            new_indices.append(indices[i])
        else:
            new_indices.append((0, 0))
    return new_indices


def fc(a: tf.Tensor, xc_tensor: tf.Tensor) -> tf.Tensor:
    x1 = xc_tensor[:, :, 0]
    x2 = xc_tensor[:, :, 1]
    y1 = tf.gather(params=a, indices=x1)
    y2 = tf.gather(params=a, indices=x2)

    # fuzzy_and - product t-norm, element-wise multiplication
    z = tf.math.multiply(y1, y2)
    return tf.reduce_max(input_tensor=z, axis=1)
