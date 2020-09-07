#!/usr/bin/env python3

from __future__ import annotations

__author__ = "Ingvaras Merkys"

import re
import typing
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from .classes import GroundAtoms
from .lernd_types import Atom, Constant, GroundAtom, Predicate, Variable


# Predicate
def str2pred(s: str) -> Predicate:
    result = re.match(r'([a-z]+)/([0-9])', s)
    if result:
        predicate_name = result.group(1)
        arity = int(result.group(2))
        return Predicate(predicate_name, arity)
    else:
        raise Exception


def pred2str(pred: Predicate) -> str:
    return f'{pred[0]}/{pred[1]}'


def arity(pred: Predicate) -> int:
    return pred[1]


# Atom
def str2atom(s: str) -> Atom:
    result = re.match(r'([a-z]+[0-9]*)\(([A-Z,]*)\)', s)
    name = result.group(1)
    vars_str = result.group(2)
    vars_strs = vars_str.split(',') if vars_str != '' else []
    vars = tuple(map(lambda x: Variable(x), vars_strs))
    arity = len(vars_strs)
    return Atom(Predicate(name, arity), vars)


def atom2str(atom: Atom) -> str:
    pred, vars = atom
    pred_name, pred_arity = pred
    assert pred_arity == len(vars), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join([v.name for v in vars]))


# GroundAtom
def str2ground_atom(s: str) -> GroundAtom:
    result = re.match(r'([a-z]+[0-9]*)\(([a-z0-9,]*)\)', s)
    name = result.group(1)
    consts_str = result.group(2)
    consts_strs = consts_str.split(',') if consts_str != '' else []
    consts = tuple(map(lambda x: Constant(x), consts_strs))
    arity = len(consts_strs)
    return GroundAtom(Predicate(name, arity), consts)


def ground_atom2str(ground_atom: GroundAtom) -> str:
    pred, consts = ground_atom
    pred_name, pred_arity = pred
    assert pred_arity == len(consts), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join([c.name for c in consts]))


# Other
def get_ground_atom_probs(a: tf.Tensor, ground_atoms: GroundAtoms) -> typing.OrderedDict[GroundAtom, np.float32]:
    a = a.numpy()
    ground_atom_probs = OrderedDict()
    for ground_atom in ground_atoms.all_ground_atom_generator():
        i = ground_atoms.get_ground_atom_index(ground_atom)
        p = a[i]
        ground_atom_probs[ground_atom] = p
    return ground_atom_probs


def softmax(weights: typing.Union[tf.Tensor, tf.Variable]) -> tf.Tensor:
    """Element-wise softmax"""
    shape = weights.shape
    flat_weights = tf.reshape(weights, [-1])
    flat_probs = tf.nn.softmax(flat_weights)
    return tf.reshape(flat_probs[:, np.newaxis], shape)
