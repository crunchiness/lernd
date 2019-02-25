#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import re

from .types import Atom, Constant, GroundAtom, Predicate, Variable


# Predicate
def str2pred(s: str) -> Predicate:
    result = re.match(r'([a-z])+/([0-9])', s)
    if result:
        predicate_name = result.group(1)
        arity = int(result.group(2))
        return Predicate((predicate_name, arity))
    else:
        raise Exception


# Atom
def str2atom(s: str) -> Atom:
    result = re.match(r'([a-z]+[0-9]*)\(([A-Z,]*)\)', s)
    name = result.group(1)
    vars_str = result.group(2)
    vars_strs = vars_str.split(',') if vars_str != '' else []
    vars = tuple(map(lambda x: Variable(x), vars_strs))
    arity = len(vars_strs)
    return Atom((Predicate((name, arity)), vars))


def atom2str(atom: Atom) -> str:
    pred, vars = atom
    pred_name, pred_arity = pred
    assert pred_arity == len(vars), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join(vars))


# GroundAtom
def str2ground_atom(s: str) -> GroundAtom:
    result = re.match(r'([a-z]+[0-9]*)\(([a-z,]*)\)', s)
    name = result.group(1)
    consts_str = result.group(2)
    consts_strs = consts_str.split(',') if consts_str != '' else []
    consts = tuple(map(lambda x: Constant(x), consts_strs))
    arity = len(consts_strs)
    return GroundAtom((Predicate((name, arity)), consts))


def ground_atom2str(ground_atom: GroundAtom) -> str:
    pred, consts = ground_atom
    pred_name, pred_arity = pred
    assert pred_arity == len(consts), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join(consts))


def arity(pred: Predicate) -> int:
    return pred[1]
