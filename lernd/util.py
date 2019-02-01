#!/usr/bin/env python3
import re

from .types import Predicate, FullPredicate, Variable


def str2pred(s: str) -> Predicate:
    result = re.match(r'([a-z])+/([0-9])', s)
    if result:
        predicate_name = result.group(1)
        arity = int(result.group(2))
        return Predicate((predicate_name, arity))
    else:
        raise Exception


def str2fpred(s: str) -> FullPredicate:
    result = re.match(r'([a-z]+[0-9]*)\(([A-Z,]*)\)', s)
    name = result.group(1)
    vars_str = result.group(2)
    vars_strs = vars_str.split(',') if vars_str != '' else []
    vars = list(map(lambda x: Variable(x), vars_strs))
    arity = len(vars_strs)
    return FullPredicate((Predicate((name, arity)), vars))


def fpred2str(p: FullPredicate) -> str:
    pred, args = p
    pred_name, pred_arity = pred
    assert pred_arity == len(args), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join(args))
