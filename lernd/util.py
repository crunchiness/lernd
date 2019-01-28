#!/usr/bin/env python3
import re

from .types import Predicate, FullPredicate


def str2pred(s: str) -> Predicate:
    result = re.match(r'([a-z])+/([0-9])', s)
    if result:
        predicate_name = result.group(1)
        arity = int(result.group(2))
        return Predicate((predicate_name, arity))
    else:
        raise Exception


def fpred2str(p: FullPredicate) -> str:
    pred, args = p
    pred_name, pred_arity = pred
    assert pred_arity == len(args), 'Too many arguments for the predicate!'
    return '{0}({1})'.format(pred_name, ','.join(args))
