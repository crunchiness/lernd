#!/usr/bin/env python3
import unittest

from lernd import main as m
from lernd import util as u
from .types import Predicate, Variable, FullPredicate, RuleTemplate


class TestUtil(unittest.TestCase):
    def test_fpred2str(self):
        p = Predicate(('p', 2))
        var1 = Variable('X')
        var2 = Variable('Y')
        pred = FullPredicate((p, [var1, var2]))
        self.assertEqual(u.fpred2str(pred), 'p(X,Y)')

    def test_str2pred(self):
        pred_str = 'q/2'
        pred = Predicate(('q', 2))
        self.assertEqual(u.str2pred(pred_str), pred)


class TestMain(unittest.TestCase):

    def test_arity(self):
        p = Predicate(('p', 2))
        self.assertEqual(m.arity(p), 2)

    def test_Clause_str(self):
        pred1 = FullPredicate((Predicate(('p', 2)), [Variable('X'), Variable('Y')]))
        pred2 = FullPredicate((Predicate(('q', 2)), [Variable('X'), Variable('Z')]))
        pred3 = FullPredicate((Predicate(('t', 2)), [Variable('Y'), Variable('X')]))
        clause = m.Clause(pred1, [pred2, pred3])
        self.assertEqual(clause.__str__(), 'p(X,Y)->q(X,Z),t(Y,X)')

    def test_cl(self):
        # Language frame
        target = u.str2pred('q/2')
        # target = Predicate(('q', 2))

        P_e = {u.str2pred('p/2')}
        # P_e = {Predicate(('p', 2))}

        C = {'a', 'b', 'c', 'd'}
        L = (target, P_e, C)

        # ILP problem
        B = {}

        p = ['a/2', 'b/1', 'c/0', 'd/2']
        vars = ['X', 'Y', 'Z']

        Pi = None
        p = None

        tau = RuleTemplate((0, True))

        clauses = m.cl(Pi, L, p, tau)
        for clause in clauses:
            print(clause)


if __name__ == '__main__':
    unittest.main()
