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

    def test_str2fpred(self):
        fpred_strs = [('pred()', 0), ('q(X)', 1), ('pred1(X,Y,Z)', 3)]
        for fpred_str, arity in fpred_strs:
            fpred = u.str2fpred(fpred_str)
            self.assertEqual(arity, fpred[0][1], fpred_str)
            self.assertEqual(fpred_str, u.fpred2str(fpred))


class TestMain(unittest.TestCase):

    def test_arity(self):
        p = Predicate(('p', 2))
        self.assertEqual(m.arity(p), 2)

    def test_Clause_str(self):
        pred1 = FullPredicate((Predicate(('p', 2)), [Variable('X'), Variable('Y')]))
        pred2 = FullPredicate((Predicate(('q', 2)), [Variable('X'), Variable('Z')]))
        pred3 = FullPredicate((Predicate(('t', 2)), [Variable('Y'), Variable('X')]))
        clause = m.Clause(pred1, [pred2, pred3])
        self.assertEqual(clause.__str__(), 'p(X,Y)<-q(X,Z), t(Y,X)')

    def test_Clause_from_str(self):
        clause_strs = ['p(X,Y)<-q(X,Z), t(Y,X)']
        for clause_str in clause_strs:
            clause = m.Clause.from_str(clause_str)
            self.assertEqual(clause_str, clause.__str__())

    def test_Clause_equality(self):
        clause1 = m.Clause.from_str('p(X)<-q(X), q(Y)')
        clause2 = m.Clause.from_str('p(X)<-q(Y), q(X)')
        clause3 = m.Clause.from_str('p(X)<-q(X), q(Z)')
        self.assertEqual(clause1, clause2)
        self.assertNotEqual(clause1, clause3)

    def test_check_clause_unsafe(self):
        safe_clauses = ['p(X)<-q(X)', 'p(X)<-q(X), q(Y)']
        unsafe_clauses = ['p(X)<-q(Y)', 'p(X)<-q(Z), q(Y)']
        for safe_clause_str in safe_clauses:
            result = m.check_clause_unsafe(m.Clause.from_str(safe_clause_str))
            self.assertEqual(result, False)
        for unsafe_clause_str in unsafe_clauses:
            result = m.check_clause_unsafe(m.Clause.from_str(unsafe_clause_str))
            self.assertEqual(result, True)

    def test_check_circular(self):
        circular_clauses = ['p(X,Y)<-p(X,Y), q(Y)']
        uncircular_clauses = ['p(X,Y)<-p(Y,X), q(Y)']
        for circular_clause_str in circular_clauses:
            result = m.check_circular(m.Clause.from_str(circular_clause_str))
            self.assertEqual(result, True)
        for uncircular_clause_str in uncircular_clauses:
            result = m.check_circular(m.Clause.from_str(uncircular_clause_str))
            self.assertEqual(result, False)

    # def test_cl(self):
    #     # Language frame
    #     target = u.str2pred('q/2')
    #     # target = Predicate(('q', 2))
    #
    #     P_e = {u.str2pred('p/2')}
    #     # P_e = {Predicate(('p', 2))}
    #
    #     C = {'a', 'b', 'c', 'd'}
    #     L = (target, P_e, C)
    #
    #     # ILP problem
    #     B = {}
    #
    #     p = ['a/2', 'b/1', 'c/0', 'd/2']
    #     vars = ['X', 'Y', 'Z']
    #
    #     Pi = None
    #     p = None
    #
    #     tau = RuleTemplate((1, True))
    #
    #     clauses = m.cl(Pi, L, p, tau)
    #     for clause in clauses:
    #         print(clause)


if __name__ == '__main__':
    unittest.main()
