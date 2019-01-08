import unittest

import main as m


class TestMain(unittest.TestCase):

    def test_arity(self):
        p = m.Predicate(('p', 2))
        self.assertEqual(m.arity(p), 2)

    def test_predicate_to_str(self):
        p = m.Predicate(('p', 2))
        var1 = m.Variable('X')
        var2 = m.Variable('Y')
        pred = m.PredicateWithArgs((p, [var1, var2]))
        self.assertEqual(m.predicate_to_str(pred), 'p(X,Y)')

    def test_Clause_str(self):
        pred1 = m.PredicateWithArgs((m.Predicate(('p', 2)), [m.Variable('X'), m.Variable('Y')]))
        pred2 = m.PredicateWithArgs((m.Predicate(('q', 2)), [m.Variable('X'), m.Variable('Z')]))
        pred3 = m.PredicateWithArgs((m.Predicate(('t', 2)), [m.Variable('Y'), m.Variable('X')]))
        clause = m.Clause(pred1, [pred2, pred3])
        self.assertEqual(clause.__str__(), 'p(X,Y)->q(X,Z),t(Y,X)')

    # def test_cl(self):
    #     # Language frame
    #     target = Predicate(('q', 2))
    #     P_e = {Predicate(('p', 2))}
    #     C = {'a', 'b', 'c', 'd'}
    #     L = (target, P_e, C)
    #
    #     # ILP problem
    #     B = {}


if __name__ == '__main__':
    unittest.main()
