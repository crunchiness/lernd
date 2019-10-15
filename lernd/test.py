#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import unittest

import numpy as np

from lernd import classes as c
from lernd import generator as g
from lernd import inferrer as i
from lernd import lernd_loss as l
from lernd import util as u
from lernd.classes import GroundAtoms, LanguageModel, MaybeGroundAtom, ProgramTemplate
from .lernd_types import Atom, Constant, Predicate, RuleTemplate, Variable


class TestUtil(unittest.TestCase):
    def test_arity(self):
        p = Predicate(('p', 2))
        self.assertEqual(u.arity(p), 2)

    def test_atom2str(self):
        p = Predicate(('p', 2))
        var1 = Variable('X')
        var2 = Variable('Y')
        pred = Atom((p, (var1, var2)))
        self.assertEqual(u.atom2str(pred), 'p(X,Y)')

    def test_str2pred(self):
        pred_str = 'q/2'
        pred = Predicate(('q', 2))
        self.assertEqual(u.str2pred(pred_str), pred)

    def test_str2atom(self):
        atom_strs = [('pred()', 0), ('q(X)', 1), ('pred1(X,Y,Z)', 3)]
        for atom_str, arity in atom_strs:
            atom = u.str2atom(atom_str)
            self.assertEqual(arity, atom[0][1], atom_str)
            self.assertEqual(atom_str, u.atom2str(atom))


class TestGenerator(unittest.TestCase):
    def test_check_clause_unsafe(self):
        safe_clauses = ['p(X)<-q(X)', 'p(X)<-q(X), q(Y)']
        unsafe_clauses = ['p(X)<-q(Y)', 'p(X)<-q(Z), q(Y)']
        for safe_clause_str in safe_clauses:
            result = g.check_clause_unsafe(c.Clause.from_str(safe_clause_str))
            self.assertEqual(result, False)
        for unsafe_clause_str in unsafe_clauses:
            result = g.check_clause_unsafe(c.Clause.from_str(unsafe_clause_str))
            self.assertEqual(result, True)

    def test_check_circular(self):
        circular_clauses = ['p(X,Y)<-p(X,Y), q(Y)']
        uncircular_clauses = ['p(X,Y)<-p(Y,X), q(Y)']
        for circular_clause_str in circular_clauses:
            result = g.check_circular(c.Clause.from_str(circular_clause_str))
            self.assertEqual(result, True)
        for uncircular_clause_str in uncircular_clauses:
            result = g.check_circular(c.Clause.from_str(uncircular_clause_str))
            self.assertEqual(result, False)

    def test_cl_1(self):
        preds_ext = [u.str2pred('p/2')]
        preds_int = [u.str2pred('q/2')]
        pred = u.str2pred('q/2')
        tau = RuleTemplate((0, False))
        expected_clauses = [
            'q(A,B)<-p(A,A), p(A,B)',
            'q(A,B)<-p(A,A), p(B,A)',
            'q(A,B)<-p(A,A), p(B,B)',
            'q(A,B)<-p(A,B), p(A,B)',
            'q(A,B)<-p(A,B), p(B,A)',
            'q(A,B)<-p(A,B), p(B,B)',
            'q(A,B)<-p(B,A), p(B,A)',
            'q(A,B)<-p(B,A), p(B,B)'
        ]
        clauses = g.cl(preds_int, preds_ext, pred, tau)
        for i, clause in enumerate(clauses):
            self.assertEqual(clause.__str__(), expected_clauses[i])

    def test_cl_2(self):
        preds_ext = [u.str2pred('p/2')]
        preds_int = [u.str2pred('q/2')]
        pred = u.str2pred('q/2')
        tau = RuleTemplate((1, True))
        expected_clauses = [
            'q(A,B)<-p(A,A), q(B,A)',
            'q(A,B)<-p(A,A), q(B,B)',
            'q(A,B)<-p(A,A), q(B,C)',
            'q(A,B)<-p(A,A), q(C,B)',
            'q(A,B)<-p(A,B), q(A,A)',
            'q(A,B)<-p(A,B), q(A,C)',
            'q(A,B)<-p(A,B), q(B,A)',
            'q(A,B)<-p(A,B), q(B,B)',
            'q(A,B)<-p(A,B), q(B,C)',
            'q(A,B)<-p(A,B), q(C,A)',
            'q(A,B)<-p(A,B), q(C,B)',
            'q(A,B)<-p(A,B), q(C,C)',
            'q(A,B)<-p(A,C), q(B,A)',
            'q(A,B)<-p(A,C), q(B,B)',
            'q(A,B)<-p(A,C), q(B,C)',
            'q(A,B)<-p(A,C), q(C,B)'
        ]
        expected_total = 58
        clauses = g.cl(preds_int, preds_ext, pred, tau)
        self.assertEqual(len(clauses), expected_total)
        for clause, expected_clause in zip(clauses, expected_clauses):
            self.assertEqual(clause.__str__(), expected_clause)


class TestInferrer(unittest.TestCase):
    def test_xc_rec(self):
        ground_atoms = list(map(u.str2ground_atom, [
            'p(a,a)',
            'p(a,b)',
            'p(b,a)',
            'p(b,b)',
            'q(a,a)',
            'q(a,b)',
            'q(b,a)',
            'q(b,b)',
            'r(a,a)',
            'r(a,b)',
            'r(b,a)',
            'r(b,b)'
        ]))
        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        substitution_expected = [
            (
                {
                    Variable('X'): Constant('a'),
                    Variable('Y'): Constant('a')
                },
                [(1, 5), (2, 7)]
            ),
            (
                {
                    Variable('X'): Constant('a'),
                    Variable('Y'): Constant('b')
                },
                [(1, 6), (2, 8)]
            ),
            (
                {
                    Variable('X'): Constant('b'),
                    Variable('Y'): Constant('a')
                },
                [(3, 5), (4, 7)]
            ),
            (
                {
                    Variable('X'): Constant('b'),
                    Variable('Y'): Constant('b')
                },
                [(3, 6), (4, 8)]
            )
        ]

        for substitution, expected_result in substitution_expected:
            result = i.xc_rec(clause.body, ground_atoms, substitution)
            self.assertEqual(result, expected_result)

    def test_make_xc(self):
        ground_atoms = list(map(u.str2ground_atom, [
            'p(a,a)',
            'p(a,b)',
            'p(b,a)',
            'p(b,b)',
            'q(a,a)',
            'q(a,b)',
            'q(b,a)',
            'q(b,b)',
            'r(a,a)',
            'r(a,b)',
            'r(b,a)',
            'r(b,b)'
        ]))
        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        f = u.str2ground_atom
        expected = [
            (f('p(a,a)'), []),
            (f('p(a,b)'), []),
            (f('p(b,a)'), []),
            (f('p(b,b)'), []),
            (f('q(a,a)'), []),
            (f('q(a,b)'), []),
            (f('q(b,a)'), []),
            (f('q(b,b)'), []),
            (f('r(a,a)'), [(1, 5), (2, 7)]),
            (f('r(a,b)'), [(1, 6), (2, 8)]),
            (f('r(b,a)'), [(3, 5), (4, 7)]),
            (f('r(b,b)'), [(3, 6), (4, 8)])
        ]
        self.assertEqual(i.make_xc(clause, ground_atoms), expected)

    def test_make_xc_alt(self):
        target_pred = u.str2pred('r/2')
        preds_ext = [u.str2pred('p/2'), u.str2pred('q/2')]
        preds_aux = []
        language_model = LanguageModel(target_pred, preds_ext, [Constant('a'), Constant('b')])
        program_template = ProgramTemplate(preds_aux, {}, 0)
        ground_atoms = GroundAtoms(language_model, program_template)

        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        f = u.str2ground_atom
        expected = [
            (f('r(a,a)'), [(1, 5), (2, 7)]),
            (f('r(a,b)'), [(1, 6), (2, 8)]),
            (f('r(b,a)'), [(3, 5), (4, 7)]),
            (f('r(b,b)'), [(3, 6), (4, 8)])
        ]
        self.assertEqual(i.make_xc_alt(clause, ground_atoms), expected)

    def test_make_xc_tensor(self):
        f = u.str2ground_atom
        xc = [
            (f('p(a,a)'), []),
            (f('p(a,b)'), []),
            (f('p(b,a)'), []),
            (f('p(b,b)'), []),
            (f('q(a,a)'), []),
            (f('q(a,b)'), []),
            (f('q(b,a)'), []),
            (f('q(b,b)'), []),
            (f('r(a,a)'), [(1, 5), (2, 7)]),
            (f('r(a,b)'), [(1, 6), (2, 8)]),
            (f('r(b,a)'), [(3, 5), (4, 7)]),
            (f('r(b,b)'), [(3, 6), (4, 8)])
        ]
        constants = [Constant('a'), Constant('b')]
        tau = RuleTemplate((1, False))
        ground_atoms = list(map(u.str2ground_atom, [
            'p(a,a)',
            'p(a,b)',
            'p(b,a)',
            'p(b,b)',
            'q(a,a)',
            'q(a,b)',
            'q(b,a)',
            'q(b,b)',
            'r(a,a)',
            'r(a,b)',
            'r(b,a)',
            'r(b,b)'
        ]))
        expected_tensor = np.array([
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(0, 0),
             (0, 0)],
            [(1, 5),
             (2, 7)],
            [(1, 6),
             (2, 8)],
            [(3, 5),
             (4, 7)],
            [(3, 6),
             (4, 8)]
        ])
        result = i.make_xc_tensor(xc, constants, tau, ground_atoms)
        self.assertEqual(result.tolist(), expected_tensor.tolist())

    def test_fc(self):
        # def fc(a, c: Clause, ground_atoms: List[GroundAtom], constants, tau: RuleTemplate):
        a = np.array([0, 1, 0.9, 0, 0, 0.1, 0, 0.2, 0.8, 0, 0, 0, 0])
        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        ground_atoms = list(map(u.str2ground_atom, [
            'p(a,a)',
            'p(a,b)',
            'p(b,a)',
            'p(b,b)',
            'q(a,a)',
            'q(a,b)',
            'q(b,a)',
            'q(b,b)',
            'r(a,a)',
            'r(a,b)',
            'r(b,a)',
            'r(b,b)'
        ]))
        constants = {Constant('a'), Constant('b')}
        tau = RuleTemplate((1, False))
        expected_a_apostrophe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18, 0.72, 0, 0])
        a_apostrophe = i.fc(a, clause, ground_atoms, constants, tau)
        try:
            np.testing.assert_array_almost_equal(a_apostrophe, expected_a_apostrophe)
            self.assertTrue(True)
        except AssertionError:
            self.assertTrue(False)

    def test_fc_alt(self):
        target_pred = u.str2pred('r/2')
        preds_ext = [u.str2pred('p/2'), u.str2pred('q/2')]
        preds_aux = []
        language_model = LanguageModel(target_pred, preds_ext, [Constant('a'), Constant('b')])
        program_template = ProgramTemplate(preds_aux, {}, 0)
        ground_atoms = GroundAtoms(language_model, program_template)

        a = np.array([0, 1, 0.9, 0, 0, 0.1, 0, 0.2, 0.8, 0, 0, 0, 0])
        clause = c.Clause.from_str('r(X,Y)<-p(X,Z), q(Z,Y)')
        constants = {Constant('a'), Constant('b')}
        tau = RuleTemplate((1, False))
        expected_a_apostrophe = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.18, 0.72, 0, 0])
        a_apostrophe = i.fc_alt(a, clause, ground_atoms, constants, tau)
        try:
            np.testing.assert_array_almost_equal(a_apostrophe, expected_a_apostrophe)
            self.assertTrue(True)
        except AssertionError:
            self.assertTrue(False)


class TestLerndLoss(unittest.TestCase):
    def test_get_ground_atoms(self):
        target_pred = u.str2pred('q/2')
        preds_ext = [u.str2pred('p/0')]
        preds_aux = [u.str2pred('t/1')]
        language_model = LanguageModel(target_pred, preds_ext, [Constant(x) for x in ['a', 'b', 'c']])
        program_template = ProgramTemplate(preds_aux, {}, 0)
        f = u.str2ground_atom
        expected_ground_atoms = [
            f('p()'),
            f('t(a)'),
            f('t(b)'),
            f('t(c)'),
            f('q(a,a)'),
            f('q(a,b)'),
            f('q(a,c)'),
            f('q(b,a)'),
            f('q(b,b)'),
            f('q(b,c)'),
            f('q(c,a)'),
            f('q(c,b)'),
            f('q(c,c)')
        ]
        self.assertEqual(l.get_ground_atoms(language_model, program_template), expected_ground_atoms)


class TestClasses(unittest.TestCase):
    def test_Clause_str(self):
        pred1 = Atom((Predicate(('p', 2)), (Variable('X'), Variable('Y'))))
        pred2 = Atom((Predicate(('q', 2)), (Variable('X'), Variable('Z'))))
        pred3 = Atom((Predicate(('t', 2)), (Variable('Y'), Variable('X'))))
        clause = c.Clause(pred1, (pred2, pred3))
        self.assertEqual(clause.__str__(), 'p(X,Y)<-q(X,Z), t(Y,X)')

    def test_Clause_from_str(self):
        clause_strs = ['p(X,Y)<-q(X,Z), t(Y,X)']
        for clause_str in clause_strs:
            clause = c.Clause.from_str(clause_str)
            self.assertEqual(clause_str, clause.__str__())

    def test_GroundAtoms_all_ground_atom_generator(self):
        target_pred = u.str2pred('r/2')
        preds_ext = [u.str2pred('p/2'), u.str2pred('q/2')]
        preds_aux = []
        language_model = LanguageModel(target_pred, preds_ext, [Constant('a'), Constant('b')])
        program_template = ProgramTemplate(preds_aux, {}, 0)
        ground_atoms = GroundAtoms(language_model, program_template)
        f = u.str2ground_atom
        expected_ground_atoms = [
            f('p(a,a)'),
            f('p(a,b)'),
            f('p(b,a)'),
            f('p(b,b)'),
            f('q(a,a)'),
            f('q(a,b)'),
            f('q(b,a)'),
            f('q(b,b)'),
            f('r(a,a)'),
            f('r(a,b)'),
            f('r(b,a)'),
            f('r(b,b)')
        ]
        actual_ground_atoms = list(ground_atoms.all_ground_atom_generator())
        self.assertEqual(actual_ground_atoms, expected_ground_atoms)

    def test_GroundAtoms_ground_atom_generator(self):
        target_pred = u.str2pred('r/2')
        preds_ext = [u.str2pred('p/2'), u.str2pred('q/2')]
        preds_aux = []
        language_model = LanguageModel(target_pred, preds_ext, [Constant('a'), Constant('b')])
        program_template = ProgramTemplate(preds_aux, {}, 0)
        ground_atoms = GroundAtoms(language_model, program_template)
        maybe_ground_atom = MaybeGroundAtom.from_atom(Atom((target_pred, (Variable('C'), Variable('C')))))
        actual_ground_atoms = list(list(zip(*(ground_atoms.ground_atom_generator(maybe_ground_atom))))[0])
        f = u.str2ground_atom
        expected_ground_atoms = [
            f('r(a,a)'),
            f('r(b,b)')
        ]
        self.assertEqual(actual_ground_atoms, expected_ground_atoms)


if __name__ == '__main__':
    unittest.main()
