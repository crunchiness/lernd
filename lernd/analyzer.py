import os
from typing import Dict, Tuple, List, Iterable, Set

from lernd.classes import Clause, MaybeGroundAtom, Substitution
from lernd.lernd_types import GroundAtom, Predicate, Variable
from lernd.util import str2ground_atom, ground_atom2str


def calculate_heads(aux_pred: Clause, background: Set[GroundAtom]) -> Iterable[GroundAtom]:
    """Match background atoms to clause body to get heads that are true"""
    atom1, atom2 = aux_pred.body
    for b in filter(lambda x: x.pred == atom1.pred, background):
        s = Substitution(atom1.vars, b.consts)
        for b1 in filter(lambda x: x.pred == atom2.pred, background):
            ma2 = MaybeGroundAtom.from_atom(atom2)
            ma2.apply_substitutions(s)
            if ma2.is_ground():
                yield s.apply_to_atom(aux_pred.head)
            else:
                ma2arg1 = ma2.arg_at(0)
                ma2arg2 = ma2.arg_at(1) if ma2.arity == 2 else None
                b1const1 = b1.consts[0]
                b1const2 = b1.consts[1] if len(b1.consts) == 2 else None
                # if 2nd is const which matches
                if type(ma2arg1) == Variable and ma2arg2 == b1const2 and s.get_constant(b1const1) is None:
                    s.add(ma2arg1, b1const1)
                    yield s.apply_to_atom(aux_pred.head)
                elif type(ma2arg2) == Variable and ma2arg1 == b1const1 and s.get_constant(b1const2) is None:
                    s.add(ma2arg2, b1const2)
                    yield s.apply_to_atom(aux_pred.head)
            s = Substitution(atom1.vars, b.consts)


def try_definition(
        definition: Dict[Predicate, Tuple[float, List[Clause]]],
        positive: List[GroundAtom],
        negative: List[GroundAtom],
        background: List[GroundAtom]
):
    background = set(background)
    unique_bg_preds = set(map(lambda x: x.pred, background))
    target_pred = positive[0].pred
    results = set()
    for clause in definition[target_pred][1]:
        # check if body atoms are in the background, if not, probably need to do those first
        for atom in clause.body:
            if atom.pred not in unique_bg_preds:
                if atom.pred in definition:
                    # TODO: make recursive
                    background = add_to_background(definition[atom.pred][1], background)
                else:
                    raise Exception('Unknown predicate')
        for ground_atom in calculate_heads(clause, background):
            results.add(ground_atom)

    for positive_ground_atom in positive:
        if positive_ground_atom in results:
            results.remove(positive_ground_atom)
            print(ground_atom2str(positive_ground_atom), 'TRUE')
        else:
            print(ground_atom2str(positive_ground_atom), 'FALSE')

    for ground_atom in negative:
        if ground_atom in results:
            print('¬' + ground_atom2str(ground_atom), 'FALSE')
            results.remove(ground_atom)
        else:
            print('¬' + ground_atom2str(ground_atom), 'TRUE')

    for ground_atom in results:
        print(ground_atom2str(ground_atom), 'NEW')


def add_to_background(clauses: List[Clause], background: Set[GroundAtom]) -> Set[GroundAtom]:
    background_ = background.copy()
    for clause in clauses:
        for ground_atom in calculate_heads(clause, background):
            background_.add(ground_atom)
    return background_


def parse_definition(results_str: str) -> Dict[Predicate, Tuple[float, List[Clause]]]:
    results = []
    for bit in filter(lambda x: x[0] == '0', results_str.split('With probability (confidence): ')):
        bits = bit.split('\n')
        confidence = float(bits[0])
        preds = list(map(Clause.from_str, filter(check_clause, bits[1:])))
        results.append((confidence, preds))
    definition = {}
    for confidence, preds in results:
        pred = preds[0].head.pred
        if pred in definition:
            if definition[pred][0] < confidence:
                definition[pred] = (confidence, preds)
        else:
            definition[pred] = (confidence, preds)
    return definition


def check_clause(s: str):
    try:
        Clause.from_str(s)
    except:
        return False
    return True


def print_definition(d):
    for key, value in d.items():
        for thing in value[1]:
            print(thing)
        print()


background = list(map(str2ground_atom, ['zero(0)', 'succ(0,1)', 'succ(1,2)', 'succ(2,3)', 'succ(4,5)']))
problem = 'even_noisy_worlds'

count_good = 0
count_good_2 = 0
for filename in sorted(filter(lambda x: x[-11:] == '_losses.txt', os.listdir(problem))):
    print(filename)
    with open(os.path.join(problem, filename), 'r') as f:
        final_loss = float(f.readlines()[-1])
        if final_loss < 0.1:
            count_good += 1
        if final_loss < 0.2:
            count_good_2 += 1
    definitions_file = filename[:-11] + '_definitions.txt'
    with open(os.path.join(problem, definitions_file)) as f:
        contents = f.read()
        definitions = parse_definition(contents)
        ground_zero = str2ground_atom('zero(0)')
        print(definitions_file)
        print_definition(definitions)
        try_definition(definitions,
                       positive=[str2ground_atom(f'even({i})') for i in [0, 2, 4, 6, 8]],
                       negative=[str2ground_atom(f'even({i})') for i in [1, 3, 5, 7, 9, 10]],
                       background=[ground_zero] + [str2ground_atom(f'succ({i},{i + 1})') for i in range(10)])
        print('---')
