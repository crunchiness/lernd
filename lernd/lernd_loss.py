#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import itertools
import random
from typing import Dict, List, Optional, OrderedDict, Tuple

import tensorflow as tf
import numpy as np
from ordered_set import OrderedSet

import lernd.util as u
from lernd.classes import GroundAtoms, ILP, LanguageModel, ProgramTemplate
from lernd.generator import f_generate
from lernd.inferrer import Inferrer
from lernd.lernd_types import GroundAtom, Predicate, RuleTemplate


def all_variables(weights):
    return [weights_ for weights_ in weights.values()]


def f_convert(background_axioms: List[GroundAtom], ground_atoms: GroundAtoms) -> tf.Tensor:
    # non-differentiable operation
    # order must be the same as in ground_atoms
    return tf.convert_to_tensor(
        [0] + [1 if gamma in background_axioms else 0 for gamma in ground_atoms.all_ground_atom_generator()],
        dtype=np.float32
    )


def get_ground_atoms(language_model: LanguageModel, program_template: ProgramTemplate) -> List[GroundAtom]:
    preds_ext = language_model.preds_ext
    preds_aux = program_template.preds_aux
    preds = preds_ext + preds_aux + [language_model.target]
    ground_atoms = []
    for pred in preds:
        for constant_combination in itertools.product(language_model.constants, repeat=u.arity(pred)):
            ground_atoms.append(GroundAtom(pred, constant_combination))
    return ground_atoms


def make_lambda(
        positive_examples: List[GroundAtom],
        negative_examples: List[GroundAtom],
        ground_atoms: GroundAtoms
) -> Tuple[tf.Tensor, tf.Tensor]:
    example_indices = []
    example_values = []
    for ground_atom in positive_examples:
        example_indices.append(ground_atoms.get_ground_atom_index(ground_atom))
        example_values.append(1)
    for ground_atom in negative_examples:
        example_indices.append(ground_atoms.get_ground_atom_index(ground_atom))
        example_values.append(0)
    return tf.convert_to_tensor(example_indices, dtype=np.int32), tf.convert_to_tensor(example_values, dtype=np.float32)


class Lernd:
    def __init__(self,
                 ilp_problem: ILP,
                 program_template: ProgramTemplate,
                 mini_batch: float = 1.0,
                 full_loss: bool = True,
                 worlds: bool = False
                 ):
        self._ilp_problem = ilp_problem
        self._language_model = ilp_problem.language_model
        self._program_template = program_template
        self._full_loss = full_loss

        # Random number generator
        self._rng = np.random.default_rng()

        print('Generating clauses...')
        self._clauses = f_generate(self._program_template, self._language_model)

        print('Generating ground atoms...')
        self._ground_atoms = GroundAtoms(self._language_model, self._program_template)

        print('Making big lambda...')
        self._big_lambda = make_lambda(ilp_problem.positive_examples, ilp_problem.negative_examples, self._ground_atoms)

        print('Generating initial valuation...')
        self._initial_valuation = f_convert(self._ilp_problem.background_axioms, self._ground_atoms)

        print('Initializing Inferrer')
        self._inferrer = Inferrer(self._ground_atoms, self._language_model, self._clauses, self._program_template)

        # Mini batching
        if mini_batch < 1.0:
            # self._mb = MiniBatchSettings(fraction=mini_batch, data_size=len(self._big_lambda[0]))
            self._mb = True
            self._mb_fraction = mini_batch
            self._mb_data_size = len(self._big_lambda[0])
            self._mb_batch_size = int(self._mb_fraction * self._mb_data_size)
        else:
            self._mb = False

        self._worlds = worlds
        if worlds:
            # TODO: worlds - split into three random sets
            # Splitting into 2 sets for now
            world_set = set([x for x in range(self._mb_data_size)])
            world_size = int(self._mb_data_size / 2)
            self._worlds_bpn1 = set(self._rng.choice(len(world_set), world_size, replace=False))
            self._worlds_bpn2 = world_set - self._worlds_bpn1

    @property
    def ilp_problem(self) -> ILP:
        return self._ilp_problem

    @property
    def forward_chaining_steps(self) -> int:
        return self._program_template.forward_chaining_steps

    @property
    def language_model(self) -> LanguageModel:
        return self._language_model

    @property
    def program_template(self) -> ProgramTemplate:
        return self._program_template

    @property
    def clauses(self) -> Dict[Predicate, Tuple[Tuple[OrderedSet, RuleTemplate], Tuple[OrderedSet, RuleTemplate]]]:
        return self._clauses

    @property
    def ground_atoms(self) -> GroundAtoms:
        return self._ground_atoms

    @property
    def big_lambda(self) -> Tuple[tf.Tensor, tf.Tensor]:
        return self._big_lambda

    def get_indices(self):
        # if not mb raise exception
        # choose world
        # choose indices
        if not self._worlds:
            indices = self._rng.choice(self._mb_data_size, self._mb_batch_size, replace=False)
            return indices
        else:
            # TODO: make prettier
            which = random.choice([0, 1])
            list_ = list(self._worlds_bpn1) if which == 0 else list(self._worlds_bpn2)
            random.shuffle(list_)
            return list_[:self._mb_batch_size]

    # loss is cross-entropy
    def loss(self, weights: OrderedDict[Predicate, tf.Variable]) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        alphas, small_lambdas = self._big_lambda
        valuation = self._inferrer.f_infer(self._initial_valuation, weights)

        # Extracting predictions for given (positive and negative) examples (f_extract)
        predictions = tf.gather(valuation, alphas)

        # full_loss is not used for training if mini batching
        if not self._mb or self._full_loss:
            full_loss = -tf.reduce_mean(
                input_tensor=small_lambdas * tf.math.log(predictions + 1e-12) +
                (1 - small_lambdas) * tf.math.log(1 - predictions + 1e-12)
            )
        else:
            full_loss = None

        if self._mb:
            indices = self.get_indices()
            small_lambdas = tf.gather(small_lambdas, indices)
            predictions = tf.gather(predictions, indices)
            training_loss = -tf.reduce_mean(
                input_tensor=small_lambdas * tf.math.log(predictions + 1e-12) +
                (1 - small_lambdas) * tf.math.log(1 - predictions + 1e-12)
            )
        else:
            training_loss = full_loss

        return training_loss, valuation, full_loss

    def grad(self,
             weights: OrderedDict[Predicate, tf.Variable]
             ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
        with tf.GradientTape() as tape:
            # print('Calculating loss...')
            loss_value, valuation, full_loss = self.loss(weights)
        # print('Calculating loss gradient...')
        return tape.gradient(loss_value, all_variables(weights)), loss_value, valuation, full_loss
