from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
import typing
from IPython.core.display import clear_output
from matplotlib import pyplot as plt
from ordered_set import OrderedSet

from lernd.classes import Clause
from lernd.experiments import setup_even_noisy
from lernd.lernd_loss import Lernd
from lernd.lernd_types import Predicate, RuleTemplate
from lernd.main import generate_weight_matrices
from lernd.util import softmax

from matplotlib import rc
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
rc('text', usetex=True)
# Had to install cm-super and dvipng
plt.rc('text.latex', preamble='\\usepackage{parskip}')


def extract_definitions(
        clauses: Dict[
            Predicate,
            Tuple[Tuple[OrderedSet[Clause], RuleTemplate], Tuple[OrderedSet[Clause], RuleTemplate]]
        ],
        weights: typing.OrderedDict[Predicate, tf.Variable]
):
    output = ''
    for pred, ((clauses_1, tau1), (clauses_2, tau2)) in clauses.items():
        shape = weights[pred].shape
        pred_weights = tf.reshape(weights[pred], [-1])
        pred_probs_flat = tf.nn.softmax(pred_weights)
        max_value = np.max(pred_probs_flat)
        clause_prob_threshold = max_value
        pred_probs = tf.reshape(pred_probs_flat[:, np.newaxis], shape)
        indices = np.nonzero(pred_probs >= clause_prob_threshold)
        if tau2 is not None:
            for index_tuple in zip(indices[0], indices[1]):
                output += f'\\\\P = {pred_probs[index_tuple]}\\\\'
                output += clauses_1[index_tuple[0]].to_latex() + '\\\\'
                output += clauses_2[index_tuple[1]].to_latex() + '\\\\'
        else:
            for index in indices[0]:
                output += f'\\\\P = {pred_probs[index][0]}\\\\'
                output += clauses_1[index].to_latex() + '\\\\'
    return output[2:-2]


def start_even():
    ilp_problem, program_template = setup_even_noisy()

    aux_pred = program_template.preds_aux[0]
    target_pred = ilp_problem.language_model.target

    lernd_model = Lernd(ilp_problem, program_template, mini_batch=0.3)
    weights = generate_weight_matrices(lernd_model.clauses)

    losses = []
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.5)

    for i in range(1, 351):
        loss_grad, loss, valuation, full_loss = lernd_model.grad(weights)
        optimizer.apply_gradients(zip(loss_grad, list(weights.values())))
        loss_float = float(full_loss.numpy())
        mb_loss_float = float(loss.numpy())
        losses.append(loss_float)
        if i % 10 == 0 or True:
            print(f'Step {i} loss: {loss_float}, mini_batch loss: {mb_loss_float}\\\\')
            fig, axs = plt.subplots(figsize=(6.4, 6.4), ncols=3, gridspec_kw={'width_ratios': [1, 3, 0.2]})
            fig.subplots_adjust(top=0.72, bottom=0.3, wspace=0.6)
            fig.suptitle(f'\\textbf{{Lernd}} is learning the predicate $even$. Step {i}', fontsize=18, y=0.96)
            im0 = axs[0].pcolormesh(softmax(weights[aux_pred]).numpy(), cmap='viridis', vmin=0, vmax=1)
            axs[0].set_title('Auxiliary predicate')
            axs[1].pcolormesh(np.transpose(softmax(weights[target_pred]).numpy()), cmap='viridis', vmin=0, vmax=1)
            axs[1].set_title('Target predicate')
            fig.colorbar(im0, cax=axs[2])

            bpn = 'Positive examples: even(0), even(2), even(4), even(6), even(8).\n' +\
                  'Negative examples: even(1), even(3), even(5), even(7), even(9), \\underline{\\textbf{even(10)}' \
                  '}.\n' +\
                  'Background axioms: zero(0), succ(0,1), succ(1,2), succ(2,3), \\dots, succ(9, 10).'
            plt.text(0.1, 0.88, bpn, fontsize=12, transform=plt.gcf().transFigure, va='top')

            text = extract_definitions(lernd_model.clauses, weights)
            plt.text(0.1, 0.22, text, fontsize=13, transform=plt.gcf().transFigure, va='top')

            loss_float = 0.0 if loss_float == 0. else loss_float
            plt.text(0.6, 0.22, f'Loss = {loss_float}', fontsize=13, transform=plt.gcf().transFigure, va='top')

            plt.text(0.995, 0.01, f'© ingvaras.com', fontsize=9, transform=plt.gcf().transFigure, ha='right',
                     fontdict={'color': 'gray'})
            plt.savefig(f'plot_step_{i:03}.png')
            plt.close()
            if i != 350:
                clear_output(wait=True)


if __name__ == '__main__':
    with tf.device('/CPU:0'):
        start_even()
