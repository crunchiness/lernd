import numpy as np
import ray
import tensorflow as tf
from matplotlib import pyplot as plt

from lernd.experiments import setup_even
from lernd.main import main_loop
from lernd.util import str2pred, softmax


@ray.remote
def start_even(task_name):
    ilp_problem, program_template = setup_even()
    steps = 500
    mini_batch = 0.3
    weights, losses = main_loop(ilp_problem, program_template, steps=steps, mini_batch=mini_batch, plot_loss=False)

    aux_pred = str2pred('pred/2')
    target_pred = str2pred('even/1')

    fig, axs = plt.subplots(ncols=3, gridspec_kw={'width_ratios': [1, 3, 0.2]})
    fig.subplots_adjust(top=0.8, wspace=0.6)
    fig.suptitle(f'Softmaxed weight matrices at the end of task {task_name}', fontsize=16)
    im0 = axs[0].pcolormesh(softmax(weights[aux_pred]).numpy(), cmap='viridis', vmin=0, vmax=1)
    axs[0].set_title('Auxiliary predicate')
    axs[1].pcolormesh(np.transpose(softmax(weights[target_pred]).numpy()), cmap='viridis', vmin=0, vmax=1)
    axs[1].set_title('Target predicate')
    fig.colorbar(im0, cax=axs[2])
    plt.savefig(f'plot_task{task_name}.png')
    return losses[-1]


if __name__ == '__main__':
    ray.init(num_cpus=7)
    with tf.device('/CPU:0'):
        print(ray.get([start_even.remote(str(i)) for i in range(94, 101)]))
