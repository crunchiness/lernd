import os

import ray
import tensorflow as tf

from lernd.experiments import setup_even_noisy
from lernd.main import main_loop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ilp_problem, program_template = setup_even_noisy()
# ilp_problem.name += '_worlds'

kwargs = {
    'steps': 300,
    'mini_batch': 0.3,  # loss is based on 30% of random given examples
    'worlds': False,
    'save_output': True
}


@ray.remote
def run_many():
    main_loop(ilp_problem, program_template, **kwargs)


if __name__ == '__main__':
    if not os.path.isdir(ilp_problem.name):
        os.mkdir(ilp_problem.name)
    else:
        if len(os.listdir(ilp_problem.name)) > 0:
            print(f'Warning: directory {ilp_problem.name} is not empty!')
            # raise Exception(f'Directory "{ilp_problem.name}" exists and is not empty!')
    os.chdir(ilp_problem.name)
    with open('config.txt', 'w') as f:
        f.write(f'Name: {ilp_problem.name}\n')
        f.write(f'Runs: {100}\n\n')
        f.write('kwargs\n')
        for k, v in kwargs.items():
            f.write(f'{k}: {v}\n')

    ray.init(num_cpus=5)
    with tf.device('/CPU:0'):
        print(ray.get([run_many.remote() for _ in range(90)]))
