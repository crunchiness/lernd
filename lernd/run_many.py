#!/usr/bin/env python3

__author__ = "Ingvaras Merkys"

import argparse
import os
from pathlib import Path

import psutil
import ray
import tensorflow as tf

from lernd.experiments import setup_even, setup_predecessor
from lernd.main import main_loop

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Need to add lernd to PYTHONPATH for ray to work
os.environ['PYTHONPATH'] = str(Path(os.path.realpath(__file__)).parent.parent)


@ray.remote
def run_many():
    main_loop(ilp_problem, program_template, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the task many times over. Saves results for further analysis.')
    parser.add_argument('problem', type=str, choices=['predecessor', 'even'], help='Problem to solve')
    parser.add_argument('runs', type=int, default=100, nargs='?', help='How many times to run?')
    parser.add_argument('--cpus', type=int, default=psutil.cpu_count(), help='Number of CPUs to use concurrently (ray)')
    args = parser.parse_args()

    if args.problem == 'predecessor':
        ilp_problem, program_template = setup_predecessor()
        kwargs = {
            'steps': 100,
            'mini_batch': 1.0,  # no mini batching
            'save_output': True
        }
    elif args.problem == 'even':
        ilp_problem, program_template = setup_even()
        kwargs = {
            'steps': 300,
            'mini_batch': 0.3,  # loss is based on 30% of random given examples
            'save_output': True
        }
    else:
        raise Exception('Unknown problem')

    # Make folder
    if not os.path.isdir(ilp_problem.name):
        os.mkdir(ilp_problem.name)
    else:
        if len(os.listdir(ilp_problem.name)) > 0:
            print(f'Warning: directory {ilp_problem.name} is not empty!')
    os.chdir(ilp_problem.name)

    # Write config to file
    with open('config.txt', 'w') as f:
        f.write(f'Name: {ilp_problem.name}\n')
        f.write(f'Runs: {args.runs}\n\n')
        f.write('kwargs\n')
        for k, v in kwargs.items():
            f.write(f'{k}: {v}\n')

    # Run the tasks
    ray.init(num_cpus=args.cpus)
    with tf.device('/CPU:0'):
        ray.get([run_many.remote() for _ in range(args.runs)])
