import pickle

import tensorflow as tf

with tf.device('/CPU:0'):
    smth = pickle.load(open('predecessor_2020-09-17_11-07-58.794596_weights.pickle', 'rb'))
    print(smth)
