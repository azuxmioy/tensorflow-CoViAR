import os
import tensorflow as tf
import random
import numpy as np


def inference_feature(_X, _weights, _biases, nSegments, nClasses, name=''):

    fc_1 = tf.nn.relu(tf.matmul(_X, _weights['w1']) + _biases['b1'], name=str(name+'_fc')) 
    sc = tf.matmul(fc_1, _weights['w2']) + _biases['b2']
    sc = tf.reshape(sc, [-1, nSegments, nClasses])
    avg_sc  = tf.reduce_mean (sc, axis = 1)
    pred = tf.argmax(tf.nn.softmax(avg_sc, axis=1), axis=1)
    return avg_sc, pred


def inference_fusion ( sc_A, sc_B, sc_C, _weights):

    combine_sc = tf.scalar_mul(_weights[0],sc_A) + tf.scalar_mul(_weights[1],sc_B) + tf.scalar_mul(_weights[2], sc_C)
    pred_class = tf.argmax(tf.nn.softmax(combine_sc, axis=1), axis=1)

    return combine_sc, pred_class

    