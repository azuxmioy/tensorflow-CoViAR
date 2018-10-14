import os
import time
import random
import numpy as np
import PIL.Image as Image

gpunumber = 0
os.environ["CUDA_VISIBLE_DEVICES"]= str(gpunumber)
import tensorflow as tf
import dataset
import model

from coviar import get_num_frames
from coviar import load
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim import nets
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_utils

flags = tf.app.flags

# Loading path
flags.DEFINE_string('dataset', 'ucf101', 'name of the dataset')
flags.DEFINE_string('train_list', 'data/datalists/ucf101_split1_train.txt', 'train list')
flags.DEFINE_string('valid_list', 'data/datalists/ucf101_split1_test.txt', 'valid list')
flags.DEFINE_string('data_path', 'data/ucf101/mpeg4_videos', 'data path')
flags.DEFINE_string('pretrained_path', 'new_pretrained', 'name of the dataset')

# Saving directory
flags.DEFINE_string('log_path', 'logs/coviar', 'save log dir')
flags.DEFINE_string('save_path', 'model/coviar', 'save model directotry')

# Training parameters
flags.DEFINE_integer('batch_size', 4, 'Batch size.')
flags.DEFINE_integer('num_segments', 3, 'segments in each videos')
flags.DEFINE_integer('max_steps', 100000, 'Number of steps to run training.')

# Learning rate
flags.DEFINE_float('resnet_lr', 1e-6, 'earning rate')
flags.DEFINE_float('i_lr', 1e-4, 'learning rate')
flags.DEFINE_float('mv_lr', 1e-5, 'learning rate')
flags.DEFINE_float('r_lr', 1e-4, 'learning rate')

FLAGS = flags.FLAGS

if (FLAGS.dataset == 'ucf101'):
    N_CLASS = 101
elif (FLAGS.dataset == 'hmdb51'):
    N_CLASS = 51
else:
    raise ValueError('Unknown dataset ')


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def _variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def main(_):

    with tf.name_scope('input_placeholder'):
        mv_placeholder = tf.placeholder(tf.float32, 
                    shape=(None, FLAGS.num_segments, 224, 224, 3 ), name = 'mv_frame')
        i_placeholder = tf.placeholder(tf.float32,
                    shape=(None, FLAGS.num_segments, 224, 224, 3 ), name = 'i_frame')
        r_placeholder = tf.placeholder(tf.float32,
                    shape=(None, FLAGS.num_segments, 224, 224, 3 ), name = 'r_frame')

    with tf.name_scope('label_placeholder'):
        label_placeholder = tf.placeholder(tf.int32, shape=(None), name = 'labels')

    with tf.name_scope('accuracy'):
        combine_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        i_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        mv_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        r_value_ = tf.placeholder(tf.float32, shape=(), name = 'accuracy')
        tf.summary.scalar('combine_acc', combine_value_)
        tf.summary.scalar('i_acc', i_value_)
        tf.summary.scalar('mv_acc', mv_value_)
        tf.summary.scalar('r_acc', r_value_)


    with tf.name_scope('flatten_input'):
        b_size = tf.shape(mv_placeholder)[0]
        flat_mv = tf.reshape(mv_placeholder, [b_size * FLAGS.num_segments, 224, 224, 3]) # Since we have mulitple segments in a single video
        flat_i = tf.reshape(i_placeholder, [b_size * FLAGS.num_segments, 224, 224, 3])
        flat_r = tf.reshape(r_placeholder, [b_size * FLAGS.num_segments, 224, 224, 3])

    with tf.variable_scope('fc_var') as var_scope:
        mv_weights = {
            'w1': _variable_with_weight_decay('wmv1', [2048 , 512 ], 0.0005),
            'w2': _variable_with_weight_decay('wmv2', [512 , N_CLASS], 0.0005)
        }
        mv_biases = {
            'b1': _variable_with_weight_decay('bmv1', [ 512 ], 0.00),
            'b2': _variable_with_weight_decay('bmv2', [ N_CLASS ], 0.00)
        }
        i_weights = {
            'w1': _variable_with_weight_decay('wi1', [2048 , 512 ], 0.0005),
            'w2': _variable_with_weight_decay('wi2', [512 , N_CLASS], 0.0005)
        }
        i_biases = {
            'b1': _variable_with_weight_decay('bi1', [ 512 ], 0.00),
            'b2': _variable_with_weight_decay('bi2', [ N_CLASS ], 0.00)
        }
        r_weights = {
            'w1': _variable_with_weight_decay('wr1', [2048 , 512 ], 0.0005),
            'w2': _variable_with_weight_decay('wr2', [512 , N_CLASS], 0.0005)
        }
        r_biases = {
            'b1': _variable_with_weight_decay('br1', [ 512 ], 0.00),
            'b2': _variable_with_weight_decay('br2', [ N_CLASS ], 0.00)
        }

    with tf.variable_scope('fusion_var'):
        fusion = tf.get_variable('fusion', [3], initializer=tf.contrib.layers.xavier_initializer())

    with tf.device('/gpu:'+str(gpunumber)):

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            i_feature, _ = resnet_v1.resnet_v1_152(flat_mv, num_classes=None, is_training=True, scope='i_resnet')
            mv_feature, _ = resnet_v1.resnet_v1_50(flat_i, num_classes=None, is_training=True, scope='mv_resnet')
            r_feature, _ = resnet_v1.resnet_v1_50(flat_r, num_classes=None, is_training=True, scope='r_resnet')


        with tf.name_scope('reshape_feature'):
            i_feature = tf.reshape(i_feature, [-1, 2048])
            mv_feature = tf.reshape(mv_feature, [-1, 2048])
            r_feature = tf.reshape(r_feature, [-1, 2048])


        with tf.name_scope('inference_model'):

            i_sc, i_pred = model.inference_feature (i_feature, i_weights, i_biases,
                                                      FLAGS.num_segments, N_CLASS, name = 'i_inf')

            mv_sc, mv_pred = model.inference_feature (mv_feature, mv_weights, mv_biases,
                                                      FLAGS.num_segments, N_CLASS, name = 'mv_inf')

            r_sc, r_pred = model.inference_feature (r_feature, r_weights, r_biases,
                                                      FLAGS.num_segments, N_CLASS, name = 'r_inf')

            combine_sc, pred_class = model.inference_fusion ( i_sc, mv_sc, r_sc, fusion)


    with tf.name_scope('classiciation_loss'):
        one_hot_labels = tf.one_hot(label_placeholder, N_CLASS)
        mv_class_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = mv_sc, labels = one_hot_labels, dim=1))
        i_class_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = i_sc, labels = one_hot_labels, dim=1))
        r_class_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = r_sc, labels = one_hot_labels, dim=1))
        tf.summary.scalar('mv_class_loss', mv_class_loss) 
        tf.summary.scalar('i_class_loss', i_class_loss) 
        tf.summary.scalar('r_class_loss', r_class_loss)

        combine_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits = combine_sc, labels = one_hot_labels, dim=1))
        tf.summary.scalar('combine_class_loss', combine_loss)


    with tf.name_scope('weigh_decay'):
        weight_loss = sum(tf.get_collection('losses'))
        tf.summary.scalar('eight_decay_loss', weight_loss)

    with tf.name_scope('training_var_list'):
        mv_variable_list = list ( set(mv_weights.values()) | set(mv_biases.values()) )
        mv_resnet_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='mv_resnet')
        i_variable_list = list ( set(i_weights.values()) | set(i_biases.values()) )
        i_resnet_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='i_resnet')
        r_variable_list = list ( set(r_weights.values()) | set(r_biases.values()) )
        r_resnet_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='r_resnet')
    
    with tf.name_scope('summary_var'):
        _variable_summaries(mv_weights['w1'])
        _variable_summaries(i_weights['w2'])
        _variable_summaries(r_weights['w2'])
        _variable_summaries(mv_resnet_variables[0])
        _variable_summaries(i_resnet_variables[0])
        _variable_summaries(r_resnet_variables[0])
        _variable_summaries(fusion)

    with tf.name_scope('optimizer'):
        mv_fc_opt = tf.train.AdamOptimizer(FLAGS.mv_lr).minimize(mv_class_loss + weight_loss, var_list = mv_variable_list)
        mv_res_opt = tf.train.AdamOptimizer(FLAGS.resnet_lr).minimize(mv_class_loss, var_list = mv_resnet_variables)
        i_fc_opt = tf.train.AdamOptimizer(FLAGS.i_lr).minimize(i_class_loss + weight_loss, var_list = i_variable_list)
        i_res_opt = tf.train.AdamOptimizer(FLAGS.resnet_lr).minimize(i_class_loss, var_list = i_resnet_variables)
        r_fc_opt = tf.train.AdamOptimizer(FLAGS.r_lr).minimize(r_class_loss + weight_loss, var_list = r_variable_list)
        r_res_opt = tf.train.AdamOptimizer(FLAGS.resnet_lr).minimize(r_class_loss, var_list = r_resnet_variables)
        fusion_opt = tf.train.GradientDescentOptimizer(10e-6).minimize(combine_loss, var_list = fusion)


    with tf.name_scope('init_function'):
        init_var = tf.global_variables_initializer()
        init_i_resent = slim.assign_from_checkpoint_fn(
                os.path.join(FLAGS.pretrained_path, 'i_resnet.chkp'),
                slim.get_model_variables('i_resnet'))
        init_mv_resent = slim.assign_from_checkpoint_fn(
                os.path.join(FLAGS.pretrained_path, 'mv_resnet.chkp'),
                slim.get_model_variables('mv_resnet'))
        init_r_resent = slim.assign_from_checkpoint_fn(
            os.path.join(FLAGS.pretrained_path, 'r_resnet.chkp'),
            slim.get_model_variables('r_resnet'))
    
    with tf.name_scope('video_dataset'):
        train_data = dataset.buildTrainDataset_v2(FLAGS.train_list, FLAGS.data_path, FLAGS.num_segments,
                                                  batch_size = FLAGS.batch_size, augment = False,
                                                  shuffle = True, num_threads=1, buffer=100)
        test_data = dataset.buildTestDataset(FLAGS.valid_list, FLAGS.data_path, FLAGS.num_segments, 
                                             batch_size = FLAGS.batch_size, num_threads = 1, buffer = 30)

        with tf.name_scope('dataset_iterator'):
            it = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            next_data = it.get_next()
            init_data = it.make_initializer(train_data)
            it_test = tf.data.Iterator.from_structure(test_data.output_types, test_data.output_shapes)
            next_test_data = it_test.get_next()
            init_test_data = it_test.make_initializer(train_data)

    config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=config)
    
    with tf.name_scope('writer'):

        merged = tf.summary.merge_all()
        if not tf.gfile.Exists(FLAGS.log_path):
            tf.gfile.MakeDirs(FLAGS.log_path)
        previous_runs = os.listdir(FLAGS.log_path)
        if len(previous_runs) == 0:
            run_number = 1
        else:
            run_number = len(previous_runs) + 1
        logdir = 'run_%02d' % run_number
        tf.gfile.MakeDirs(os.path.join(FLAGS.log_path, logdir))
        writer = tf.summary.FileWriter(os.path.join(FLAGS.log_path, logdir), sess.graph)

    with tf.name_scope('saver'):

        if not tf.gfile.Exists(FLAGS.save_path):
            tf.gfile.MakeDirs(FLAGS.save_path)

        i_saver = tf.train.Saver(i_variable_list)
        mv_saver = tf.train.Saver(mv_variable_list)
        r_saver = tf.train.Saver(r_variable_list)
        i_resnet_saver = tf.train.Saver(i_resnet_variables)
        mv_resnet_saver = tf.train.Saver(mv_resnet_variables)
        r_resnet_saver = tf.train.Saver(r_resnet_variables)

    with tf.name_scope('intialization'):
        sess.run(init_var)
        sess.run(init_data)

        init_i_resent (sess)
        init_mv_resent (sess)
        init_r_resent(sess)

    '''
    Main training loop
    '''
    combine_acc = 0
    i_acc = 0
    mv_acc = 0
    r_acc = 0
    start_time = time.time()
    for step in range(FLAGS.max_steps):
       
        # Validation

        if (step) % 1000 == 0 :
            combine_classes = []
            mv_classes = []
            i_classes = []
            r_classes = []
            gt_label = []
            sess.run(init_test_data)

            for i in range(100):
                ti_arr, tmv_arr, tr_arr, tlabel = sess.run(next_test_data)
                print(i)
                i_class, mv_class, r_class, com_class = sess.run([i_pred, mv_pred, r_pred, pred_class], 
                                    feed_dict={mv_placeholder: tmv_arr, i_placeholder: ti_arr,
                                               r_placeholder: tr_arr , label_placeholder : tlabel })
                combine_classes = np.append(combine_classes, com_class)
                mv_classes = np.append(mv_classes, mv_class)
                i_classes = np.append(i_classes, i_class)
                r_classes = np.append(r_classes, r_class)
                gt_label = np.append(gt_label, tlabel)
            
            combine_acc = np.sum((combine_classes == gt_label)) / gt_label.size
            i_acc = np.sum((i_classes == gt_label)) / gt_label.size
            mv_acc = np.sum((mv_classes == gt_label)) / gt_label.size
            r_acc = np.sum((r_classes == gt_label)) / gt_label.size

            print('Step %d finished with accuracy: %f , %f , %f, %f' % (step, i_acc, mv_acc, r_acc, combine_acc))
        
        # Training procedure

        i_arr, mv_arr, r_arr, label = sess.run(next_data)
        summary, _ , _, _, _, _, _ , _, pred = sess.run([merged, mv_fc_opt, mv_res_opt, i_fc_opt, i_res_opt,
                                               r_fc_opt, r_res_opt, fusion_opt, pred_class],
                                    feed_dict={mv_placeholder: mv_arr, i_placeholder: i_arr,
                                               r_placeholder: r_arr , label_placeholder : label,
                                               combine_value_: combine_acc, i_value_ : i_acc,  
                                               mv_value_: mv_acc, r_value_ : r_acc})
        print(r_arr.shape)
        print(label)
        print(pred)
        if (step) % 10 == 0 :
            duration = time.time() - start_time
            print('Step %d: %.3f sec' % (step, duration))

            writer.add_summary(summary, step)
            start_time = time.time()

        # Model Saving 

        if (step) % 5000 == 0 and not step == 0 :
            i_saver.save(sess, os.path.join(FLAGS.save_path, 'i_model.chkp'), global_step = step)
            mv_saver.save(sess, os.path.join(FLAGS.save_path, 'mv_model.chkp'), global_step = step)
            r_saver.save(sess, os.path.join(FLAGS.save_path, 'r_model.chkp'), global_step = step)

        if (step) % 10000 == 0 and not step == 0 :
            i_resnet_saver.save(sess, os.path.join(FLAGS.save_path, 'i_resnet.chkp'), global_step = step)
            mv_resnet_saver.save(sess, os.path.join(FLAGS.save_path, 'mv_resnet.chkp'), global_step = step)
            r_resnet_saver.save(sess, os.path.join(FLAGS.save_path, 'r_resnet.chkp'), global_step = step)

    
    writer.close()


if __name__ == '__main__':
    tf.app.run()