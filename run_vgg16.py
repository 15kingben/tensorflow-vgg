"""
Simple tester for the vgg19_trainable
"""

import tensorflow as tf
import numpy as np

import vgg16_trainable as vgg16
import utils
from sklearn.metrics import classification_report, confusion_matrix


import os
import os.path
import sys

import pickle



def make_shuffle_file(path, all_labels):
    if not os.path.isdir('.shuffle'):
        os.mkdir('.shuffle')

    shuffle = np.arange(len(all_labels))
    np.random.shuffle(shuffle)

    with open(path, 'wb') as f:
        pickle.dump(shuffle, f)

    return shuffle




##########################################
##ENTER YOUR IMAGE FILE DIRECTORIES HERE##
##########################################
data_dir = 'data/class1/'
NPY_PATH = 'output/vgg-demo.npy'
#########################################


all_images = [data_dir + i for i in os.listdir(data_dir)]



### Manually Enter ####
OUTPUT_DIM = 2
#######################


if not os.path.exists(NPY_PATH):
    print('.npy file not found, exiting')
    exit()


def load_val_batch(val_set, start, finish):
    imgs = np.array([utils.load_image(i) for i in val_set[start:finish]])
    return imgs



def load_batch(batch_size = 16):
    idxs = np.random.choice(all_images.shape[0], size = batch_size, replace = False)
    img_paths = [all_images[i] for i in idxs ]
    imgs = [utils.load_image(i) for i in img_paths]
    labels = [all_labels[i] for i in idxs]
    return imgs, labels


def run_batch(val_set, batch_size, fetches):
    tpred = None
    tpc = None

    count = 0.0
    for i in range(len(val_set) / batch_size):

        count+=1.0
        start = i*batch_size
        finish = min((i+1)*batch_size, len(val_set))

        imgs = load_val_batch(val_set, start, finish)

        pc, pred = sess.run(fetches, feed_dict={images : imgs, train_mode : False})

        if tpred is None:
            tpred = pred
        else:
            tpred = np.concatenate([tpred, pred])
        if tpc is None:
            tpc = pc
        else:
            tpc  = np.concatenate([tpc, pc])

    return tpred, tpc


batch_size = 16
training_iters = 24000
display_step = 20
validate_step = 200
save_copy_step = 800



with tf.device('/gpu:0'):



    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    true_out = tf.placeholder(tf.int64, [None])
    train_mode = tf.placeholder(tf.bool)



    vgg = vgg16.Vgg16(vgg16_npy_path= (NPY_PATH if os.path.isfile(NPY_PATH) else 'vgg16.npy'), output_dim = OUTPUT_DIM, retrain="semi")
    vgg.build(images, train_mode)



    learning_rate = .001

    # Define loss and Optimizer

    prediction = vgg.prob
    predicted_class = tf.argmax(prediction, 1)
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_out, logits=prediction))
    correct = tf.equal(predicted_class, true_out)
    accuracy = tf.reduce_mean( tf.cast(correct, 'float') )


    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    sess.run(tf.initialize_all_variables())



    step = 1
    # Keep training until reach max iterations


    print("Optimization Finished")


    fetches=[predicted_class,vgg.prob]
    tpred, tpc = run_batch(all_images, batch_size, fetches)


    print('Predictions:')

    print(tpred)
    print(tpc)
