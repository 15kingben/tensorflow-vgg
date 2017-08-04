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
data_dirs = {'data/class1/': 0, 'data/class2/': 1}
NPY_PATH = 'output/vgg-demo.npy'
#########################################

all_images = []
all_labels = []

for d in data_dirs:
    ls = [d + i for i in os.listdir(d)]
    all_images += ls
    all_labels += [data_dirs[d]] * len(ls)

OUTPUT_DIM = max(all_labels) + 1



if os.path.exists(NPY_PATH):
    resume = True
else:
    resume = False




if resume:
    print("resuming training on: " + NPY_PATH)

    path = '.shuffle/shuffle_' + NPY_PATH.split('/')[-1].split('.')[0] + '.pickle'
    if not os.path.exists(path):
        print("Warning. Shuffle file not found on exisiting training file. Validation set no longer independent.")
        shuffle = make_shuffle_file(path, all_labels)

    else:
        with open(path, 'rb') as f:
            shuffle = pickle.load(f)

else:
    path = '.shuffle/shuffle_' + NPY_PATH.split('/')[-1].split('.')[0] + '.pickle'
    shuffle = make_shuffle_file(path, all_labels)



all_images = np.array(all_images)[shuffle]
all_labels = np.array(all_labels)[shuffle]


val_set_size = int(len(all_images) / 10)


val_set = all_images[-1 * val_set_size:]
val_set_labels = all_labels[-1 * val_set_size:]

all_images = all_images[:-1* val_set_size]
all_labels = all_labels[:-1* val_set_size]


def load_val_batch(start, finish):
    imgs = np.array([utils.load_image(i) for i in val_set[start:finish]])
    labels = np.array([i for i in val_set_labels[start:finish]])
    return imgs, labels



def load_batch(batch_size = 16):
    idxs = np.random.choice(all_images.shape[0], size = batch_size, replace = False)
    img_paths = [all_images[i] for i in idxs ]
    imgs = [utils.load_image(i) for i in img_paths]
    labels = [all_labels[i] for i in idxs]
    return imgs, labels


def run_batch(val_set, batch_size, fetches, feed_dict):
    tloss = 0.0
    tlabels = None
    tpc = None
    tpred = None
    tacc = 0.0
    tcpred = None

    count = 0.0
    for i in range(len(val_set) / batch_size):

        count+=1.0
        start = i*batch_size
        finish = min((i+1)*batch_size, len(val_set))

        imgs, labels = load_val_batch(start, finish)

        loss, acc, cpred, pc, pred = sess.run(fetches, feed_dict=feed_dict)
        tloss += loss
        tacc += acc

        if tcpred is None:
            tcpred = cpred
        else:
            tcpred = np.concatenate([tcpred, cpred])
        if tlabels is None:
            tlabels = labels
        else:
            tlabels = np.concatenate([tlabels, labels])
        if tpred is None:
            tpred = pred
        else:
            tpred = np.concatenate([tpred, pred])
        if tpc is None:
            tpc = pc
        else:
            tpc  = np.concatenate([tpc, pc])

    tacc /= count
    tloss /= count

    return tacc, tloss, tcpred, tlabels, tpred, tpc


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

    while step * batch_size < training_iters:
        imgs, labels = load_batch(batch_size)
        batch = np.array(imgs).reshape((-1, 224, 224, 3))
        # Run optimization op (backprop)
        sess.run(train, feed_dict={images: batch, true_out: labels, train_mode: True})

        if step % display_step == 0:
            loss, acc, cpred, pc,pred = sess.run([cost, accuracy, correct, predicted_class,vgg.prob], feed_dict={images: imgs,
                                                             true_out: labels,
                                                              train_mode: False})

            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))



        if step % validate_step == 0:
            fetches=[cost, accuracy, correct, predicted_class,vgg.prob]
            feed_dict={images: imgs, true_out: labels, train_mode: False}

            tacc, tloss, tcpred, tlabels, tpred, tpc = run_batch(val_set, batch_size, fetches, feed_dict)


            print('validation set')
            print("Accuracy: "  + str(tacc))
            print("Loss: " + str(tloss))


            vgg.save_npy(sess, NPY_PATH )
            if step % save_copy_step  == 0 and save_copy_step != -1:
                vgg.save_npy(sess, NPY_PATH[:-4] + str(step/save_copy_step) + '.npy')


            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tloss) + ", Training Accuracy= " + \
                  "{:.5f}".format(tacc))


        step += 1


print("Optimization Finished")




fetches=[cost, accuracy, correct, predicted_class,vgg.prob]
feed_dict={images: imgs, true_out: labels, train_mode: False}

tacc, tloss, tcpred, tlabels, tpred, tpc = run_batch(val_set, batch_size, fetches, feed_dict)


print('validation set')
print("Accuracy: "  + str(tacc))
print("Loss: " + str(tloss))

tpc = tpc.astype(int)
print(tlabels)
print(tpc)


print(classification_report(tlabels, tpc))# sess = tf.Session()


print(confusion_matrix(tlabels, tpc))
