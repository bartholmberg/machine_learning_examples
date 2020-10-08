# Vanilla deep network
# https://deeplearningcourses.com/c/deep-learning-convolutional-neural-networks-theano-tensorflow
# https://udemy.com/deep-learning-convolutional-neural-networks-theano-tensorflow
# get the data: http://ufldl.stanford.edu/housenumbers/
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import os
import numpy as np
#import random
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy.io import loadmat
from sklearn.utils import shuffle
from datetime import datetime


def error_rate(p, t):
    return np.mean(p != t)


def flatten(X):
    # input will be (32, 32, 3, N)
    # output will be (N, 3072)
    N = X.shape[-1]
    flat = np.zeros((N, 3072))
    for i in range(N):
        flat[i] = X[:,:,:,i].reshape(3072)
    return flat

# In [6]: train['X'].shape
# Out[6]: (32, 32, 3, 73257)

# In [7]: train['y'].shape
# Out[7]: (73257, 1)

# In [8]: set(train['y'].flatten().tolist())
# Out[8]: {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
# We will change these to 0..9 to be 0-indexed

# In [12]: test['X'].shape
# Out[12]: (32, 32, 3, 26032)

# In [13]: test['y'].shape
# Out[13]: (26032, 1)


def get_data():
    if not os.path.exists('../large_files/train_32x32.mat'):
        print('Looking for ../large_files/train_32x32.mat')
        print('You have not downloaded the data and/or not placed the files in the correct location.')
        print('Please get the data from: http://ufldl.stanford.edu/housenumbers')
        print('Place train_32x32.mat and test_32x32.mat in the folder large_files adjacent to the class folder')
        exit()

    train = loadmat('../large_files/train_32x32.mat')
    test  = loadmat('../large_files/test_32x32.mat')
    print('loaded data')
    return train, test


def main():
    os.chdir('C:\\repo\\machine_learning_examples\\cnn_class')
    tf.compat.v1.disable_eager_execution()
    train, test = get_data()

    print("Train labels : " , train.keys())
    print ("Train shape : ", train['X'].shape,train['y'].shape)
    # Need to scale! don't leave as 0..255
    # Y is a N x 1 matrix with values 1..10 (MATLAB indexes by 1)
    # So flatten it and make it 0..9
    # Also need indicator matrix for cost calculation
    Xtrain = flatten(train['X'].astype(np.float32) / 255.)
    Ytrain = train['y'].flatten() - 1
    Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
    Xtest  = flatten(test['X'].astype(np.float32) / 255.)
    Ytest  = test['y'].flatten() - 1
    Xtest,Ytest=shuffle(Xtest,Ytest)
    plt.show()

    print( train['y'][0:10])
    for i in range(0,11):
        plt.imshow(np.squeeze( train['X'][:,:,:,i] ))
        plt.show(block=False)
    # gradient descent params
    max_iter = 30
    print_period = 10
    N, D = Xtrain.shape
    print("Train shape N,D: " , N,D)
    #return
    batch_sz = 500
    n_batches = N // batch_sz

    # initial weights
    M1 = 1000 # hidden layer size
    M2 = 500
    K = 10
    W1_init = np.random.randn(D, M1) / np.sqrt(D + M1)
    b1_init = np.zeros(M1)
    W2_init = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b2_init = np.zeros(M2)
    W3_init = np.random.randn(M2, K) / np.sqrt(M2 + K)
    b3_init = np.zeros(K)

    # define variables and expressions
    X = tf.compat.v1.placeholder(tf.float32, shape=(None, D), name='X')
    T = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='T')
    W1 = tf.Variable(W1_init.astype(np.float32))
    b1 = tf.Variable(b1_init.astype(np.float32))
    W2 = tf.Variable(W2_init.astype(np.float32))
    b2 = tf.Variable(b2_init.astype(np.float32))
    W3 = tf.Variable(W3_init.astype(np.float32))
    b3 = tf.Variable(b3_init.astype(np.float32))

    Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )
    Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )
    logits = tf.matmul(Z2, W3) + b3

    cost = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=T
        )
    )

    train_op_small = tf.compat.v1.train.RMSPropOptimizer(0.00008, decay=0.99, momentum=0.9).minimize(cost)
    train_op = tf.compat.v1.train.RMSPropOptimizer(0.0003, decay=0.995, momentum=0.9).minimize(cost)
    #train_op=tf.keras.optimizers.RMSprop(0.0003, rho=0.995, momentum=0.9).minimize(cost)
    # we'll use this to calculate the error rate
    predict_op = tf.argmax(logits, 1)
    err=1.0
    t0 = datetime.now()
    LL = []
    init = tf.compat.v1.global_variables_initializer()
    saver=tf.compat.v1.train.Saver()
  

    with tf.compat.v1.Session() as session:
        session.run(init)
        if os.path.exists('./foo_model-100.meta'): 
            #saver=tf.compat.v1.train.import_meta_graph('foo_model-100.meta')
            #saver.restore(session, tf.train.latest_checkpoint('./'))
            #graph=tf.compat.v1.get_default_graph()
            #X = graph.get_tensor_by_name("X:0")
            #W1 = tf.compat.v1.get_variable("W1:0")
            #return
            #W3 = graph.get_tensor_by_name("W3:0")
            print('loaded pre-trained model')
        for i in range(max_iter):
            for j in range(n_batches):
                Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
                Ybatch = Ytrain[j*batch_sz:(j*batch_sz + batch_sz),]
                #print ("Xtest.shape:",Xtest.shape)
                if ( abs(err) >0.4): 
                    session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})
                    print("Using large learninge coef")
                else:
                    session.run(train_op_small, feed_dict={X: Xbatch, T: Ybatch})
                saver.save(session,'foo_model',global_step=100)
                if j % print_period == 0:
                    test_cost = session.run(cost, feed_dict={X: Xtest, T: Ytest})
                    prediction = session.run(predict_op, feed_dict={X: Xtest})
                    err = error_rate(prediction, Ytest)

                    print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, err))
                    #print( "prediction size:" , prediction.shape)
                    #return
                    for k in range(8):
                        ind=np.random.random_integers(0,26031 )
                        print ("prediction at rand ind, pred, label: ",ind, (prediction[ind]+1)%10, (Ytest[ind]+1)%10)
                        XtestImage=np.squeeze( Xtest[ind,:]).reshape(32,32,3)
                        plt.imshow( XtestImage)
                        plt.show(block=False)
                        plt.draw()
                        plt.pause(1.5)
                        plt.cla()
                    LL.append(test_cost)
    print("Elapsed time:", (datetime.now() - t0))
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    main()
