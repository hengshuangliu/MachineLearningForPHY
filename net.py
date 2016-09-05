# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 18:25:39 2016

@author: CJL
"""
import re
import tensorflow as tf
#epochs=1
#DATA_DIR='/home/wangdong/PythonCode/AI/phy/converted_data'
#LEARNING_RATE=0.05
#BATCH_SIZE=100
#MAX_STEPS=100000
#SAVER_DIR='/home/wangdong/PythonCode/AI/phy/saver/0714_simple'
#SUMMARY_DIR='/home/wangdong/PythonCode/AI/phy/summary/0714_simple'
#TRAIN_FILE_LIST=['train1.tfrecords','train2.tfrecords','train3.tfrecords','train4.tfrecords','train5.tfrecords',
#                 'train6.tfrecords','train7.tfrecords','train8.tfrecords','train9.tfrecords','train10.tfrecords',
#                 'train11.tfrecords','train12.tfrecords','train13.tfrecords','train14.tfrecords','train15.tfrecords',
#                 'train16.tfrecords','train17.tfrecords','train18.tfrecords','train19.tfrecords','train20.tfrecords', 
#                 'train21.tfrecords','train22.tfrecords','train23.tfrecords','train24.tfrecords','train25.tfrecords',
#                 'train26.tfrecords','train27.tfrecords','train28.tfrecords','train29.tfrecords','train30.tfrecords']  
#TEST_FILE_LIST=['train31.tfrecords']
#IMAGE_SIZE=14
#LABEL_SIZE=1

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def _activation_summary(x):
    """Helper to create summaries for activations.
    
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    
    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def PhyNet(data):
    """@Function.
    Args:
       data: a tensor with shape [batch,14]. 
    """
    X = tf.cast(data, tf.float32)
    
    weights0 = tf.Variable(tf.random_normal([14, 300],stddev=0.1))
    h0_ = tf.matmul(X, weights0)
    _activation_summary(h0_)
    
    h0 = tf.tanh(h0_)
    
    print(h0)
    
    weights1 = tf.Variable(tf.random_normal([300, 300],stddev=0.05))
    h1_ = tf.matmul(h0, weights1)
    _activation_summary(h1_)
    
    h1 = tf.tanh(h1_)
    
    weights2 = tf.Variable(tf.random_normal([300, 300],stddev=0.05))
    h2_ = tf.matmul(h1, weights2)
    _activation_summary(h2_)
    
    h2 = tf.tanh(h2_)
    
    weights3 = tf.Variable(tf.random_normal([300, 300],stddev=0.05))
    h3_ = tf.matmul(h2, weights3)
    _activation_summary(h3_)
    
    h3 = tf.tanh(h3_)
    
    weights4 = tf.Variable(tf.random_normal([300, 1],stddev=0.001))
    y = tf.matmul(h3, weights4)
    _activation_summary(y)

    net_out = tf.sigmoid(y)    
    
    saver = tf.train.Saver({v.op.name: v for v in [weights0,weights1,weights2,weights3,weights4]})
    
    return net_out, saver

def PhyNet_L2(data):
    """@Function.
    Args:
       data: a tensor with shape [batch,14]. 
    """
    X = tf.cast(data, tf.float32)
    
    weights0 = tf.Variable(tf.random_normal([14, 500],stddev=0.1))
    h0_ = tf.matmul(X, weights0)
    #_activation_summary(h0_)
    _activation_summary(weights0)
    
    h0 = tf.tanh(h0_)
    
    print(h0)
    
    weights1 = tf.Variable(tf.random_normal([500, 1000],stddev=0.05))
    h1_ = tf.matmul(h0, weights1)
    #_activation_summary(weights1)
    
    h1 = tf.tanh(h1_)
    _activation_summary(h1)
    
    weights2 = tf.Variable(tf.random_normal([1000, 1000],stddev=0.05))
    h2_ = tf.matmul(h1, weights2)
    #_activation_summary(weights2)
    
    h2 = tf.tanh(h2_)
    _activation_summary(h2)
    
    weights3 = tf.Variable(tf.random_normal([1000, 1000],stddev=0.05))
    h3_ = tf.matmul(h2, weights3)
    #_activation_summary(weights3)
    
    h3 = tf.tanh(h3_)
    _activation_summary(h3)
    
    weights3a = tf.Variable(tf.random_normal([1000, 500],stddev=0.05))
    h3a_ = tf.matmul(h3, weights3a)
    #_activation_summary(weights3)
    
    h3a = tf.tanh(h3a_)
    _activation_summary(h3a)    
    
    weights4 = tf.Variable(tf.random_normal([500, 1],stddev=0.001))
    y = tf.matmul(h3a, weights4)
    #_activation_summary(y)
    #_activation_summary(weights4)

    #net_out = tf.sigmoid(y) 
    _activation_summary(y)
    
    saver = tf.train.Saver({v.op.name: v for v in [weights0,weights1,weights2,weights3,weights3a,weights4]})
    
    regularizers = (tf.nn.l2_loss(weights0) + tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)+
                  tf.nn.l2_loss(weights3a) + tf.nn.l2_loss(weights3) +tf.nn.l2_loss(weights4))
    return y, saver, regularizers