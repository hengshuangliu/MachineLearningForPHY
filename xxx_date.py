#! usr/bin/env python
"""
Created on Sun Sep  4 17:31:16 2016

This is a configurable script for our project.

@author: wangdong
"""
#===================================Import statement=====================================================
#===================================Can not modified=====================================================
import train_test
import tensorflow as tf
from net import _activation_summary

#==================================*******************===================================================
#==================================Define your network===================================================
#==================================*******************===================================================
def PhyNet(data):
    """@Function. Define your network.
    Args:
       data: a tensor with shape [batch,14]. 
    Returns:
        y: netout.
        saver and regularizers.
    """
    X = tf.cast(data, tf.float32)
    
    #********** define layer******************
    weights0 = tf.Variable(tf.random_normal([14, 500],stddev=0.1))
    h0_ = tf.matmul(X, weights0)

    h0 = tf.tanh(h0_)
    _activation_summary(weights0)
    
    #********** define layer******************
    weights1 = tf.Variable(tf.random_normal([500, 1000],stddev=0.05))
    h1_ = tf.matmul(h0, weights1)
    
    h1 = tf.tanh(h1_)
    _activation_summary(h1)
    
    #********** define layer******************
    weights2 = tf.Variable(tf.random_normal([1000, 1000],stddev=0.05))
    h2_ = tf.matmul(h1, weights2)
    
    h2 = tf.tanh(h2_)
    _activation_summary(h2)
    
    #********** define layer******************
    weights3 = tf.Variable(tf.random_normal([1000, 1000],stddev=0.05))
    h3_ = tf.matmul(h2, weights3)
    
    h3 = tf.tanh(h3_)
    _activation_summary(h3)
    
    #********** define layer******************
    weights3a = tf.Variable(tf.random_normal([1000, 500],stddev=0.05))
    h3a_ = tf.matmul(h3, weights3a)
    
    h3a = tf.tanh(h3a_)
    _activation_summary(h3a)    
    
    #********** define layer******************
    weights4 = tf.Variable(tf.random_normal([500, 1],stddev=0.001))
    y = tf.matmul(h3a, weights4)
    _activation_summary(y)
    
    #*****************************************
    #********** define saver******************
    saver = tf.train.Saver({v.op.name: v for v in [weights0,weights1,weights2,weights3,weights3a,weights4]})

    #*****************************************
    #********** define regularizers************
    regularizers = (tf.nn.l2_loss(weights0) + tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2)+
                  tf.nn.l2_loss(weights3a) + tf.nn.l2_loss(weights3) +tf.nn.l2_loss(weights4))
    return y, saver, regularizers


#==================================***********************===================================================
#==================================Configurable parameters===================================================
#==================================***********************===================================================
train_test.PHY_NET=PhyNet # Function name of your network.

train_test.LEARNING_RATE=0.05
train_test.MIN_LEARNING_RATE=0.000001 # Floor of Leaning rate.
train_test.L_DECAY_RATE=0.999998  # Decay rate for learning rate.
train_test.WEIGHT_DECAY=0   # Constant for Regularization
# optimizer value = "GradientDescentOptimizer" or "MomentumOptimizer"
train_test.OPTIMIZER="GradientDescentOptimizer"
train_test.MOMENTUM=0.9 # This parameter for MomentumOptimizer.

train_test.EPOCHS=200 # Maximal Epoch for train.
train_test.EVAL_FREQUENCY=5  # How many epochs for each validation test.
train_test.ACCURACY_CONTROL=0.00001 # Condition for stop training.
train_test.EPOCH_CONTROL=10 # After EPOCH_CONTROL(epochs), start to decide if stop training. 

train_test.SAVER_DIR='/home/wangdong/PythonCode/AI/phy/saver/0904_0'
train_test.SUMMARY_DIR='/home/wangdong/PythonCode/AI/phy/summary/0904_0'
train_test.DATA_DIR='/home/wangdong/PythonCode/AI/phy/converted_data/processed'
train_test.TRAIN_FILE_LIST=['data_s1.txt','data_s2.txt']
train_test.TEST_FILE_LIST=['data_s11.txt','data_s12.txt']
train_test.VALIDATION_FILE_LIST=['data_s21.txt','data_s22.txt','data_s23.txt','data_s24.txt','data_s25.txt']
                      
train_test.BATCH_SIZE=100 # Batch size for train and test.

#====================================Main statement=====================================================
#===================================Can not modified====================================================
if __name__ == '__main__':
    # train and test for your network.
    train_test.train()
