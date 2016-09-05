#! usr/bin/env python
"""
Created on 2016-08-29 17:24:20

@author: shuang
"""

from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import time
import os
import sys
import phy_inputs
import net


# global parameters for what you want to run.
# ==== 1: for run_trainning
# ==== 2: for run_testing
RUN_FUNCTION=1
PHY_NET=net.PhyNet_L2
MODEL='phy_net'

LEARNING_RATE=0.05
MIN_LEARNING_RATE=0.000001
L_DECAY_RATE=0.999998
MOMENTUM=0.9
WEIGHT_DECAY=0
# optimizer value = "GradientDescentOptimizer" or "MomentumOptimizer"
OPTIMIZER="GradientDescentOptimizer" 

EPOCHS=200
EVAL_FREQUENCY=5  # epoch unit.
ACCURACY_CONTROL=0.00001
EPOCH_CONTROL=10
EPS=0.000000001

SAVER_DIR='/home/wangdong/PythonCode/AI/phy/saver/0904_0'
SUMMARY_DIR='/home/wangdong/PythonCode/AI/phy/summary/0904_0'
DATA_DIR='/home/wangdong/PythonCode/AI/phy/converted_data/processed'
TRAIN_FILE_LIST=['data_s1.txt','data_s2.txt']#,'data_s3.txt','data_s4.txt','data_s5.txt']
                 #'data_s6.txt','data_s7.txt','data_s8.txt','data_s9.txt','data_s10.txt']
TEST_FILE_LIST=['data_s11.txt','data_s12.txt']#,'data_s13.txt','data_s14.txt','data_s15.txt']
                 #'data_s16.txt','data_s17.txt','data_s18.txt','data_s19.txt','data_s20.txt']
VALIDATION_FILE_LIST=['data_s21.txt','data_s22.txt','data_s23.txt','data_s24.txt','data_s25.txt']
                      #'data_s26.txt','data_s27.txt','data_s28.txt','data_s29.txt','data_s30.txt']
IMAGE_SIZE=14
LABEL_SIZE=1
BATCH_SIZE=100

def _check_dir(chk_dir):
    """
    check if chk_dir is already existed. if not, create it.
    Args:
        chk_dir: string, directory to be checking.
    """
    if os.path.exists(chk_dir):
        if os.path.isabs(chk_dir):
            print("%s is an absolute path"%(chk_dir))
        else:
            print("%s is a relative path"%(chk_dir))
    else:
        print(chk_dir+" is not existed.")
        os.mkdir(chk_dir)
        msg=chk_dir+" is created."
        print(msg)
    return True
    
def loss(net_out, labels, regularizers):
    """Calculates the loss from the net_out and the labels.
    Args:
        net_out: tensor, float - [batch_size, NUM_CLASSES].
        labels: tensor, int32 - [batch_size].
    Returns:
        loss: Loss tensor of type float.  
    """
    labels = tf.cast(labels, tf.float32)
    labels = tf.reshape(labels, [BATCH_SIZE,1])
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net_out, labels, name="softmax")
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(net_out, labels, name="sogmoid")   
    loss = tf.reduce_mean(cross_entropy, name='reduce_mean')
    loss += WEIGHT_DECAY*regularizers
    return loss
    
def train_op(loss_with_L2, train_size, init_learning_rate, momentum=MOMENTUM):
    """
    Sets up the training Ops.
    
    Creates a summarizer to track the loss over time in TensorBoard.
    
    Creates an optimizer and applies the gradients to all trainable variables.
    
    The Op returned by this function is what must be passed to the 
    `sess.run()` call to cause the model to train.
    
    Args:
        loss: Loss tensor, from loss().
        learning_rate: The learning rate to use for gradient descent.
    Returns:
        train_op: The Op for training.
    """
    tf.scalar_summary("loss_with_L2", loss_with_L2)
    # Create a variable to track the global step.
    g_step = tf.Variable(0, name='global_step', trainable=False)
    
    # learning dacay is computed as:
    # decayed_learning_rate = learning_rate *decay_rate ^ (global_step / decay_steps)
    learning_rate = tf.train.exponential_decay(
      init_learning_rate,                # Base learning rate.
      g_step,     # Current index.
      train_size/BATCH_SIZE,          # Decay step.
      L_DECAY_RATE,                # Decay rate.
      staircase=True)
    # ensure minimum of learning rate.
    learning_rate = tf.maximum(learning_rate,tf.constant(MIN_LEARNING_RATE))
    tf.scalar_summary("learning_rate", learning_rate)
    
    # define optimizer.
    try: 
        if OPTIMIZER=="MomentumOptimizer":
            optimizer = tf.train.MomentumOptimizer(learning_rate,momentum)
        elif OPTIMIZER=="GradientDescentOptimizer":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        else:
            print("Error configuration:",OPTIMIZER)
            raise ValueError('%s is not exist.'%OPTIMIZER)
    except ValueError as e:
        print(e)
        return False
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss_with_L2, global_step=g_step)
    return train_op,learning_rate
    
def evaluation(net_out, labels, accuracy):
    """Evaluate the quality of the net_out at predicting the label.

     Args:
         net_out: net_out tensor, float - [batch_size, NUM_CLASSES].
         labels: Labels tensor, int32 - [batch_size], with values in the
          range [0, NUM_CLASSES).
    Returns:
         accuracy in a batch with a float32.
    """
    y = tf.sigmoid(net_out)
    condition_tensor = tf.fill(y.get_shape().as_list(), 0.5)
    predict_labels = tf.cast(tf.greater_equal(y,condition_tensor),tf.float32)
    correct_pred = tf.equal(predict_labels,labels)
    accuracy_n = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy_o = accuracy.assign_add(accuracy_n)

    return accuracy_o


def train(data_dir=DATA_DIR,train_files=TRAIN_FILE_LIST,valid_files=VALIDATION_FILE_LIST,
          test_files=TEST_FILE_LIST,learning_rate=LEARNING_RATE,num_epochs=EPOCHS):
    """Run train for network.
    """
    train_data,train_labels = phy_inputs.raw_data_inputs(data_dir,train_files)
    valid_data,valid_labels = phy_inputs.raw_data_inputs(data_dir,valid_files)
    test_data,test_labels = phy_inputs.raw_data_inputs(data_dir,test_files)
    
    #pre process train_data.
#    d_mean,d_var = phy_inputs.reslv_xml(XML)
#    for i in xrange(IMAGE_SIZE):
#        train_data[:,i]=(train_data[:,i]-d_mean[i])/(math.sqrt(d_var[i])+EPS)
#        valid_data[:,i]=(valid_data[:,i]-d_mean[i])/(math.sqrt(d_var[i])+EPS)
#        test_data[:,i]=(test_data[:,i]-d_mean[i])/(math.sqrt(d_var[i])+EPS)

    train_size = train_labels.shape[0]
    valid_size = valid_labels.shape[0]
    test_size = test_labels.shape[0]
    print('train_size:',train_size)
    print('valid_size:',valid_size)
    print('test_size:',test_size)
    
    data_node = tf.placeholder(tf.float32,shape=(BATCH_SIZE, IMAGE_SIZE))
    labels_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE,))
    
    net_out, saver, regularizers= PHY_NET(data_node) 
    loss_out = loss(net_out, labels_node, regularizers)
    
    # variable for mean of validation's accuracy.
    accuracy_var = tf.Variable(0.0)
    div_node = tf.placeholder(tf.float32)
    
    accuracy = evaluation(net_out, labels_node, accuracy_var)
    accuracy_m = tf.div(accuracy_var,div_node)
    tf.scalar_summary("valid_accuracy", accuracy_m)
    # reset ops for 'accuracy_var' variable
    accuracy_rst = accuracy_var.assign(0.0)
 
    train_op_out,learning_r = train_op(loss_out,train_size,learning_rate)
    
    summary_op = tf.merge_all_summaries()
    
    start_time = time.time()
    valid_accuracy = 0.0
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, graph=sess.graph)
        _check_dir(SAVER_DIR)
        print('Initialized!')
        breakFlag = 1
        for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            
            feed_dict = {data_node: batch_data, labels_node: batch_labels, div_node: float(valid_size//BATCH_SIZE)}
            _, l, lr,accuracy_m_r,summary_str= sess.run([train_op_out, loss_out, learning_r, accuracy_m,summary_op],feed_dict=feed_dict)
            epoch_i = float(step) * BATCH_SIZE / train_size
            #print('gstep:',tf.train.global_step(sess, g_step) )
            #print("step:%d (epoch:%.2f)"%(step,epoch_i))
            #print('loss:%.6f, learning_r:%.6f'%(l,lr))
            #print("accuracy_m_r",accuracy_m_r)
            
            if int(epoch_i)%EVAL_FREQUENCY==0 and (offset + 2*BATCH_SIZE)>train_size:
                # accuracy for validation data.
                accuracy_rst_r = sess.run(accuracy_rst)
                print("initial validation accuracy:%f"%(accuracy_rst_r))
                for step_v in xrange(valid_size//BATCH_SIZE):
                    offset_v = step_v * BATCH_SIZE
                    v_data = valid_data[offset_v:(offset_v + BATCH_SIZE), ...]
                    v_labels = valid_labels[offset_v:(offset_v + BATCH_SIZE)]
                    
                    feed_dict = {data_node: v_data, labels_node: v_labels}
                    accuracy_r, = sess.run([accuracy],feed_dict=feed_dict)
                    
                v_accuracy = sess.run(accuracy_m,feed_dict={div_node:float(valid_size//BATCH_SIZE)})
                print("validation accuracy:%.6f"%(v_accuracy))
                if 0< v_accuracy-valid_accuracy <= ACCURACY_CONTROL and epoch_i > EPOCH_CONTROL:
                    if breakFlag == 3:
                        break
                    else:
                        breakFlag +=1
                else:
                    breakFlag = 0
                valid_accuracy = v_accuracy  
            
            #print some information and record variable.
            if (offset + 2*BATCH_SIZE)>train_size:
                print("step:%d (epoch:%.2f)"%(step,epoch_i))
                print('loss:%.6f, learning_r:%.6f'%(l,lr))
                
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print("time %.6f for one epoch"%(1000 * elapsed_time))
                
                #summary and saver.
                #print("accuracy_m_r",accuracy_m_r)
                summary_writer.add_summary(summary_str, epoch_i)
                saver.save(sess, SAVER_DIR+'/'+MODEL, global_step=step)
                summary_writer.flush()
                sys.stdout.flush()
        
        # Finally test mode.
        accuracy_rst_r = sess.run(accuracy_rst)
        print("initial test accuracy:%f"%(accuracy_rst_r))
        for step_t in xrange(test_size//BATCH_SIZE):
            offset_v = step_v * BATCH_SIZE
            t_data = valid_data[offset_v:(offset_v + BATCH_SIZE), ...]
            t_labels = valid_labels[offset_v:(offset_v + BATCH_SIZE)]
            
            feed_dict = {data_node: t_data, labels_node: t_labels}
            accuracy_r, = sess.run([accuracy],feed_dict=feed_dict)
        
        t_accuracy = sess.run(accuracy_m,feed_dict={div_node:float(test_size//BATCH_SIZE)})
        print("test accuracy:%.6f"%(t_accuracy))
        print("Great work.")
        sys.stdout.flush()
    
    return True
    
def main():
        if RUN_FUNCTION==1:
            train()  
        elif RUN_FUNCTION==2:
            # run_test()
            pass
        else:
            print('wrong')

if __name__ == '__main__':
    main()
    #test_one_train()
    