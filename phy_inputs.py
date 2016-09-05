# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:47:54 2016

@author: cjl
"""
from __future__ import print_function

import tensorflow as tf
import os.path
import convert_phy
import numpy as np
from xml.dom import minidom, Node

debug=True
FILENAME_LIST=['train30.tfrecords']

def read_and_decode(filename,imshape, labshape, flatten=True):
    ''''''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename)
    features = tf.parse_single_example(
    serialized_example,
    features={
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.string)
    })
    image = tf.decode_raw(features['image_raw'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float32)
    if flatten:
        num_elements = 1
        for i in imshape: num_elements = num_elements * i
        print(num_elements)
        image = tf.reshape(image, [num_elements])
        image.set_shape(num_elements)
    else:
        image = tf.reshape(image, imshape)
        image.set_shape(imshape)
    if flatten:
        num_elements = 1
        for i in labshape: num_elements = num_elements * i
        print(num_elements)
        label = tf.reshape(label, [num_elements])
        label.set_shape(num_elements)
    else:
        label = tf.reshape(label, labshape)
        label.set_shape(labshape)
    return image, label


def inputs(file_list, batch_size, num_epochs, num_threads,imshape, labshape, num_examples_per_epoch=128):
    if not num_epochs:
        num_epochs = None
    with tf.name_scope('input'):
        filename = tf.train.string_input_producer(
          file_list, num_epochs=num_epochs, name='string_input_producer')
        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename, imshape,labshape,flatten=False)
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch *min_fraction_of_examples_in_queue)
        images, sparse_labels = tf.train.shuffle_batch(
          [image, label], batch_size=batch_size, num_threads=num_threads,
            capacity=min_queue_examples + 3 * batch_size, enqueue_many=False,
           # Ensures a minimum amount of shuffling of examples.
           min_after_dequeue=min_queue_examples, name='batching_shuffling')
    return images, sparse_labels
    
    
def phy_inputs(data_dir, filename_list, batch_size, num_epochs, imshape, labshape):
    file_list=[]
    for filename in filename_list:
        file_list.append(os.path.join(data_dir,filename))
    images, labels = inputs(file_list=file_list, batch_size=batch_size, num_epochs=num_epochs,
                            num_threads=1, imshape=imshape, labshape=labshape)
    return images, labels


def test_phy_inputs():
    '''test phy_inputs()'''
    with tf.Graph().as_default():
        images, labels = phy_inputs(data_dir=convert_phy.CONVERTED_DIR, filename_list=FILENAME_LIST, 
                                     batch_size=2, num_epochs=0, imshape=[14], labshape=[1])
        img=images
        init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                step=0
                while not coord.should_stop():
                    if step > 50:
                        coord.request_stop()
                    else:
                        img_r,labels_r= sess.run([img,labels])
                        print("img_r.shape: ",img_r.shape)
                        print("img_r",img_r)
                        print("labels_r.shape ",labels_r.shape)
                        print("labels_r",labels_r)
                    step +=1
            except tf.errors.OutOfRangeError:
                print('Done training for %d steps.' % (step))
            finally:
                coord.request_stop()
            coord.join(threads)
        print('great work')


def raw_data_inputs(data_dir, filename_list):
    """Raw data from txt files.
    """
    for index in xrange(len(filename_list)):
        phy_images,phy_labels=convert_phy.generate_to(data_dir,filename_list[index])
        if index==0:
            raw_images=phy_images
            raw_labels=phy_labels
        else:
            raw_images=np.concatenate((raw_images,phy_images))
            raw_labels=np.concatenate((raw_labels,phy_labels))
    print(raw_images.shape)
    print(raw_labels.shape)
    return raw_images,raw_labels
    
def test_raw_data_inputs():
    raw_data_inputs('/home/wangdong/PythonCode/AI/phy/raw_data',['data_s1.txt','data_s2.txt'])
    print('great work')

def reslv_xml(xmlFile):
    """Read mean and variance from xmlFile and process data before train and test.
    """
    d_mean = []
    d_var = []
    # resolve xml file.
    doc = minidom.parse(xmlFile)
    resultNode = doc.childNodes[0]
    if resultNode.nodeType == Node.ELEMENT_NODE:
        print("<%s>"%(resultNode.tagName))
        for fileNode in resultNode.childNodes:
            if fileNode.nodeType == Node.ELEMENT_NODE and fileNode.tagName=='file':
                filename = fileNode.getAttribute('name')
                if filename=="total_analysis":
                    print('<file:total_analysis>')
                    for child in fileNode.childNodes:
                        if child.nodeType == Node.ELEMENT_NODE and child.tagName=='mean':
                            for mean in child.childNodes:
                                if mean.nodeType==Node.ELEMENT_NODE:
                                    print('<%s>'%(mean.tagName))
                                    m = mean.firstChild.nodeValue
                                    d_mean.append(float(m))
                                    print('mean:',m)
                        if child.nodeType == Node.ELEMENT_NODE and child.tagName=='var':
                            for var in child.childNodes:
                                if var.nodeType==Node.ELEMENT_NODE:
                                    print('<%s>'%(var.tagName))
                                    v = var.firstChild.nodeValue
                                    d_var.append(float(v))
                                    print('var:',v)
    # process data.
    print(d_mean)
    print(d_var)
    return d_mean,d_var

def main():
    if debug:
        #test_phy_inputs()
        test_raw_data_inputs()
    else:
        pass

if __name__=="__main__":
    main()