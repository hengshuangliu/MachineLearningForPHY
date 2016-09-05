#! usr/bin/env python
"""
Created on Tue May 24 08:58:05 2016

@author: cjl
"""
# Copyright 2016 Hamed MP. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
from xml.dom import minidom
import traceback
import os
import re
import tensorflow as tf
import phy_inputs
import math

RAW_DIR='/home/wangdong/PythonCode/AI/phy/raw_data'
RAW_FILENAME='Lambda_14.txt'
CONVERTED_DIR="/home/wangdong/PythonCode/AI/phy/converted_data/processed"

CONVERTED_FILENAME="example2"
TEMP_FILENAME='data_s31.txt'
TEMP_CONVERTED_FILENAME='train31'

START=1
END=31
'''

 while l<32:
    f = open(raw_dir+'/data_s'+str(l)+'.txt','w')
    if len(s)-i>100000:
            for i in range(100000*(l-1),100000*l):
                    f.write(s[i])
    else:
            for i in range(100000*(l-1),len(s)):
                    f.write(s[i])
    l=l+1

'''

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
        print(chk_dir+" is created.")
    return True


def txt_to_array(raw_dir,raw_filename):
    """read data , translate txt to array data."""
    data=[]
    pattern = re.compile(r'^Number')
    filename = os.path.join(raw_dir,raw_filename)
    with open(filename) as f:
        while True:
            s = f.readline()
            if not s.strip():
                break
            if pattern.match(s):
                print(s)
            else:
                arr=s.split( )
                data.append(map(float,arr))
        print("Finishing converting from .txt to .list")
   
    phy = np.array(data,np.float32)
    print(phy.shape)
    print("close %s file"%filename)
    return phy


def generate_to(raw_dir,raw_filename):
    """get images and labels"""
    raw_data=txt_to_array(raw_dir,raw_filename)
    image = raw_data[ : ,0:14]
    
    label = raw_data[:,-1]
    phy_images = np.array(image,np.float32)
    phy_labels = np.array(label,np.float32)
    print(phy_images.shape)
    print(phy_labels.shape)
    return phy_images,phy_labels


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  


def convert_to(images,labels,name):
    """ convert to tfrecords file """
    num_examples = labels.shape[0]
    print('labels shape is ', labels.shape[0])
    if images.shape[0] != num_examples:
        raise ValueError("Images size %d does not match label size %d." %
                     (images.shape[0], num_examples))
    rows = images.shape[1]
    #cols = images.shape[2]
    #depth = images.shape[3]
    
    _check_dir(CONVERTED_DIR)
    if os.path.splitext(name)[1] == '.tfrecords':
        filename = os.path.join(CONVERTED_DIR, name)
    else:
        filename = os.path.join(CONVERTED_DIR, name+'.tfrecords')
    print('Writing', filename)
    writer = tf.python_io.TFRecordWriter(filename)
    for index in xrange(num_examples):
        image_raw = images[index].tostring()
        label_raw = labels[index].tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            #'width': _int64_feature(cols),
            #'depth': _int64_feature(depth),
            'label': _bytes_feature(label_raw),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString()) 

    
def auto_convert_muti(start=START, end=END):
    """
        Auto convert order_dir/steptwoxx.txt, order_dir/steptwo_nullxx.txt and 
        traffice_dir/steptwoxx.txt into .tfrecords file.
    Args:
        order_dir: string, directory for order and n_order data files.
        traffic_dir: string, directory for traffic data files.
        start: interger, start point for filename.
        end: iterger, end point for fielname.
    """
    for index in xrange(start, end+1):
        raw_filename = 'data_s'+str(index)+'.txt'
        filename = 'train'+str(index)
        images, labels = generate_to(raw_dir=RAW_DIR,raw_filename=raw_filename)
        print("-------end generate_data()-------")
        convert_to(images,labels,filename)
        print("-------end convert_to(images,labels)------")


def phy_to_tf():
    print("-------start------")
    images, labels = generate_to(raw_dir=RAW_DIR,raw_filename=TEMP_FILENAME)
    #data = txt_to_array(raw_dir=RAW_DIR,raw_filename=RAW_FILENAME)
    print("-------end generate_data()-------")
    convert_to(images, labels,TEMP_CONVERTED_FILENAME)
    print("-------end convert_to(images,labels)------")
    print("-----great work------")
    return True        

def data_analysis(start=START, end=END):
    """analysis data to get more information.
    """
    try:
        f = open("analysis.xml", "w")
        try:
            doc = minidom.Document()
            resultNode = doc.createElement("result")
            doc.appendChild(resultNode)
            
            # analysis each txt file.
            print("analysis each txt file.")
            for index in xrange(start, end+1):
                raw_filename = 'data_s'+str(index)+'.txt'
                
                images, labels = generate_to(raw_dir=RAW_DIR,raw_filename=raw_filename)
                print("analysis file:",raw_filename)
                fileNode = doc.createElement("file")
                fileNode.setAttribute("name", raw_filename)
                resultNode.appendChild(fileNode)
                
                data_size = labels.shape[0]
                image_width = images.shape[1]
                positive_label = np.mean(labels)
                
                dataSizeNode = doc.createElement("dataSize")
                positiveLabelNode = doc.createElement("positiveLabel")
                fileNode.appendChild(dataSizeNode)
                fileNode.appendChild(positiveLabelNode)
                dataSizeTxtNode = doc.createTextNode(str(data_size))
                positiveLabelTxtNode = doc.createTextNode(str(positive_label))
                dataSizeNode.appendChild(dataSizeTxtNode)
                positiveLabelNode.appendChild(positiveLabelTxtNode)
                
                
                meanNode = doc.createElement("mean")
                varNode = doc.createElement("var")
                fileNode.appendChild(meanNode)
                fileNode.appendChild(varNode)
                for j in xrange(image_width):
                    j_mean = np.mean(images[:,j])
                    j_var = np.var(images[:,j])
                    mNode = doc.createElement("m"+str(j+1))
                    vNode = doc.createElement("v"+str(j+1))
                    meanNode.appendChild(mNode)
                    varNode.appendChild(vNode)
                    mTxtNode = doc.createTextNode(str(j_mean))
                    vTxtNode = doc.createTextNode(str(j_var))
                    mNode.appendChild(mTxtNode)
                    vNode.appendChild(vTxtNode)
                    
                if index==start:
                    raw_images=images
                    raw_labels=labels
                else:
                    raw_images=np.concatenate((raw_images,images))
                    raw_labels=np.concatenate((raw_labels,labels))
            
            # total analysis.
            print("total analysis.")
            fileNode = doc.createElement("file")
            fileNode.setAttribute("name", "total_analysis")
            resultNode.appendChild(fileNode)
            
            data_size = raw_labels.shape[0]
            image_width = raw_images.shape[1]
            positive_label = np.mean(raw_labels)
            
            dataSizeNode = doc.createElement("dataSize")
            positiveLabelNode = doc.createElement("positiveLabel")
            fileNode.appendChild(dataSizeNode)
            fileNode.appendChild(positiveLabelNode)
            dataSizeTxtNode = doc.createTextNode(str(data_size))
            positiveLabelTxtNode = doc.createTextNode(str(positive_label))
            dataSizeNode.appendChild(dataSizeTxtNode)
            positiveLabelNode.appendChild(positiveLabelTxtNode)
            
            meanNode = doc.createElement("mean")
            varNode = doc.createElement("var")
            fileNode.appendChild(meanNode)
            fileNode.appendChild(varNode)
            for j in xrange(image_width):
                j_mean = np.mean(raw_images[:,j])
                j_var = np.var(raw_images[:,j])
                mNode = doc.createElement("m"+str(j+1))
                vNode = doc.createElement("v"+str(j+1))
                meanNode.appendChild(mNode)
                varNode.appendChild(vNode)
                mTxtNode = doc.createTextNode(str(j_mean))
                vTxtNode = doc.createTextNode(str(j_var))
                mNode.appendChild(mTxtNode)
                vNode.appendChild(vTxtNode)
            
            # finally write to file.
            doc.writexml(f, "\t", "\t", "\n", "utf-8")
        except:
            traceback.print_exc()
        finally:
            f.close()
    except:
        print("open file failed")
    return True



XML='analysis.xml'
IMAGE_SIZE=14
EPS=0.000000001
def pre_process(outputDir=CONVERTED_DIR,start=START, end=END):
    """Do process for raw data.
    """
    _check_dir(outputDir)
    d_mean,d_var = phy_inputs.reslv_xml(xmlFile=XML)
    for index in xrange(start, end+1):
        raw_filename = 'data_s'+str(index)+'.txt'
        
        images, labels = generate_to(raw_dir=RAW_DIR,raw_filename=raw_filename)
        for i in xrange(IMAGE_SIZE):
            images[:,i]=(images[:,i]-d_mean[i])/(math.sqrt(d_var[i])+EPS)
        data_size = labels.shape[0]
        
        filename = os.path.join(outputDir,raw_filename)
        f = open(filename,"w")
        for line in xrange(data_size):
            string = ''
            for j in xrange(IMAGE_SIZE):
                string = string + str(images[line,j])+'  '
            string = string + str(labels[line])+'\n'
            f.write(string)
        f.close()
    print("Great work.")
    return True

def main():
    # phy_to_tf()  
    #data_analysis()
    pre_process()

if __name__=="__main__":
    main()
    








