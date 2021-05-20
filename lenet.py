#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import Input
import numpy as np


# In[3]:


class lenet(tf.keras.Model):
    def __init__(self):
        super(lenet,self).__init__()
    
        self.conv1 = tf.keras.layers.Conv2D(filters=6,kernel_size=(5,5),strides=(1, 1),activation='relu',padding='valid',input_shape=(32,32,1))
        self.pooling1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid')
        self.conv2 = tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),strides=(1, 1),activation='relu',padding='valid')
        self.pooling2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2, 2),padding='valid')
        self.flatten_layer = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(120,activation='relu')
        self.fc2 = tf.keras.layers.Dense(84,activation='relu')
        self.fc3 = tf.keras.layers.Dense(10,activation='softmax')
        print('aa')
    
    def __call__(self,inputs):
        print('a+')
        x = self.conv1(inputs)
        x = self.pooling1(x)
        x = self.conv2(x)
        x = self.pooling2(x)
        x = self.flatten_layer(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return tf.keras.models.Model(inputs=inputs, outputs=x)


# In[ ]:




