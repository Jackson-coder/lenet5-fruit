#!/usr/bin/env python
# coding: utf-8

# In[33]:


import tensorflow as tf
from tensorflow.keras.layers import Input
import os
import cv2 as cv
import numpy as np


# In[41]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from lenet import lenet

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(len(gpus))
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# "data/train/"
# "data/test/"
def data_process(train_path, test_path):
    dic = {}
    train_label = []
    train_data = []
    train_fruit = []
    train_fruit_num = 0        
        
    for root, dirs, files in os.walk(train_path):   
        if len(dirs) != 0:
            continue
            
        #print(root,dirs,len(dirs))    
        train_fruit.append(root[11:])
        dic[train_fruit_num]=root[11:]
        
        for File in files:
            path = root + '/' + File
            file_png = cv.imread(path)
            file_png = cv.cvtColor(file_png, cv.COLOR_BGR2GRAY)
            file_png = cv.resize(file_png, (32, 32))
            file_png = tf.expand_dims(file_png, -1)
            file_png = file_png/255

            train_data.append(file_png)
            train_label.append(train_fruit_num)
        train_fruit_num+=1
        
    train_data = np.array(train_data)
    train_label = np.array(train_label)

    np.random.seed(80)
    np.random.shuffle(train_data)
    np.random.seed(80)
    np.random.shuffle(train_label)

    test_label = []
    test_data = []
    test_fruit = []
    test_fruit_num = 0
    for root, dirs, files in os.walk(test_path):
        if len(dirs) != 0:
            continue

        test_fruit.append(root[11:])
        
        for File in files:
            path = root + '/' + File
            file_png = cv.imread(path)
            file_png = cv.cvtColor(file_png, cv.COLOR_BGR2GRAY)
            file_png = cv.resize(file_png, (32, 32))
            file_png = tf.expand_dims(file_png, -1)
            file_png = file_png/255

            test_data.append(file_png)
            test_label.append(test_fruit_num)
        test_fruit_num+=1
        
    test_data = np.array(test_data)
    test_label = np.array(test_label)

    np.random.seed(80)
    np.random.shuffle(test_data)
    np.random.seed(80)
    np.random.shuffle(test_label)

    return train_data, train_label, test_data, test_label, dic

train_data, train_label, test_data, test_label,dic = data_process("data/train/", "data/test/")
print(train_data.shape,train_label.shape)
print(test_data.shape,test_label.shape)

model = lenet()(Input(shape=(32, 32, 1)))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', patience=3, verbose=1, factor=0.5)
checkpoint = ModelCheckpoint(filepath="logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",verbose=1, save_best_only=False, save_weights_only=True, period=10)


init_learning_rate=1e-4
BATCH_SIZE = 2

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_learning_rate), loss='sparse_categorical_crossentropy')
model.fit(train_data,
          train_label,
          batch_size=BATCH_SIZE,
          epochs=1000,
          verbose=2,
          validation_data=(test_data, test_label),
          initial_epoch=0,
          steps_per_epoch=max(1, len(train_label)//BATCH_SIZE),
          callbacks=[early_stopping, reduce_lr, checkpoint]
          )
model.save_weights('logs/last.h5')



model.summary()
print(dic)



