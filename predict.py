import tensorflow as tf
import cv2 as cv
from tensorflow.keras.layers import Input
from lenet import lenet
import numpy as np
import os

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def data_process(path):
    dic = {}
    train_fruit_num = 0 
    for root, dirs, files in os.walk(path):   
        if len(dirs) != 0:
            continue   
        dic[train_fruit_num]=root[11:]
        train_fruit_num += 1
    return dic

weights_path = "logs/last.h5"
dic = data_process("data/train/")
model = lenet()(Input(shape=(32, 32, 1)))
model.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
while True:
    img = input('Input image filename:')
    try:
        image = cv.imread(img)
        #cv.imshow("p1",image)
    except:
        print('Open Error! Try again!')
        continue
    else:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.resize(image, (32, 32))
        image = tf.expand_dims(image, -1)
        image = image/255
        image = tf.expand_dims(image, 0)
        print(image.shape)
        
        fruit_num = model.predict(image)
        fruit_num = np.array(fruit_num)
        fruit = np.argmax(fruit_num)
        print(dic)
        print(fruit_num,dic[fruit])
