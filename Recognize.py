import numpy as np
import os
import cv2
import Constant
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def getModelStruct():
    model = keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(   Constant.char_height, 
                                            Constant.char_width, 
                                            Constant.char_channel)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(35, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def char_predict(list_char_img, model):
    if list_char_img is None:
        return None
    model=getModelStruct()
    model.load_weights(Constant.model_file)
    result=""
    for char_img in list_char_img:
        char_img = char_img.reshape(-1, Constant.char_height, 
                                        Constant.char_width , 
                                        Constant.char_channel)
        label = model.predict(char_img)

        label = np.argmax(label)
        char = Constant.label_to_char[label]
        result = result + char
    return result

