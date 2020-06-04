import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import Recognize
import Constant


def SaveDataSet(datas,labels):
    path="./DataSet"
    for i in range(len(datas)):
        img_name = 10000 + i
        img_name = (str)(img_name) + ".jpg"
        img_dir_path = path + "/" + chr(labels[i]) 
        img_path = img_dir_path + "/" + img_name
        img = datas[i] 
        cv2.imwrite(img_path,img)

def ReadTrainData():
    datas = []
    labels = []
    path = "./DataSet"
    list_dir = os.listdir(path)
    for label in list_dir:
        img_dir_path = os.path.join(path,label) 
        list_img_path = [os.path.join(img_dir_path,f) for f in os.listdir(img_dir_path)]
        for img_path in list_img_path:
            img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
            datas.append(img)
            labels.append(ord(label)) 
    return datas,labels


#Read data for train
train_images, train_labels = ReadTrainData()
#Resharp data prepare for train
train_images=np.array(train_images)
train_images=train_images.reshape(-1,   Constant.char_height, 
                                        Constant.char_width , 
                                        Constant.char_channel)

train_labels=np.array(train_labels)
for i in range(len(train_labels)):
    train_labels[i] = Constant.char_to_label[chr(train_labels[i])]
    
model = Recognize.getModelStruct()              
model.fit(train_images, train_labels, epochs=50)

model.save(Constant.model_file)



