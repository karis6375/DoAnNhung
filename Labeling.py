import cv2
import numpy as numpy
import os

def Labeling(imgs):
    str_plate = input()
    start_id = (int)(input())
    path = "D:/DoAn_LTHTN/train_DoAn/NewLabel"
    #os.chdir(path)
    for i in range(len(str_plate)):
        img_fileName = (str)(10000+start_id) + ".png"
        start_id = start_id +1
        dir_path = path + "/" + str_plate[i]
        img_path = dir_path + "/" + img_fileName
        if os.path.exists(dir_path) == False:
            os.mkdir(dir_path)
        cv2.imwrite(img_path,imgs[i])

