import cv2
import numpy as np

import PlateDetect
import CharacterSegment
import Recognize
import Constant
import Labeling

PlateDetect.show = True
CharacterSegment.show = True

def ReadAndResize(img_path):
    try:
        img = cv2.imread(img_path) 
        h, w, c = img.shape
        h = (int)(600*(h/w))
        img = cv2.resize(img,(600,h))
        return img
    except:
        return None

def PlateRecognize(img, model):
    try:
        #Tìm biển số xe
        img = PlateDetect.MainProcess(img)
        list_Of_Char_Img,All_Char_Img = CharacterSegment.MainProcess(img)
        cv2.imshow(" ",All_Char_Img)
        cv2.waitKey(0)
        #Labeling.Labeling(list_Of_Char_Img)   
        list_Of_Char = Recognize.char_predict(list_Of_Char_Img, model)
        return list_Of_Char
    except:
        return None


Constant.image_file = 'Test1.jpg'

#Load hình ảnh từ file lên
#31,33 loi
#Test28 1430 
model = Recognize.getModelStruct()
img = ReadAndResize(Constant.image_file) 
list_Of_Char = PlateRecognize(img, model)
print(list_Of_Char)
cv2.waitKey(0)

