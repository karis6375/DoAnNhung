import cv2
import numpy as np
import Constant
###############################################################################
show = True

###############################################################################
# 1 class de luu thong tin cac ky tu
class Character:
    def __init__(self,x,y,data):
        self.x_center=x
        self.y_center=y
        self.data=data
###############################################################################
def findCharacter(img_cropped):
    thre,dilated = Preprocess(img_cropped)
    #Tìm các contour
    contours,hier = cv2.findContours(dilated,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE) 
    #Xác định lại kích thước biển số xe
    plate_h, plate_w = thre.shape

    line1=[]
    line2=[]
    List_Of_Character=[]

    for i in range(len(contours)):
        c = contours[i]
        
        c_x, c_y, c_w, c_h = cv2.boundingRect(c) #Lấy hình chữ nhật bao từng contour
        #-------------------------------------------------------------------#
        if show == True:
            img_cropped = cv2.rectangle(img_cropped,(c_x,c_y),(c_x+c_w,c_y+c_h),
                                        color= (255,0,0), thickness=  1)          
        #-------------------------------------------------------------------#
        #nếu chiều rộng, chiều cao nằm trong một khoảng cho phép thì nó là ký tự
        if(0.02*plate_w < c_w <= 0.3*plate_w and 0.25*plate_h <= c_h <= 0.6*plate_h) :
            if ((float)(c_h)/c_w > Constant.char_min_hwRatio):
                if (hier[0][i][3] == -1) :
                    Char_img=np.copy(thre[c_y:c_y+c_h,c_x:c_x+c_w])
                    if(c_h/c_w > Constant.char_std_hwRatio):
                        right_blank = (int)(c_h/Constant.char_std_hwRatio-c_w)
                        Char_img=cv2.copyMakeBorder(Char_img, 0, 0, 0, right_blank,
                                                    borderType=cv2.BORDER_CONSTANT,value=0)   
                    Char_img=cv2.resize(Char_img,(Constant.char_width,Constant.char_height),
                                                    interpolation=cv2.INTER_AREA)  
                    char=Character(c_x+c_w/2,c_y+c_h/2,Char_img)
                    if(char.y_center < plate_h/2):
                        line1.append(char)
                    else:
                        line2.append(char)
                    if show == True:
                        img_cropped = cv2.rectangle(img_cropped,(c_x,c_y),(c_x+c_w,c_y+c_h),(0,255,0),1)

    line1=sorted(line1, key=lambda Character: Character.x_center)
    line2=sorted(line2, key=lambda Character: Character.x_center)
    List_Of_Char=line1+line2
    List_Of_Character_Img = []
    All_Char_Img = np.zeros((56,0),dtype=np.uint8)
    for char in List_Of_Char:
        List_Of_Character_Img.append(char.data)
        All_Char_Img = np.concatenate((All_Char_Img,char.data),1)
 
    if show == True:
        cv2.imshow("CharacterSegmentation",img_cropped)

    return List_Of_Character_Img,All_Char_Img
###############################################################################
def Plate_BoundingRect(cropped):
    #_____________Now useless_____________#
    cropped = cv2.bitwise_not(cropped)
    contours,hier2 = cv2.findContours(cropped,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
    contours = sorted(contours, key = cv2.contourArea, reverse = True) [:1]
    p_x,p_y,p_w,p_h = cv2.boundingRect(contours[0])
    #----------------------------------------------------------------------------------
    if (show == True):
        cv2.cvtColor(cropped,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(cropped,contours[0],1,(0,255,0),3)
        cv2.imshow("",cropped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return p_x,p_y,p_w,p_h
###############################################################################
def Preprocess(img):
        
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if(show==True):
        cv2.imshow("Gray",img)     
    #Chuyển thành màu xám     
      
    img = cv2.GaussianBlur(img,(3,3),1)   
    if(show==True):
        cv2.imshow("Blur",img)
    #Làm mờ để giảm nhiễu

    # img = cv2.equalizeHist(img)  
    # if(show==True):
    #     cv2.imshow("Equalize_histogram",img)
    # #Cân bằng sáng

    kernel_TopHat = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
    imgTopHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel_TopHat)
    
    kernel_BlackHat = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
    imgBlackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel_BlackHat)
    
    img = cv2.add(img, imgTopHat)
    if(show==True):
        cv2.imshow("Add_Top_Hat",img)

    img = cv2.subtract(img, imgBlackHat)
    if(show==True):
        cv2.imshow("Sub_Black_Hat",img)  
    #tăng độ tương phản cho các chi tiết nhỏ như ký tự trong biển số xe
       
    ret, img =cv2.threshold(img,120,255,cv2.THRESH_BINARY_INV)
    plate_thre=np.copy(img)
    if(show == True):
        cv2.imshow("Threshold",img)
    #Chuyển ảnh thành trắng đen 
     
    kerel_dilated=cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))   
    img = cv2.morphologyEx(img,cv2.MORPH_DILATE,kerel_dilated) 
    plate_dilated=np.copy(img)
    if(show == True):
        cv2.imshow("Dilated",img)
    #Làm đậm lên/ nối liền những nét đứt        

    #----------------------------------------------------------------------------------
    # if (show == True):
    #     cv2.imshow("Top_Hat",imgTopHat)
    #     cv2.imshow("Black_Hat",imgBlackHat)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    return plate_thre,plate_dilated
###############################################################################
def MainProcess(img_crop):
    if img_crop is None:
        return None            
    return findCharacter(img_crop)
