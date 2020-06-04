import cv2
import numpy as np

#####################################################################################
show=False

###################################################################################
def findPlate(img):
    img_gray,img_dilated = Preprocess(img)
    img_height, img_width = img_dilated.shape 
    img_area=img_height*img_width
    #read image's size    

    contours,hierarchy = cv2.findContours(img_dilated,cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)
    #Tìm tất cả contour trong hình
        
    contours = sorted(contours, key = cv2.contourArea, reverse = True) [:10]
    #Sắp xếp và lấy ra 10 contour lớn nhất

    if show == True:
        img_draw_contour = np.copy(img)
        for c in contours:  
            img_draw_contour = cv2.drawContours(img_draw_contour,[c],-1,(255,0,0),2)

    for c in contours:
        peri = cv2.arcLength(c,True)      
        # Này là sai số khi xấp xỉ thành đa giác

        approx = cv2.approxPolyDP(c,0.04*peri,True) 
        #Xấp xỉ contour đó thành đa giác

        if  (img_area*0.01 < cv2.contourArea(approx)):         
            if len(approx) == 4 and (img_area*0.01 < cv2.contourArea(approx)):
            #Nếu đa giác xấp xỉ có cạnh =4 và diện tích vừa phải thì có thể đó là biển số xe
                x,y,w,h = cv2.boundingRect(approx)
                if 1.0 < w/h < 3.0:
                    img_show = img
                    img_show = cv2.rectangle(img_show,(x,y),(x+w,y+h),(0,255,0),1)
                    
                    if show == True:
                        img_draw_contour = cv2.drawContours(img_draw_contour,[c],-1,(0,255,0),2)
                        cv2.imshow("Find Plate",img_draw_contour)
                        cv2.waitKey(0)

                    return c
    return None
############################################################################################
def cropPlate( plate_contour , img):
    #
    cv2.destroyAllWindows()

    x,y,w,h = cv2.boundingRect(plate_contour) 
    #Lấy hình chữ nhật bao biển số xe
    top = y
    bot = y+h
    left = x
    right = x+w
                                                    
    cropped_copy=np.copy(img[top:bot,left:right] )                                      

    return cropped_copy
############################################################################################
def Preprocess(img):
    #Hàm này sẽ nhận tham số là 1 ảnh và trả về 2 ảnh lần lượt là ảnh xám và ảnh trắng đen 
    #________________________________________________________________________#
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    img_gray = img  
    if(show==True):
        cv2.imshow("Gray",img)
    #Chuyển từ ảnh màu thành ảnh xám
        
    img = cv2.bilateralFilter(img,11,17,17)  
    if(show==True):
        cv2.imshow("Remove_noise",img)
    #Lọc hình ảnh để giảm nhiễu    
        
    # img = cv2.GaussianBlur(img,(3,3),1)   
    # if(show==True):
    #     cv2.imshow("Blur",img)  


    # img = cv2.equalizeHist(img)  
    # if(show==True):
    #     cv2.imshow("Equalize_histogram",img)
    # #Cân bằng sáng hình ảnh  

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

    ret,img = cv2.threshold(img,120,255,cv2.THRESH_OTSU)
    if(show==True):
        cv2.imshow("Thresh",img)
    #Tạo ảnh trắng đen bằng cách sử dụng ngưỡng

    img = cv2.Canny(img,0,255)    
    if(show==True):
        cv2.imshow("Canny",img)
    #Tìm biên của hình ảnh bằng bộ lọc canny
        
    kernel_dilate = np.ones((1,1), np.uint8)
    img = cv2.dilate(img, kernel_dilate,iterations=1) 
    img_binary = img
    if(show==True):
        cv2.imshow("dilate",img)
    #Làm đậm biên của hình ảnh        

    #-----------------------------------------------------------------------#
    if (show==True):
        cv2.waitKey(0)
        cv2.destroyAllWindows()
   
    return img_gray,img_binary
###################################################################################
def MainProcess(img):
    approx = findPlate(img)
    if approx is None:
        print("No lincese plate is detected")
        return None
    else:
        crop = cropPlate(approx,img)
        return crop




