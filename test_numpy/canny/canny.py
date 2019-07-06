import numpy as np
import cv2
img=cv2.imread('girl_mini.jpg',0)
def nothing(x):
    pass
cv2.namedWindow('res')
cv2.createTrackbar('min','res',0,25,nothing)
cv2.createTrackbar('max','res',0,25,nothing)
while(1):
    if cv2.waitKey(1)&0xFF==27: #esc退出
        break
    maxVal=cv2.getTrackbarPos('max','res')
    minVal=cv2.getTrackbarPos('min','res')
    canny=cv2.Canny(img,10*minVal,10*maxVal)
    cv2.imshow('res',canny)

cv2.destroyAllWindows()
