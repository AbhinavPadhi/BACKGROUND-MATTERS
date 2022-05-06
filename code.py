import cv2 
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*"XVID")
outputFile = cv2.VideoWriter("output.avi" , fourcc , 20.0 , (640 , 480))
webcam = cv2.VideoCapture(0)
time.sleep(2)
bg = 0

for a in range(60):
    ret,bg = webcam.read()

bg = np.flip(bg , axis = 1)

while(webcam.isOpened()):
    ret,img = webcam.read()
    if not ret :
        break

    img = np.flip(img, axis=1)

    hsv = cv2.cvtColor(img , cv2.COLOR_BGR2HSV)

    lowerBlack = np.array([20,15,15])
    upperBlack = np.array([90,90,90])
    mask1 = cv2.inRange(hsv,lowerBlack , upperBlack)


    lowerBlack = np.array([30,50,50])
    upperBlack = np.array([100,110,120])
    mask2 = cv2.inRange(hsv,lowerBlack , upperBlack)

    mask1 = mask1+mask2

    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_OPEN , np.ones((3,3),np.uint8))
    mask1 = cv2.morphologyEx(mask1,cv2.MORPH_DILATE , np.ones((3,3),np.uint8))

    mask2 = cv2.bitwise_not(mask1)

    result1 = cv2.bitwise_and(img,img,mask = mask2)

    result2 = cv2.bitwise_and(bg,bg,mask = mask1)

    finalOutput = cv2.addWeighted(result1,1,result2,1,0)

    outputFile.write(finalOutput)

    cv2.imshow("magic" , finalOutput)

    cv2.waitKey(1)

webcam.release()
cv2.destroyAllWindows()    
