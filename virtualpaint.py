from email import header

import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

#########################
brushthickness =15
#####################

folderPath= "header"
myList = os.listdir(folderPath)
overLayList = []



for imgPath in myList:
    image =cv2.imread(f'{folderPath}/{imgPath}')
    overLayList.append(image)

header=overLayList[0]  #screen par by deafualt start hote he
drawcolor =(255,0,255)
xp , yp= 0 ,0

imgCanvas = np.zeros((720,1280,3),np.uint8) # fomed kyuki draw me har baar new frame hone ke baad previous drwaing nhi aa rahi thi

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)   #3 tells width 4 tells height
detector = htm.handDetector(min_detection_confidence=0.85)
while True:
    #1 import frames from video
    success , img = cap.read()
    cv2.flip(img,1)

    #2 find handlandmarks
    img=detector.findhands(img)
    lmList = detector.findposition(img , draw=False)

    if lmList != []:
        # print (lmList)

        #position of tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x3, y3 = lmList[4][1:]

        #3 checking which finger is up
        fingers = detector.fingerUp()
        # print(fingers)


        #4 if selection mode 2 fingers are up
        if fingers[0]==1 and fingers[1]==1 and fingers[2]==0 and fingers[3]==0:
            brushthickness=brushthickness+1
        if fingers[4]==1 and fingers[1]==0 and fingers[2]==0 and fingers[3]==0:
            brushthickness=brushthickness-1

        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            print("selection mode")

            #checking selection
            if y1 < 125 :
                if 250 <x1<450:
                    header=overLayList[0]
                    drawcolor = (255,0,255)

                elif 550 <x1 <750:
                    header=overLayList[1]
                    drawcolor = (0, 0, 255)

                elif 800<x1<950:
                    header=overLayList[2]
                    drawColor = (255, 0, 130)

                elif 1050<x1<1200:
                    header=overLayList[3]
                    drawcolor = (0, 0, 0)
                    brushthickness=50









        #5 if drawing 1 finger is up
        if fingers[1]==1 and fingers[2]==0 and fingers[3]==0 and fingers[4]==0:
            print("drawing mode")
            if xp==0 and yp==0:
                xp =x1
                yp =y1
            cv2.line(img,(xp,yp),(x1,y1),drawcolor,brushthickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
            xp =x1
            yp =y1




    #setting the header image in video ,,,frame set ho rahe hai img variable me and header img is saved in header
    img[0:125, 0:1280]=header


    cv2.imshow('img',img)
    cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(1)
