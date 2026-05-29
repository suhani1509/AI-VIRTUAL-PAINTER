from email import header

import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath= "header"
myList = os.listdir(folderPath)
overLayList = []

for imgPath in myList:
    image =cv2.imread(f'{folderPath}/{imgPath}')
    overLayList.append(image)

header=overLayList[0]  #screen par by deafualt start hote he

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
        print (lmList)

        #position of tip of index and middle finger
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    #3 checking which finger is up
    #4 if selection mode 2 fingers are up
    #5 if drawing 1 finger is up

    #setting the header image in video ,,,frame set ho rahe hai img variable me and header img is saved in header
    img[0:125, 0:1280]=header


    cv2.imshow('img',img)
    cv2.waitKey(1)
