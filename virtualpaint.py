import math
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
saved = False
# saveMessageTime=0
thickness_locked = True


for imgPath in myList:
    image =cv2.imread(f'{folderPath}/{imgPath}')
    overLayList.append(image)

header=overLayList[0]  #screen par by deafualt start hote he
drawcolor =(119 , 0, 200)
xp , yp= 0 ,0

imgCanvas = np.zeros((720,1280,3),np.uint8) # fomed kyuki draw me har baar new frame hone ke baad previous drwaing nhi aa rahi thi

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)   #3 tells width 4 tells height
detector = htm.handDetector(min_detection_confidence=0.85)
while True:
    #1 import frames from video
    success , img = cap.read()
    if not success:
        break

    img=cv2.flip(img,1)

    #2 find handlandmarks
    img=detector.findhands(img)
    lmList = detector.findposition(img , draw=False)

    if lmList != []:
        # print (lmList)

        #position of tip of index and middle finger
        x1, y1 = lmList[8][1:]   #index
        x2, y2 = lmList[12][1:]  #middle
        x3, y3 = lmList[4][1:]  #thumb
        x4 , y4 =lmList[20][1:]   #pinky

        #3 checking which finger is up
        fingers = detector.fingerUp()
        print(fingers)


        #clean canvas
        if fingers==[1,1,1,1,1]:
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)
            print("Canvas Cleared!")
            xp=0
            yp=0





        #save image
        elif fingers == [0, 1, 1, 1, 0]:
              if not saved:
                cv2.imwrite("Drawing.png", imgCanvas)
                print("Image Saved")
                saved = True
              xp=0
              yp=0

        # 4 if selection mode 2 fingers are up
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0:
            print("selection mode")
            xp=0
            yp=0

            #checking selection
            if y1 < 125 :
                if 250 <x1<450:
                    header=overLayList[0]
                    drawcolor = (119 , 0, 200)

                elif 550 <x1 <750:
                    header=overLayList[1]
                    drawcolor = (0, 0, 255)

                elif 800<x1<950:
                    header=overLayList[2]
                    drawcolor = (255,0,255)

                elif 1050<x1<1200:
                    header=overLayList[3]
                    drawcolor = (0, 0, 0)
                    brushthickness=50


        #5 if drawing 1 finger is up and dynamic thickness
        elif fingers == [1, 1, 0, 0, 0]:
            distance = math.hypot(x1 - x3, y1 - y3)

            # Jab distance 30 se kam ho (ungliyan chipki ho), tabhi thickness lock/unlock ho
            if distance < 30:
                thickness_locked = False  # Unlock ho gaya, ab distance badha kar size change karo
                print("Thickness Unlocked - Adjusting...")

            if not thickness_locked:
                brushthickness = int(np.interp(distance, [30, 180], [5, 40]))
                brushthickness = min(40, brushthickness)
            xp , yp =0 ,0

            # ---- DRAWING MODE (Only Index Up) ----
            # Jaise hi sirf index up ho, thickness ko turant lock kar do
        elif fingers == [0,1,0,0,0] :

            if not thickness_locked:
                thickness_locked = True
                print("Drawing Mode - Thickness Locked!")

            if xp == 0 and yp == 0:
                xp = x1
                yp = y1

            cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
            xp = x1
            yp = y1

            # Agar koi aur gesture hai, toh drawing coordinates reset karein
        else:
            xp, yp = 0, 0

        if fingers != [0, 1, 1, 1, 0]:
            saved = False






    #next 2 lines me canvas me jaha draw kara hai wo 50 above hoga usse black kar dega and rest 50
    # below usse background smjh kar white kar dega and usse imgInv me save kar dega
    img_Gray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _ , imgInv =cv2.threshold(img_Gray,50,255,cv2.THRESH_BINARY_INV)

    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # original image aur inverted mask ka and operation kar raha hai . kyuki maske me drwaing wala hissa black 0 tha
    # toh original me image me thik use jagah canvas me jaha black hota and karke 0 kar dega and black bhole bana deta\
    # hai and baaki hissa bilkul waisa he rahega
    img = cv2.bitwise_and(img,imgInv)
    img =cv2.bitwise_or(img,imgCanvas)


    #setting the header image in video ,,,frame set ho rahe hai img variable me and header img is saved in header
    img[0:125, 0:1280]=header

    # img =cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    img = cv2.addWeighted(img, 1, imgCanvas, 0.2, 0)
    cv2.putText(img,
                f'Thickness: {brushthickness}',
                (900, 680),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2)

    cv2.imshow('img',img)
    cv2.imshow('Canvas', imgCanvas)
    cv2.waitKey(1)

