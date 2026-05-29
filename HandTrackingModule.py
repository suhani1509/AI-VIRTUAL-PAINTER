import cv2
import mediapipe as mp

import time




class handDetector():
    def __init__(self,mode  = False,
                 max_num_hands: int = 4,
                 model_complexity: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) :
        self.mode = mode
        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.tips=[4,8,12,16,20]

        # default code
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.max_num_hands,self.model_complexity, self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findhands(self , img , draw=True):
         imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
         self.results = self.hands.process(imgRGB)
            # print(results.multi_hand_landmarks)
         if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                 if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
         return img

    def findposition(self, img, handNo=0, draw=True):
        self.lmList = []

        # PEHLE CHECK KARO: Kya MediaPipe ko koi haath mila?
        if self.results.multi_hand_landmarks:

            # Agar haath mila, tabhi loop chalega aur code crash nahi hoga
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), cv2.FILLED)

        return self.lmList


    def  fingerUp(self):
        fingers=[]

        # for thumb
        if self.lmList[self.tips[0]][1] > self.lmList[self.tips[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #for fingers
        for id in range(1,5):
            if self.lmList[self.tips[id]][2] < self.lmList[self.tips[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers





def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector() #forming object of class



    ctime = 0
    ptime = 0

    while (True):
        success, img = cap.read()
        img=detector.findhands(img)
        lmList= detector.findposition(img)
        if len(lmList) != 0:
            print(lmList[4])


        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        cv2.waitKey(1)










if __name__ == '__main__':
    main()