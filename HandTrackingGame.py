import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0  # previous time
cTime = 0  # current time
cap = cv2.VideoCapture(0)  # create the video object

# create our object of handDetector class
detector = htm.handDetector()

while True:
    success, img = cap.read()  # get the frame
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[4])

    # except detecting the hand track, try to perform a certain task (what exactly can do)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # ^ onWhere, what, pos, font, scale, color, thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)