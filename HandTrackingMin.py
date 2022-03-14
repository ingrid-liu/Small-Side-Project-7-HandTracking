import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands = 2, model_complexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon

        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity,
                                        self.detectionCon, self.trackCon)


# create the video object
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()       # create an object called hands; skip setting static_image_mode for now
mpDraw = mp.solutions.drawing_utils

# frame rates
pTime = 0   # previous time
cTime = 0   # current time

while True:
    success, img = cap.read()       # get the frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # ^ the hands object only uses RGB images, we need to convert it first
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # get the info of the hand
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                # lm contains x, y coordinates, we will use it to find the landmark on the hand
                # id is a decimal, it's the ratio of the width and the height
                # --> to get the pixel value (height, width, channels of the images
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # position of the center
                # print(cx, cy)       # --> give us the all 21 values (0~20)
                print(id, cx, cy)
                if id == 4 or id == 8:     # 0 is the twist, 4 is thumb point
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)     # single hand e.g. Hand #1

    # except detecting the hand track, try to perform a certain task (what exactly can do)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime


    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    # ^ onWhere, what, pos, font, scale, color, thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)

