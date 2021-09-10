import cv2 as cv
import time
import os
import HandTrackingModule as htm

wCam, hCam = 640, 480
cap = cv.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = "Images"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

print(len(overlayList))
pTime = 0

detector = htm.HandDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)

    if len(lmList) != 0:
        fingers = []
        # Thumb
        # if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]: #for left hand
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]: #for right hand
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]

        cv.rectangle(img, (20,250), (190,450), (0,255,0),cv.FILLED)
        cv.putText(img, str(totalFingers),(55,400), cv.FONT_HERSHEY_PLAIN,10,(255,0,0),10)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, f'FPS: {int(fps)}', (400, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv.imshow('Image', img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
