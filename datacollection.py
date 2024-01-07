import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offsetmargin = 20
imgsize = 300
counter = 0
folder = "Data/thumbsup"



while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        # only for 1 hand
        hand = hands[0]
        # get bounding box info
        x, y, w, h = hand['bbox']

        ###############
        # creating from matrix
        imgwhite = np.ones((300, 300, 3), np.uint8) * 255
        ###############

        # starting h : end h , starting width: ending width
        imgcrop = img[y - offsetmargin:y + h + offsetmargin, x - offsetmargin:x + w + offsetmargin]

        # overlaying imagecrop on imagewhite
        # or putting imagecrop in imagewhite

        # starting point of height = 0
        #   ending point of height = hight of imgcrop
        # imgcropshape = imgcrop.shape
        # height = imgcropshape[0]
        # width = imgcropshape[1]
        # # overlaying
        # imgwhite[0:height, 0: width] = imgcrop

        aspectRatio = h / w

        if aspectRatio > 1:
            #     means height is greater than width
            constatntK = imgsize / h
            #     if  height is stretched by constant k then width should also be stretched accordingly
            wcal = math.ceil(constatntK * w)
            #                      new cal width  , height
            imgResize = cv2.resize(imgcrop, (wcal, imgsize))

            # will give height and width
            imgResizeShape = imgResize.shape
            # centering the image in white image
            wGap = math.ceil((300 - wcal) / 2)
            height = imgResizeShape[0]
            width = imgResizeShape[1]
            imgwhite[0:height, wGap:wcal + wGap] = imgResize

        else:
            #     means width is greater than width
            constatntK = imgsize / w
            #     if  width is stretched by constant k then height should also be stretched accordingly
            hcal = math.ceil(constatntK * h)
            #                         new cal width  , height
            imgResize = cv2.resize(imgcrop, (imgsize, hcal))

            # will give height and width
            imgResizeShape = imgResize.shape
            # centering the image in white image
            hGap = math.ceil((300 - hcal) / 2)
            height = imgResizeShape[0]
            width = imgResizeShape[1]
            imgwhite[hGap:hcal + hGap, 0:width] = imgResize

        cv2.imshow("imagecrop", imgcrop)
        cv2.imshow("imagewhite", imgwhite)

    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgwhite )
        print(counter)