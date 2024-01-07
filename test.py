import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("model_9_classes/keras_model.h5", "model_9_classes/labels.txt")

offsetmargin = 20
imgsize = 300
counter = 0
folder = "Data/C"

labels = ["A", "B", "C", "D", "E", "F", "love", "rock", "thumbsup"]


# There are 21 hand landmarks, each composed of x, y and z coordinates.
# The x and y coordinates are normalized to [0.0, 1.0] by the image width and height, respectively.
# The z coordinate represents the landmark depth, with the depth at the wrist being the origin.
# The smaller the value, the closer the landmark is to the camera.
# The magnitude of z uses roughly the same scale as x.


while True:
    success, img = cap.read()
    imgOutput = img.copy()
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
            prediction, index = classifier.getPrediction(imgwhite, draw=False)
            print(prediction, index)


        else:
            #     means width is greater than width
            constantK = imgsize / w
            #     if  width is stretched by constant k then height should also be stretched accordingly
            hcal = math.ceil(constantK * h)
            #                         new cal width  , height
            imgResize = cv2.resize(imgcrop, (imgsize, hcal))

            # will give height and width
            imgResizeShape = imgResize.shape
            # centering the image in white image
            hGap = math.ceil((300 - hcal) / 2)
            height = imgResizeShape[0]
            width = imgResizeShape[1]
            imgwhite[hGap:hcal + hGap, 0:width] = imgResize
            prediction, index = classifier.getPrediction(imgwhite, draw=False)

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
        cv2.rectangle(imgOutput, (x - offsetmargin, y - offsetmargin),
                      (x + w + offsetmargin, y + h + offsetmargin), (255, 0, 255), 4)

        # cv2.imshow("imagecrop", imgcrop)
        cv2.imshow("imagewhite", imgwhite)

    cv2.imshow("image", imgOutput)
    cv2.waitKey(1)
