import numpy as np
import cv2

import constants

import scaling
import calibration


cameraMatrix = calibration.cameraMatrix
distCoeffs = calibration.distCoeffs

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

siftKP = []
siftDesc = []
inputs = []
scaled = []
scaledKP = []
width = constants.width
height = constants.height

for i in range(12):
    inputs.append(cv2.imread('images/' + str(i+1) + '.png'))
    kp, desc = sift.detectAndCompute(cv2.cvtColor(inputs[i], cv2.COLOR_BGR2GRAY), None)
    siftKP.append(kp)
    siftDesc.append(desc)
    scaled.append(cv2.resize(inputs[i], (width, height)))

for i in range(12):
    for keyPts in siftKP:
        scaledPts = []
        for pts in keyPts:
            scaledPts.append(scaling.scale(pts.pt, inputs[i].shape, (width, height)))
        scaledKP.append(scaledPts)
