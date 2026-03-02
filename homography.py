import cv2
import numpy as np

import constants
import preprocess

cameraMatrix = preprocess.cameraMatrix
dist = preprocess.distCoeffs
sift = preprocess.sift
bf = preprocess.bf

siftKP = preprocess.siftKP
siftDesc = preprocess.siftDesc
inputs = preprocess.inputs
scaledKP = preprocess.scaledKP
width = constants.width
height = constants.height

def detectSIFT(frame):
    kp, desc = sift.detectAndCompute(frame, None) # https://amroamroamro.github.io/mexopencv/matlab/cv.SIFT.detectAndCompute.html
    best = 0
    bestID = -1
    bestMatches = False
    H = None
    if desc is None:
        return False
    for i in range(constants.templateCount):
        ratio = 0.5
        matches = bf.knnMatch(siftDesc[i], desc, k=2) # https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html

        betterMatches = []
        if(len(matches) == 0 or len(matches[0]) != 2):
            continue

        for m, n in matches:
            if m.distance / n.distance < ratio:
                betterMatches.append(m)

        if len(betterMatches) < 4:
            continue

        sourcePts = np.float32([ scaledKP[i][m.queryIdx] for m in betterMatches]).reshape(-1,1,2)
        destPts = np.float32([ kp[m.trainIdx].pt for m in betterMatches]).reshape(-1,1,2)

        M, mask = cv2.findHomography(sourcePts, destPts, cv2.USAC_DEFAULT, 1.0)
        if M is None:
            continue

        matchesMask = mask.ravel().tolist()
        validCount = 0
        validMatches = []
        for j in range(len(matchesMask)):
            if matchesMask[j]:
                validMatches.append(betterMatches[j])
        if len(validMatches) > best:
            best = len(validMatches)
            bestID = i
            bestMatches = validMatches
            H = M
    if best != 0:
        return [bestID, bestMatches, kp, H, sourcePts, destPts]
    else:
        return [bestMatches]

def getHomography(H):
    w = constants.width
    h = constants.height
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,H)
    return dst

def getPose(dst):
    w = constants.width
    h = constants.height
    objPts = np.float32([ [0-w/2,0-h/2, 0],[0-w/2,h-1-h/2, 0],[w-1-w/2,h-1-h/2, 0],[w-1-w/2,0-h/2, 0] ])
    imgPts = dst
    success, rvec, tvec = cv2.solvePnP(objPts, imgPts, cameraMatrix, dist, useExtrinsicGuess = False, flags = cv2.SOLVEPNP_IPPE)
    return [success, rvec, tvec]

def PNP(obj, img):
    w = constants.width
    h = constants.height
    translatedObj = []
    for pt in obj:
        translatedObj.append([pt[0][0]-w/2, pt[0][1]-h/2, 0])

    success, rvec, tvec, inlier = cv2.solvePnPRansac(np.float32(translatedObj), img, cameraMatrix, dist, iterationsCount=1000, reprojectionError=0.5, confidence=0.99, useExtrinsicGuess = False, flags = cv2.SOLVEPNP_IPPE)
    return [success, rvec, tvec]


def main():
    vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        print('Failed to open camera')
        exit()

    while True:
        ret, frame = vid.read()
        if not ret:
            continue

        matchResult = detectSIFT(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        if len(matchResult) != 1:
            matchID, matches, matchKP, H, obj, img = matchResult
            dst = getHomography(H)
            frame = cv2.polylines(frame, [np.int32(dst)],True,255,3, cv2.LINE_AA)
            frame = cv2.drawMatches(inputs[matchID], siftKP[matchID], frame, matchKP, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            print(getPose(dst)[2])
            print(PNP(obj, img)[2])



        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
