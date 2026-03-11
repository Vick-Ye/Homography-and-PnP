import cv2
import numpy as np

import constants

numRows = constants.numRows
numCols = constants.numCols

# Create grid
gridPt = np.zeros((numRows*numCols,3), np.float32)
gridPt[:,:2] = np.mgrid[0:numRows,0:numCols].T.reshape(-1,2)
gridPt *= constants.squareSizeMM

# Corner match criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

calibrationImages = []
objPoints = []  # 3D
imgPoints = []  # 2D

# Load images
for i in range(constants.imageCount):
    calibrationImages.append(cv2.imread('calibrationImages/' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))

# Process corners
for img in calibrationImages:
    ret, corners = cv2.findChessboardCorners(img, (numRows, numCols), None)
    if ret == True:
        corners_ = cv2.cornerSubPix(img, corners, (1, 1), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, (numRows, numCols), corners_, ret) 
        imgPoints.append(corners_)
        objPoints.append(gridPt)
        if __name__ == '__main__':
            cv2.imshow('match', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()



ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objPoints, imgPoints, calibrationImages[0].shape[::-1], None, None
)


maxError = 0
for i in range(len(objPoints)):
    projPoints, _ = cv2.projectPoints(objPoints[i], rvecs[i], tvecs[i], cameraMatrix, distCoeffs)
    error = cv2.norm(imgPoints[i], projPoints, cv2.NORM_L2) / len(projPoints)
    if error > maxError:
        maxError = error

if __name__ == '__main__':
    print('calibration results')
<<<<<<< Updated upstream
    print('matrix: ')
    print(cameraMatrix)
    print('distortion: ')
    print(distCoeffs)
    print('max projection error: ')
    print(maxError)
    for img in calibrationImages:
        h, w = img.shape[:2]
        newcam, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 0, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, distCoeffs, None, newcam, (w, h), 5)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]
        cv2.imshow('undistorted', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #img = cv2.undistort(img, cameraMatrix, distCoeffs, None, newcam)
        #x, y, w, h = roi
        #img = img[y:y+h, x:x+w]
        #cv2.imshow('undistorted', img)
        #cv2.waitKey(0)
=======
    #print('matrix: ' + cameraMatrix)
    print(cameraMatrix)
    print('distortion: ' + distCoeffs)
    print('max projection error: ' + maxError)
>>>>>>> Stashed changes
