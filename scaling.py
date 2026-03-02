import numpy as np
import cv2

import constants

def scale(pt, initial, transformed):
    return np.array([pt[0] * transformed[0]/initial[0], pt[1] * transformed[1]/initial[1]])

def test():
    sift = cv2.xfeatures2d.SIFT_create()
    siftKP = []
    siftDesc = []
    inputs = []
    scaled = []
    scaledKP = []
    for i in range(12):
        inputs.append(cv2.imread('images/' + str(i) + '.png', cv2.IMREAD_GRAYSCALE))
        kp, desc = sift.detectAndCompute(inputs[i], None)
        siftKP.append(kp)
        siftDesc.append(desc)
        scaled.append(cv2.resize(inputs[i], (constants.width, constants.height)))

    for i in range(12):
        for keyPts in siftKP:
            scaledPts = []
            for pts in keyPts:
                scaledPts.append(scale(pts.pt, inputs[i].shape, (constants.width, constants.height)))
            scaledKP.append(scaledPts)

    for i in range(12):
        scaled[i] = cv2.cvtColor(scaled[i], cv2.COLOR_GRAY2BGR)
        for pt in scaledKP[i]:
            cv2.circle(scaled[i], (int(pt[0]), int(pt[1])), radius=3, color=(0, 255, 0), thickness=-1)

        cv2.imshow(str(i), scaled[i])

        if cv2.waitKey(0) and 0xFF == ord('q'):
            cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
