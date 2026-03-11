import cv2
import numpy as np

import constants

vid = cv2.VideoCapture('calibrationVid.avi')

if not vid.isOpened():
    print('Failed to open video')
    exit()

for i in range(constants.imageCount):
    print(i)
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        cv2.imshow('image', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.imwrite('calibrationImages/' + str(i) + '.png', frame)
            break

vid.release()
cv2.destroyAllWindows()

