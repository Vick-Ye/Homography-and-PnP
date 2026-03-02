import cv2
import numpy as np

import constants

vid = cv2.VideoCapture(0)

if not vid.isOpened():
    print('Failed to open camera')
    exit()

for i in range(constants.imageCount):
    print(i)
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        cv2.imshow('image', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.imwrite('calibrationImages/' + str(i) + '.png', frame)
            break

vid.release()
cv2.destroyAllWindows()

