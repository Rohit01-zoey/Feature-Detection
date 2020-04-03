import cv2 as cv
import numpy as np

img = cv.imread('/home/rohit/Pictures/chess.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

corners = cv.goodFeaturesToTrack(gray, 1000, 0.01, 2)

corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv.circle(img, (x, y), 2, (0, 255, 0), 2)
cv.imshow('dst', img)
cv.waitKey(0)
cv.destroyAllWindows()