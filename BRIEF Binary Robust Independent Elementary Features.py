import cv2 as cv
import numpy as np

img = cv.imread('/home/rohit/Pictures/simple.png')

star = cv.ORB()

kp = star.detect(img, None)
kp, des = star.compute(img, kp)
img2 = cv.drawKeypoints(img, kp, color=(0, 255, 0), flags=0)
cv.imshow('img2', img2)
cv.waitKey(0)
cv.destroyAllWindows()
