import cv2 as cv
import numpy as np

img = cv.imread('/home/rohit/Pictures/simple.png', 0)
img_copy = img.copy
cv.imshow('img', img)
fast = cv.FastFeatureDetector()
kp = fast.detect(img, None)
img2 = cv.drawKeypoints(img, kp,  color=(0, 255, 0))
cv.imshow('img2', img2)
print(fast.getInt('threshold'))
fast.setBool('nonmaxSuppressing', 0)
kp = fast.detect(img, None)
img3 = cv.drawKeypoints(img, kp, color=(0, 0, 255))
cv.imshow('img3', img3)

cv.waitKey(0)
cv.destroyAllWindows()