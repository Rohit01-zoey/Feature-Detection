import cv2 as cv
import numpy as np

img = cv.imread('/home/rohit/Pictures/chess.png')
cv.imshow('image', img)

imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

imgray = np.float32(imgray)
dst = cv.cornerHarris(imgray, 2, 3, 0.04)

dst = cv.dilate(dst, None)

img[ dst > 0.01 * dst.max()] = [0, 0, 255]

cv.imshow('dst', img)

cv.waitKey(0)
cv.destroyAllWindows()