import cv2 as cv

uvwindowname = "UV tracks"
cv.namedWindow(uvwindowname)
trackbar_name = "Hour angle 12"
cv.createTrackbar(trackbar_name, uvwindowname, 0, 12, lambda x: x) # hour_angle.set)
# cv.imshow(uvwindow, uv_baselines)

cv.waitKey()

