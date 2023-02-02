import cv2 as cv
import numpy as np

trials = 20

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cap = cv.VideoCapture(0)
count = 1
while True:
    print("pat")
    imgret, img = cap.read()
    if not imgret:
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (7, 6), None)
    if not ret:
        continue
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)

    cv.drawChessboardCorners(img, (7, 6), corners2, ret)
    cv.imshow('img', img)

    print(f"got {count} frames")
    cv.waitKey()

    count += 1
    if count > trials:
        break


input("ready for calibration")

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

input("finished")

print(f"matrix: {mtx}")
print(f"dist: {dist}")
print(f"rvecs: {rvecs}")
print(f"tvecs: {tvecs}")


cv.destroyAllWindows()