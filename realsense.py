import cv2 as cv
import pyrealsense2 as rs
import numpy as np

# Configure the RealSense D435 camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the camera stream
pipeline.start(config)

# Initialize the chessboard pattern
chessboard_size = (6, 6)
obj_points = []
img_points = []
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

count = 0

try:
    while True:
        # Wait for a new frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # Convert the color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

        print(f"here2, {ret}")
        # If the corners are found, add them to the list of image points
        if ret == True:
            img_points.append(corners)
            obj_points.append(objp)

            # Draw the chessboard corners on the color image
            cv.drawChessboardCorners(color_image, chessboard_size, corners, ret)

            # Display the color image
            cv.imshow('RealSense D435 Camera Calibration', color_image)
            cv.waitKey(500)

            count += 1
            if count > 15:
                break;

finally:
    # Stop the camera stream
    pipeline.stop()

# Calibrate the camera using the image points and object points
ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Print the camera matrix
print('Camera Matrix:')
print(camera_matrix)
