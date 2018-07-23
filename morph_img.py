# created by Sijmen van der Willik
# 23/07/2018 14:23

# camera calibration
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# get image_size
img_path = os.path.join("test_images", os.listdir("test_images")[0])
ex_img = cv2.imread(img_path)
img_size = ex_img.shape[1::-1]


def get_obj_img_points(nx=9, ny=6, src_dir="camera_cal"):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    image_list = [os.path.join(src_dir, x) for x in os.listdir(src_dir)]

    for image_path in image_list:
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    return objpoints, imgpoints


def get_calibration(src_dir="camera_cal"):
    objpoints, imgpoints = get_obj_img_points(src_dir=src_dir)

    # return the calibration values:
    return cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)


ret, mtx, dist, rvecs, tvecs = get_calibration()


def undistort_image(img):
    return cv2.undistort(ex_img, mtx, dist, None, mtx)


def get_M_and_minv():
    src_coords = np.array([
        [540, 490],
        [748, 490],
        [260, 680],
        [1045, 680]
    ], dtype=np.float32)
    dst_coords = np.array([
        [250, 200],
        [1030, 200],
        [250, 700],
        [1030, 700]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_coords, dst_coords)
    Minv = cv2.getPerspectiveTransform(dst_coords, src_coords)

    return M, Minv


# get transformation matrices
M, Minv = get_M_and_minv()


def warp_to_birds_eye(img):
    return cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


def warp_from_birds_eye(img):
    return cv2.warpPerspective(img, Minv, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


