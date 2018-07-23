# created by Sijmen van der Willik
# 21/07/2018 15:51

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import morph_img

# init all globally used variables

# get image_size
img_size = morph_img.get_img_size()

# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720  # meters per pixel in y dimension
xm_per_pix = 3.7/700  # meters per pixel in x dimension


def abs_sobel_thresh(input_img, thresh_min=5, thresh_max=100):
    """create binary image using sobel x

    :param input_img:
    :param thresh_min:
    :param thresh_max:
    :return:
    """
    # 1) Convert to grayscale
    gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)

    # 2) Take the derivative in x
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    # 3) Take the absolute value of the derivative or gradient
    sobel_abs = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled = np.uint8(sobel_abs * 255 / np.max(sobel_abs))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled)

    binary_output[(scaled >= thresh_min) & (scaled <= thresh_max)] = 1

    # 6) Return this mask as your binary_output image
    return binary_output


def build_histogram(input_img):
    """detect lane line pixel positions using a histogram

    :param input_img:
    :return:
    """
    # Use only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = input_img[input_img.shape[0] // 2:]

    # Sum across image pixels vertically
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def get_peaks(input_img):
    img_hist = build_histogram(input_img)
    # returns the peak for both the left and the right side
    half_size = len(img_hist) // 2
    return np.argmax(img_hist[:half_size]), np.argmax(img_hist[half_size:]) + half_size


def find_lane_pixels(binary_warped):
    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current, rightx_current = get_peaks(binary_warped)

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        # Find the four boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = np.intersect1d(
            np.where((nonzerox >= win_xleft_low) &
                     (nonzerox < win_xleft_high)),
            np.where((nonzeroy >= win_y_low) &
                     (nonzeroy < win_y_high))
        )

        good_right_inds = np.intersect1d(
            np.where((nonzerox >= win_xright_low) &
                     (nonzerox < win_xright_high)),
            np.where((nonzeroy >= win_y_low) &
                     (nonzeroy < win_y_high))
        )
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if pixels found > minpix pixels, recenter next window
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))

        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    return left_fit, right_fit, left_fitx, right_fitx, ploty


def measure_curvature_real(left_fit_cr, right_fit_cr):
    """
    Calculates the curvature of polynomial functions in meters.
    """

    # Define y-value where we want radius of curvature
    # Maximum y-val is bottom of image
    y_eval = img_size[1]

    # R_curve (radius of curvature) calculation
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def position_of_vehicle(left_line, right_line):
    """Calculates the position of the vehicle in m from the center
    negative is left, positive is right
    """

    pix_from_center = img_size[0] // 2 - (right_line + left_line) // 2

    return pix_from_center * xm_per_pix


def draw_lane_on_img(img, left_fitx, right_fitx, ploty):
    """ draw the detected lane on the source image

    :param img:
    :param left_fitx:
    :param right_fitx:
    :param ploty:
    :return:
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros(img.shape[:2]).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = morph_img.warp_from_birds_eye(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result


def add_info_to_img(in_img, curv, pos):
    output_img = np.copy(in_img)
    curv_pos = (10, 80)
    pos_pos = (10, 160)

    cv2.putText(output_img, "Curvature: {:.2}".format(curv),
                curv_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.putText(output_img, "Position from center: {:.2}m".format(pos),
                pos_pos, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    return output_img


def pipeline(input_img, last_fit=None):
    """entire image processing pipeline to go from source image to annotated output image

    :param input_img:
    :return:
    """
    # undistort image
    undist = morph_img.undistort_image(input_img)

    # threshold binary
    binary_threshold = abs_sobel_thresh(undist)

    # bird's eye view
    birds_eye_view = morph_img.warp_to_birds_eye(binary_threshold)

    # fit polynomial
    left_fit, right_fit, left_fitx, right_fitx, ploty = fit_polynomial(birds_eye_view)

    # calculate lane curvature
    l_curve, r_curve = measure_curvature_real(left_fit, right_fit)

    # calculate position of vehicle
    left_peak, right_peak = get_peaks(birds_eye_view)
    vehicle_pos = position_of_vehicle(left_peak,
                                      right_peak)  # TODO fix this to take most recent left/right instead o hist

    # draw the detected lane on the source image
    lane_img = draw_lane_on_img(undist, left_fitx, right_fitx, ploty)

    # annotate the image
    final_img = add_info_to_img(lane_img, (l_curve + r_curve) / 2, vehicle_pos)

    return final_img
