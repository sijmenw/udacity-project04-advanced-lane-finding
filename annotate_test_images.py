# created by Sijmen van der Willik
# 23/07/2018 15:28

import os
import cv2

import lane_detect


src_dir = "test_images"
target_dir = "output_images"

for img_name in os.listdir(src_dir):
    img_path = os.path.join(src_dir, img_name)
    print("annotating: {}".format(img_path))

    input_img = cv2.imread(img_path)

    result_img = lane_detect.pipeline(input_img)

    cv2.imwrite(os.path.join(target_dir, "out_" + img_name), result_img)
