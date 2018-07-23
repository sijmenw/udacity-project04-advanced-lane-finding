# created by Sijmen van der Willik
# 23/07/2018 15:50

import numpy as np


class Fit:
    """This class holds all necessary variables needed to describe a lane line fit

    Helper functions are included
    """
    def __init__(self):
        self.age = 0
        self.left_fit = None
        self.right_fit = None
        self.left_curvature = None
        self.right_curvature = None
        self.vehicle_pos = None
        self.left_peak = None
        self.right_peak = None
        self.search_margin = 50  # in pixels

    def set_fits(self, l_fit, r_fit):
        self.left_fit = l_fit
        self.right_fit = r_fit

    def set_curves(self, l_curve, r_curve):
        self.left_curvature = l_curve
        self.right_curvature = r_curve

    def get_peaks(self):
        return self.left_peak, self.right_peak

    def set_peaks(self, left_peak, right_peak):
        self.left_peak = left_peak
        self.right_peak = right_peak

    def is_sane(self):
        # curvatures should not differ more than 5x
        # do not check curvatures that are too close to straight
        if self.left_curvature < 6000 or self.right_curvature < 6000:
            if self.left_curvature < 0.2 * self.right_curvature\
                    or self.left_curvature > 5 * self.right_curvature:
                print("lines rejected: curvatures not similar ({:.3} and {:.3})".format(
                    self.left_curvature, self.right_curvature))
                return False

        # check approx distance in pixels
        target_distance = 700
        target_margin = 150
        if np.abs(self.left_peak - self.right_peak) - target_distance > target_margin:
            print(
                "lines rejected: distance is not approx {} pixels (margin: {})".format(
                    target_distance, target_margin))
            return False

        return True
