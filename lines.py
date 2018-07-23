# created by Sijmen van der Willik
# 23/07/2018 14:45


def sane_lines(a, b):
    # curvatures should not differ more than 5x
    # do not check curvatures that are too close to straight
    if a.curvature < 6000 or b.curvature < 6000:
        if a.curvature < 0.1 * b.curvature or a.curvature > 10 * b.curvature:
            print("lines rejected: curvatures not similar ({:.3} and {:.3})".format(a.curvature, b.curvature))
            return False

    # check approx distance in pixels
    target_distance = 700
    target_margin = 150
    if np.abs(a.position - b.position) - target_distance > target_margin:
        print("lines rejected: distance is not approx {} pixels (margin: {})".format(target_distance, target_margin))
        return False
