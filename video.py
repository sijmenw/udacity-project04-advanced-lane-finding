# created by Sijmen van der Willik
# 23/07/2018 16:05

from moviepy.editor import VideoFileClip
from IPython.display import HTML

import lane_detect

clipped_vid = VideoFileClip("project_video.mp4")

annotated_clip = clipped_vid.fl_image(lane_detect.pipeline)  # NOTE: this function expects color images!!
annotated_clip.write_videofile("output_vid.mp4", audio=False)
