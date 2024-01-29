#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""draw_functions: Define drawing functions.
-------------------------------------------------------------------------------



Copyright (c) 2024 N.Takayama @ TRaD <takayaman@takayama-rado.com>
-------------------------------------------------------------------------------
"""

# Standard modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import io
from base64 import b64encode
from IPython.display import HTML

# Third party's modules
import cv2

import numpy as np

# Local modules


# Execution settings
VERSION = u"%(prog)s dev"

# Input data directories
DIR_INPUT = None
# Output data directories
DIR_OUTPUT = None


def showvideo_on_browser(video_path, video_width=500, video_height=500):
    """Show video in a browser.

    # Args:
      - video_path: The path to a video file.
      - video_width: The width of video object in a browser.
      - video_height: The height of video object in a browser.

    # Returns:
      - The HTML object containing video data.
    """
    video = io.open(video_path, 'r+b').read()
    encoded = b64encode(video)
    decoded = encoded.decode("ascii")
    data = f"<video width={video_width} height={video_height} controls>" \
        + f'<source src="data:video/mp4;base64,{decoded}"' \
        + 'type="video/mp4" /></video>'
    return (HTML(data=data))


def draw_landmarks(draw, landmarks, connections,
                   pt_color=(0, 0, 0), line_color=(0, 0, 0),
                   pt_size=3, line_size=3,
                   do_remove_error=True):
    for i, line in enumerate(connections):
        p0 = landmarks[line[0], :2]
        p1 = landmarks[line[1], :2]
        if do_remove_error:
            if np.isnan(p0).any() or np.isnan(p1).any():
                continue
            if (p0==0).any() or (p1==0).any():
                continue
        p0 = (int(p0[0]), int(p0[1]))
        p1 = (int(p1[0]), int(p1[1]))
        cv2.line(draw, p0, p1, line_color, line_size)
        cv2.circle(draw, p0, pt_size, pt_color, -1)
        cv2.circle(draw, p1, pt_size, pt_color, -1)
    return draw


def draw_tracking(trackdata, outpath,
                  face_connections,
                  lhand_connections,
                  pose_connections,
                  rhand_connections,
                  channel_first=False,
                  width=500, height=500,
                  fps=30):
    if len(trackdata.shape) == 4:
        trackdata = trackdata[0]
    if channel_first:
        # `[C, T, J] -> [T, J, C]`
        trackdata = np.transpose(trackdata, [1, 2, 0])

    trackdata[np.isnan(trackdata)] = 0
    # If base width is smaller than 10, the trackdata should be normalized.
    # So, we project them into image plane.
    if np.abs(trackdata[:, :, 0].max() - trackdata[:, :, 0].min()) < 10:
        trackdata[:, :, 0] *= width
        trackdata[:, :, 1] *= height
    xmin_orig = trackdata[:, :, 0].min()
    ymin_orig = trackdata[:, :, 1].min()
    xmax_orig = trackdata[:, :, 0].max()
    ymax_orig = trackdata[:, :, 1].max()
    xmin_proj = int(xmin_orig)
    ymin_proj = int(ymin_orig)
    xmax_proj = int(xmax_orig)
    ymax_proj = int(ymax_orig)

    # Add offset.
    offset_x = 0
    offset_y = 0
    if xmin_proj < 0:
        offset_x = -xmin_proj
        trackdata[:, :, 0] += offset_x
    if ymin_proj < 0:
        offset_y = -ymin_proj
        trackdata[:, :, 1] += offset_y

    ywin = int(max(ymax_proj + offset_y, height + offset_y))
    xwin = int(max(xmax_proj + offset_x, width + offset_x))

    print("Window size:", xwin, ywin)
    print("Offsets:", offset_x, offset_y)

    writer = cv2.VideoWriter(str(outpath), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (xwin, ywin))
    if writer.isOpened() is False:
        print("VideoWriter failed to open.")
        if writer is not None:
            writer.release()
        return

    for frame in trackdata:
        image = np.full([height, width, 3], 255, dtype=np.uint8)

        # Draw
        draw = np.full([int(ywin), int(xwin), 3], 64, dtype=np.uint8)
        draw[offset_y: offset_y+height, offset_x: offset_x+width, :] = image

        if pose_connections is not None:
            draw = draw_landmarks(draw, frame, pose_connections, [0, 0, 255], [0, 0, 255])
        if face_connections is not None:
            draw = draw_landmarks(draw, frame, face_connections, [0, 255, 0], [0, 255, 0])
        if lhand_connections is not None:
            draw = draw_landmarks(draw, frame, lhand_connections, [255, 0, 0], [255, 0, 0])
        if rhand_connections is not None:
            draw = draw_landmarks(draw, frame, rhand_connections, [255, 0, 255], [255, 0, 255])
        writer.write(draw)
    writer.release()


# --- Execution --------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
