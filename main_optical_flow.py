# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt

FPS = 60

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, FPS)

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Start streaming
pipeline.start(config)

# Take first frame and find corners in it
old_frame = np.asanyarray(
    pipeline.wait_for_frames().get_color_frame().get_data())
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # print(depth_colormap_dim, color_colormap_dim) (480, 640, 3) (480, 640, 3)

        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(
            colorizer.colorize(depth_frame).get_data())
        plt.imshow(colorized_depth)

        # Create alignment primitive with color as its target stream:
        align = rs.align(rs.stream.color)
        frameset = align.process(frames)

        # Update color and depth frames:
        aligned_depth_frame = frameset.get_depth_frame()

        colorized_depth = np.asanyarray(
            colorizer.colorize(aligned_depth_frame).get_data())
        mapped_frame, color_source = color_image, colorized_depth

        frame_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params)

        # Select good points (st==1 means select points that are tracked)
        # https://docs.opencv.org/master/dc/d6b/group__video__track.html#ga473e4b886d0bcc6b65831eb88ed93323
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        print(good_new, good_old, sep='"\n========')

        # Calculate motion vectors from good_new and good_old??
        if len(good_new) > 3:
            for i, point in enumerate(good_new):
                print("Distance of point {}".format(i),
                      aligned_depth_frame.get_distance(point[1], point[0]))
                cv2.circle(color_image, (int(point[1]), int(point[0])),
                           3, (0, 0, 255), thickness=1, lineType=8, shift=0)

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # If depth and color resolutions are different, resize color image to match depth image for display
        images = np.hstack((color_image, colorized_depth))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        if cv2.waitKey(0) == 27:
            break
finally:

    # Stop streaming
    pipeline.stop()
