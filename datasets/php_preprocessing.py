# -*- coding: utf-8 -*-
"""
Collection of Python utility functions for the pre-processing of Ultrasound data.
Usage of the Final GHL Integration Software by Philips is subject to the rights licensed to
Philips under the License Agreement, with an effective date of May 7, 2019, between GHL, as
assignee of IGS, and Philips. The Final GHL Integration Software is not to be transferred to
any third party and Philips accepts the Final GHL Integration Software “As-Is”.
Copyright 2020, Global Health Laboratories
"""

import os
import cv2
import numpy as np


"""
functions for basic data format changes
"""


def one_channel_to_three(one_channel_data):
    """
    Just a helper function to convert one channel image to three channel.
    Copying the same data to all the channels.
    :param one_channel_data:
    :return: one_channel_datax3 : shape = width x height x 3
    """
    width, height = one_channel_data.shape
    three_channel_data = np.zeros((width, height, 3), dtype=np.float32)

    for channel_idx in range(0, 3):
        three_channel_data[:, :, channel_idx] = one_channel_data

    return three_channel_data


"""
functions for cleaning up frames - masking out text, cropping out fan area
"""


def get_mask(frame, upper_left, lower_left, upper_right, lower_right, lower_middle, circle_center):

    """
    Use fan area boundary points to create a mask
    :param frame - the frame to which mask will be applied (2D numpy array)
    :param upper_left - upper left point of fan (x, y)
    :param upper_right - upper right point of fan
    :param lower_left - lower left point of fan
    :param lower_right - lower right point of fan
    :param lower_middle - lowest point containing data
    :param circle_center - The center of the two circles to which the arcs defining the top and bottom of the fan area belong
    :return: mask - binary 2D numpy array the shape of frame
    """

    [rows, cols] = np.meshgrid(np.arange(frame.shape[0]), np.arange(frame.shape[1]), indexing='ij')
    # get equation of line through upper_left and lower_left points
    im1 = rows > (upper_left[1] - lower_left[1]) / (upper_left[0] - lower_left[0]) * (cols - lower_left[0]) + \
          lower_left[1]
    im2 = rows > (upper_right[1] - lower_right[1]) / (upper_right[0] - lower_right[0]) * (cols - lower_right[0]) + \
          lower_right[1]
    d_squared = (cols - circle_center[0]) ** 2 + (rows - circle_center[1]) ** 2
    im3 = d_squared < (lower_middle[1] - circle_center[1]) ** 2
    r_squared = (circle_center[0] - upper_left[0]) ** 2 + (circle_center[1] - upper_left[1]) ** 2
    im4 = d_squared > r_squared
    mask = np.logical_and(np.logical_and(np.logical_and(im1, im2), im3), im4)
    return mask


def get_reference_points(video, frame_dim=0):
    """
    uses frame pixel intensities to get the coordinate of the upper right,
    center top, and center bottom points of the fan
    :param video: video to test
    :param frame_dim: indicates if video is channels_first or channels_last
    :return upper_right: x and y of upper right corner point;
    :return upper_middle: x and y of upper middle point;
    :return lower_middle: x and y of lower middle point
    """
    if frame_dim == 0:
        frame = video.mean(axis=0)  # As the data is arranged in the order : 60 x width x height x channels
    else:
        frame = video.mean(axis=-1)  # As the data is arranged in the order : width x height x channels x 60
    if len(frame.shape) > 2:
        frame = frame[:, :, 0]  # eliminate channels dimension
    frame = frame > 0
    # approximate midpoint
    midpoint = int(frame.shape[1] / 2)
    # take the right half of the image, and crop off the writing at the far right
    right = frame[:, midpoint:-350]
    colsum = right.sum(axis=1) > 0
    topind = np.nonzero(colsum)[0][0]
    rightind = np.nonzero(right[topind, :])[0][0]
    rightind = rightind + midpoint
    upper_right = [int(rightind), int(topind)]

    # get the x-coordinate of the upper left point
    leftind = np.nonzero(frame[topind, :midpoint])[0][-1]
    # recalculate the midpoint
    midpoint = int((rightind + leftind)/2)
    midind = np.nonzero(frame[:, midpoint])[0][0]
    upper_middle = [int(midpoint), int(midind)]

    # get coordinates of bottom depth marker to get bottom middle
    left = frame[:, :20]
    rowsum = left.sum(axis=1) > 0
    bottomind = np.nonzero(rowsum)[0][-1]
    lower_middle = [int(midpoint), int(bottomind)]

    return upper_right, upper_middle, lower_middle


def fan_coordinates_calculation(upper_right, upper_middle, lower_middle):
    """
    Use the upper upper middle, and lower middle points of the fan area to calculate the remaining points
    :param upper_right: x2 and y2 of upper right corner point;
    :param upper_middle: x1 and y1 of upper middle point;
    :param lower_middle: x4 and y4 of lower middle point; x4=x3
    :return: lower_left: x5 and y5 of lower L corner point
             lower_right: x6 and y6 of lower R corner point, y5=y6
             origin: beam origin, circle center
    """
    x2 = upper_right[0]
    y2 = upper_right[1]
    x1 = upper_middle[0]
    y1 = upper_middle[1]
    x4 = lower_middle[0]
    y4 = lower_middle[1]

    sin_theta = abs(x2 - x1) / ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    theta = np.arcsin(sin_theta)
    phi = np.pi - 2 * theta
    sin_phi = np.sin(phi)
    r1 = abs(x2 - x1) / sin_phi
    origin = [x1, y1 - r1]
    r2 = y4 - origin[1]
    x6 = x4 + r2 * sin_phi
    x5 = x4 - r2 * sin_phi
    y5 = y4 - r2 * sin_phi / np.tan(theta)
    lower_left = [int(x5), int(y5)]
    lower_right = [int(x6), int(y5)]

    return lower_left, lower_right, origin


def get_tight_crop_data(video_data):
    """
    This function returns dynamically cropped video. Which means after removing Ultrasound parameters on the sides.
    :param video_data: video data of shape num_frames x height x width x num_channels
    :return: Video data with the same dimention as video_data but with different resolution. :
    return data type is np.float32 where values are between 0 - 255.
    """
    upper_right, upper_middle, lower_middle = get_reference_points(video_data)
    lower_left, lower_right, circle_center = fan_coordinates_calculation(upper_right, upper_middle,
                                                                         lower_middle)
    upper_left = [2 * upper_middle[0] - upper_right[0] - 1, upper_right[1]]

    no_of_frames = video_data.shape[0]
    cropping_start_point = (upper_right[1], lower_left[0])  # y_min, x_min
    cropped_video_width = lower_right[0] - lower_left[0]
    cropped_video_height = lower_middle[1] - upper_right[1]
    cropped_video = np.zeros((no_of_frames, cropped_video_height, cropped_video_width, 3))

    for frame_idx in range(no_of_frames):
        if len(video_data.shape) == 4:  # if multiple color channels
            frame_data = video_data[frame_idx, :, :, 0]
        else:
            frame_data = video_data[frame_idx, :, :]

        mask = get_mask(frame_data, upper_left, lower_left, upper_right, lower_right, lower_middle, circle_center)
        frame = np.multiply(frame_data, mask)
        cropped_frame = frame[upper_right[1]:lower_middle[1], lower_left[0]:lower_right[0]]
        cropped_video[frame_idx, :, :, :] = one_channel_to_three(cropped_frame)

    return cropped_video, cropping_start_point


"""
function for classification preprocessing
"""


def preprocess_image(frame, input_size):
    frame = frame[:, :, :input_size[3]]

    resize_image = cv2.resize(frame, (input_size[2], input_size[1]))

    # handle differences in preprocessing for pediatric vs. adult
    # normalize image
    if input_size[3] == 1:
        tmp_image = (resize_image - np.mean(resize_image)) / np.std(resize_image)
        processed_image = np.expand_dims(tmp_image, axis=2)
    else:
        processed_image = resize_image / 255

    return processed_image


def preprocess_video(video_data, resize_shape, normalize_image):
    """
    To resize and scale 4D video data.
    :param video_data: no_frames x old_height x old_width x no_channels
    :param resize_shape: new_height x new_width
    :param normalize_image: boolean (if values 0-1 or 0-255)
    :return: 4 D data - no_frames x new_height x new_width x no_channels
    """
    no_of_frames = video_data.shape[0]
    resized_video_data = np.zeros((no_of_frames, resize_shape[0], resize_shape[1], 3), dtype=np.float32)

    for frame_idx in range(no_of_frames):
        frame_data = video_data[frame_idx, :, :, :]

        if normalize_image:
            frame_data = frame_data / 255.0
        resized_video_data[frame_idx] = cv2.resize(frame_data, (resize_shape[1], resize_shape[0]))

    return resized_video_data


"""
functions for determining pixels per cm depth in an image, based on which tablet and tablet orientation was used for 
acquisition
"""


def match_template(frame, templates):
    """
    determines which of a set of templates (one per orientation/device) is the best match to the image
    :param frame: a frame from a video
    :param templates: the set of possible templates
    :return: best: the index of best match
    """

    all_corr = np.zeros([len(templates), ])
    for ind, temp in enumerate(templates):
        if len(frame.shape) > 2:
            corrcoef = np.corrcoef(temp.flatten(), frame[:, :, 0].flatten())
        else:
            corrcoef = np.corrcoef(temp.flatten(), frame.flatten())
        all_corr[ind] = corrcoef[0, 1]
    # noinspection PyArgumentList
    if all_corr.max() < 0.1:
        print("no matching template found")
        return np.nan
    return np.argmax(all_corr)


def load_templates():
    s2_portrait = np.load(os.path.join(os.path.dirname(__file__), 'device_templates', 's2_portrait_template.npy'))
    s2_landscape = np.load(os.path.join(os.path.dirname(__file__), 'device_templates', 's2_landscape_template.npy'))
    s4_portrait = np.load(os.path.join(os.path.dirname(__file__), 'device_templates', 's4_portrait_template.npy'))
    s4_landscape = np.load(os.path.join(os.path.dirname(__file__), 'device_templates', 's4_landscape_template.npy'))

    template_names = ['s2_portrait', 's2_landscape', 's4_portrait', 's4_landscape']
    templates = [s2_portrait, s2_landscape, s4_portrait, s4_landscape]
    return templates, template_names


def load_video(video_path, n_channels=1, frame_dim=0, dtype=np.float32):
    """
    Reads in the video frame by frame, and stores it in an array
    :param video_path: full filename of the video
    :param n_channels: 1 for grayscale, 3 for color
    :param frame_dim:  0 for frames along the first dimension, -1 for frames along the last
    :param dtype: data type
    :return: video_data: a 3d numpy array (height by width by frames) containing the video (single color channel)
    """
    # check to see if video is there
    if not os.path.isfile(video_path):
        print("Video stream file not found.")
        return None

    # try opening video
    try:
        video_cap = cv2.VideoCapture(video_path)
    except:
        print("Error opening video stream file.")
        return None

    if not video_cap.isOpened():
        print("Error opening video stream file.")
        return None

    # Capture video frame
    ret, frame_data = video_cap.read()
    if not ret:
        print("Error opening video stream file.")
        return None
    # frame_data_rgb = frame[start_crop[0]:end_crop[0], start_crop[1]:end_crop[1], :]
    height, width, _ = frame_data.shape

    # store the frames
    if frame_dim == 0:
        video_data = np.zeros((0, height, width, n_channels), dtype=dtype)
    else:
        video_data = np.zeros((height, width, n_channels, 0), dtype=dtype)

    frame_idx = 0
    while video_cap.isOpened():

        if frame_idx > 0:
            ret, frame_data = video_cap.read()
            if ret is False:
                video_cap.release()
                break

        frame_idx += 1

        if n_channels == 1:
            frame_data = np.expand_dims(frame_data[:, :, 0], axis=2)
        if frame_dim == 0:
            video_data = np.append(video_data, np.expand_dims(frame_data, axis=0), axis=0)
        else:
            video_data = np.append(video_data, np.expand_dims(frame_data, axis=3), axis=3)

    # get rid of any singleton dimensions
    video_data = video_data.squeeze()
    video_cap.release()
    cv2.destroyAllWindows()
    return video_data

def get_pix_per_cm(frame, is_ped):
    """
     determines pixels per cm for the device/orientation used for acquisition.
     :param frame: frame image for testing
     :param is_ped: changes the acquisition parameters - currently l12 for lumify 12 cm depth and l7 for lumify 7 cm depth (ped)
     :return: pix_per_cm pixels per centimeter
     """
    templates, template_names = load_templates()

    if is_ped:
        pix_per_cm = 60
    else:
        # deal with possibilities that frames are in the first dimension or last dimension
        try:
            dev_ind = match_template(frame[:, :, 0], templates)
        # if the resolution of the frame is one that is not unaccounted for in the templates, default to 55 pix/cm
        except:
            try:
                dev_ind = match_template(frame[0, :, :], templates)
            except:
                dev_ind = np.nan

        if np.isnan(dev_ind):
            print('assuming 55 pix/cm')
            pix_per_cm = 55
        else:
            dev_ori = template_names[dev_ind]
            if 's2' in dev_ori:
                pix_per_cm = 55
            elif 'landscape' in dev_ori:
                pix_per_cm = 70
            else:
                pix_per_cm = 45

    return pix_per_cm
