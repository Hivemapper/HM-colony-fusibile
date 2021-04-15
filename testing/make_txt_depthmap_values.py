#!/usr/bin/env python
"""
Copyright 2021, Taylor Dahlke
Output a list of triplets of (ix,iy,depthVal) to text file format.
Based on python script from:
https://github.com/YoYo000/MVSNet/blob/4c4aa5e2336a214e4bde2de31c9a46f55a8150c5/mvsnet/depthfusion.py
"""

from __future__ import print_function

import argparse
import os
import time
import glob
import random
import math
import re
import sys
import shutil
from struct import *

import cv2
import numpy as np

import pylab as plt
from preprocess import *

import imageio
import struct
import shutil

def read_in_bin(in_path):

    file = open(in_path, "rb")
    fileContent = file.read()
    file.close()
    # Read in the header
    a = struct.unpack("iii", fileContent[:12])
    height = a[0]
    width = a[1]

    # Check data type
    data_type = a[2]
    if (data_type != 5):  # data_type = 5 means type CV_32F
        print("Data type ", data_type,
              " is not recognized. For reference see: http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html")
        exit()

    # Read in the data
    a = struct.unpack("f" * ((len(fileContent) - 16) // 4), fileContent[12:-4])
    b = struct.unpack("f", fileContent[-4:])  # last element
    c = a + b
    d = np.array(c)
    output_array = np.reshape(d, (height, width))

    return output_array



def deepdso_to_gipuma(dense_folder, image_folder):

    cam_folder = os.path.join(dense_folder, 'cams')
    depth_folder = os.path.join(dense_folder, 'invdepthmaps')

    gipuma_cam_folder = os.path.join(dense_folder, 'cams')
    if not os.path.isdir(gipuma_cam_folder):
        os.mkdir(gipuma_cam_folder)

    # Convert cameras
    image_names = os.listdir(image_folder)
    intrin_cam_file = os.path.join(dense_folder, 'intrinsic_matrix.txt')
    extrin_cam_file = os.path.join(dense_folder, 'camera_poses.txt')
    camera_names = os.listdir(gipuma_cam_folder)

    # Convert inverse depth maps.
    gipuma_prefix = 'camera_data__'
    for camera_name in camera_names:
        image_prefix = os.path.splitext(camera_name)[0]
        image_suffix = os.path.splitext(camera_name)[1]
        if (image_suffix !=".P"):
            continue

        sub_depth_folder = os.path.join(
            dense_folder, gipuma_prefix+(image_prefix))
        if not os.path.isdir(sub_depth_folder):
            os.mkdir(sub_depth_folder)

        # Copy inverse depth map
        out_depth_bin = os.path.join(sub_depth_folder, 'disp.bin')

        # Save depth PNGs
        disp_image = read_in_bin(out_depth_bin)
        depth_image = 1.0/(disp_image+0.0000001)
        maxdepth=50.0
        mindepth=0.0
        depth_image[depth_image >= maxdepth] = maxdepth
        depth_image[depth_image < mindepth] = mindepth
        out_path = os.path.join(depth_folder, image_prefix+'.txt')
        print(out_path)

        # Make outputs for each camera pose
        f = open(out_path, "w")
        for i in range(0, np.shape(disp_image)[0]):
            for j in range(0, np.shape(disp_image)[1]):
                f.write('[ ' + str(j) + ', ' + str(i) +', ' + str(np.sqrt(disp_image[i][j])) + '], ')
        f.close()


if __name__ == '__main__':

    ####################################################################################################################
    #       USAGE:
    #
    #       python3  make_txt_depthmap_values.py --dense_folder=/home/ubuntu/my_deepdso_outputs --images_folder==/home/ubuntu/image_inputs
    #
    #       NOTE:
    #
    #       "--dense_folder" path must contain subfolders:  cams, invdepthmaps
    #       The txt output can be used in the webapp for quick visualization (data must be copy/pasted).
    #
    ####################################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, default='')
    parser.add_argument('--dense_folder', type=str, default='')
    parser.add_argument('--fusibile_exe_path', type=str,
                        default='/home/ubuntu/TJ/fusibile/build/fusibile')
    parser.add_argument('--disp_threshold', type=float, default='1.0')
    parser.add_argument('--num_consistent', type=float, default='3')
    args = parser.parse_args()

    fusibile_exe_path = args.fusibile_exe_path
    disp_threshold = args.disp_threshold
    num_consistent = args.num_consistent

    # Convert to txt format
    print("----------------------------------------------------------------")
    print('Convert deepDSO output to txt format')
    deepdso_to_gipuma(args.dense_folder, args.image_folder)
