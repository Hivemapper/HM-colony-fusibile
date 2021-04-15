

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
from plyfile import PlyData, PlyElement



def depth_map_fusion(fx, cx, fy, cy, disp_image, ply_vec):
    scale=10.0
    print(np.shape(disp_image))
    print(fx, cx, fy, cy)
    # Project each pixel to 3D with camera coordinate system
    for i in range(0, np.shape(disp_image)[0]):  # Y axis (rows)
        for j in range(0, np.shape(disp_image)[1]):  # X axis (columns)
            Z=disp_image[i][j]
            X = (j-cx)*Z/fx
            Y = (i-cy)*Z/fy

            # Convert to global coordinate system

            # Append to PLY file
            ply_vec.append((scale*X, scale*Y, -scale*Z))

    return


if __name__ == '__main__':

    ####################################################################################################################
    #       USAGE:
    #
    #    This script is meant to visualize depthmaps that are sourced as grayscale images. You may have to change the way that intrinsics are read in depending on your intrinsic file input.
    #
    #    python3  ground_truth_test.py --depthIN=/home/ubuntu/TJ/ground_truth_test/0000000000.png --intrinsic=/home/ubuntu/TJ/ground_truth_test/0000000000.txt --output=/home/ubuntu/TJ/ground_truth_test/ground_truth_test.ply    #
    #
    ####################################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--depthIN', type=str, default='')
    parser.add_argument('--intrinsic', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    # Grab the camera intrinsics
    with open(args.intrinsic) as f:
        for line in f:
            a = np.fromstring(line, sep=' ')
            fx = a[0]
            cx = a[2]
            fy = a[4]
            cy = a[5]

    # Read depth PNGs
    depth=imageio.imread(args.depthIN)

    # Initialize the PLY vector
    ply_vec = []

    # Depth map fusion
    print("----------------------------------------------------------------")
    print('Run depth map fusion & filter')
    depth_map_fusion(fx, cx, fy, cy, depth, ply_vec)

    # Write out the PLY file
    print("Writing PLY file ...")
    TT = np.array(ply_vec, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(TT, 'vertex')
    PlyData([el], text=True).write(args.output)
