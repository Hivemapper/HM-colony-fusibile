#!/usr/bin/env python
"""
Copyright 2021, Taylor Dahlke
Do very simple inversion projection of depthmap points into 3D coordinate space.
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
from plyfile import PlyData, PlyElement

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


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return


def gipuma_normal(in_depth_path, out_normal_path, cam_intrin_path):

    # Calculate the camera viewing vector
    # Read in the Rotation matrix (R) for this camera
    rotationM = np.zeros((3, 3), dtype=float)
    with open(cam_intrin_path) as f:
        content = f.readlines()
        count=0
        for line in content:
            newline = (line.strip()).split()
            rotationM[count, 0:3] = newline[0:3]
            count = count + 1

    # Calculate the view vector
    rotationM_T = np.transpose(rotationM)

    # Calculate a normal image
    # zy, zx = np.gradient(depth_image)
    # You may also consider using Sobel to get a joint Gaussian smoothing and differentation to reduce noise
    depth_image = read_in_bin(in_depth_path)
    zx = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=5)
    zy = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=5)

    normal = np.dstack((-zx, -zy, np.ones_like(depth_image)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n

    # Convert the normal vectors from camera coordinates to global coordinates
    # print(normal.shape)
    for x in range(0,normal.shape[0]):
        for y in range(0,normal.shape[1]):
            inputV = np.array([normal[x, y, 0], normal[x, y, 1], normal[x, y, 2]])
            convertedV = np.matmul(rotationM_T, inputV)
            normal[x, y, 0] = convertedV[0]
            normal[x, y, 1] = convertedV[1]
            normal[x, y, 2] = convertedV[2]

    normal_image=normal

    # Make masking
    image_shape = np.shape(depth_image)
    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    # Mask and write out normal image
    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)
    return


def deepdso_to_gipuma_cam(intrin_path, extrin_path, out_dir):
    '''convert deepDSO camera to gipuma camera format'''

    ########################################################################
    # camera_poses.txt is the input file from deepDSO
    #   timestamp, camToWorld.translation().x, camToWorld.translation().y, camToWorld.translation().z, camToWorld.so3().unit_quaternion().x(), camToWorld.so3().unit_quaternion().y(), camToWorld.so3().unit_quaternion().z(), camToWorld.so3().unit_quaternion().w()
    ########################################################################

    # Extrinsics matrix (3x4)
    #
    # E = [ R T ]
    #
    extrinsics = []
    camera_prefixes = []
    numposes = 0
    with open(extrin_path) as f:
        for line in f:
            numposes = numposes + 1
            if (numposes == 1):
                continue  # Skip the first line which is a header

            extrinsicM = np.zeros((3, 4), dtype=float)
            a = np.fromstring(line, dtype=float, sep=' ')
            b = line.split()
            camera_prefixes.append(b[1])
            # Translation matrix (T)
            extrinsicM[0:3, 3] = np.transpose(a[2:5])
            # Rotation matrix (R)
            extrinsicM[0, 0:3] = a[5:8]
            extrinsicM[1, 0:3] = a[8:11]
            extrinsicM[2, 0:3] = a[11:14]

            extrinsics.append(extrinsicM)
    extrinsics = np.stack(extrinsics)

    # Intrinsics matrix
    intrinsic = np.zeros((3, 3), dtype=float)

    with open(intrin_path) as f:
        count = 0
        for line in f:
            count = count + 1
            a = np.fromstring(line, sep=' ')
            if (count == 1):  # Line #1 (height, width)
                h = a[0]
                w = a[1]
            if (count == 2):  # K matrix (line 1)
                fx = a[0]
                cx = a[2]
            if(count == 3):  # K matrix (line 2)
                fy = a[1]
                cy = a[2]

    print("fx = ",fx, "  fy = ", fy, "  cx = ", cx, "  cy = ", cy, "  w = ", w, "  h = ", h)
    k = 1.0  # pixels/cm on image horizontal
    l = 1.0  # pixels/cm on image vertical
    # When k=l, the pixels are square (the most common case)
    alpha = fx*k
    beta = fy*l
    intrinsic[0, 0] = alpha
    intrinsic[1, 1] = beta
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    intrinsic[2, 2] = 1.0

    # Make outputs for each camera pose
    for pose in range(0, numposes-1):

        # Construct projection matrix
        extrinsic = extrinsics[pose, :, :]
        projection_matrix = np.matmul(intrinsic, extrinsic)

        # Write out the camera projection matrix file
        out_path = out_dir + "/" + str(camera_prefixes[pose]) + ".P"
        f = open(out_path, "w")
        for i in range(0, 3):
            for j in range(0, 4):
                f.write(str(projection_matrix[i][j]) + ' ')
            f.write('\n')
        f.write('\n')
        f.close()

        # Write out the camera extrinsics matrix file
        out_path = out_dir + "/" + str(camera_prefixes[pose]) + ".extrin"
        f = open(out_path, "w")
        for i in range(0, 3):
            for j in range(0, 4):
                f.write(str(extrinsic[i][j]) + ' ')
            f.write('\n')
        f.close()

    return


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
    deepdso_to_gipuma_cam(intrin_cam_file, extrin_cam_file, gipuma_cam_folder)
    camera_names = os.listdir(gipuma_cam_folder)

    # Convert inverse depth maps and normal maps. Only for inverse depthmaps that correspond to a saved camera.
    # If the camera wasn't saved, it is likely because deepDSO considered it an
    # invalid pose / marginalized frame (see deepDSO output logs).
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
        in_depth_bin = os.path.join(
            depth_folder, image_prefix+'.bin')
        shutil.copy(in_depth_bin, out_depth_bin)

        # Calculate normal map
        normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        cam_intrin_path = gipuma_cam_folder + "/" + image_prefix + ".extrin"
        gipuma_normal(out_depth_bin, normal_dmb, cam_intrin_path)


def depth_map_fusion(point_folder, image_folder, dense_folder, intrin_path, disp_thresh, num_consistent):
    
    gipuma_cam_folder = os.path.join(dense_folder, 'cams')
    if not os.path.isdir(gipuma_cam_folder):
        os.mkdir(gipuma_cam_folder)
    camera_names = os.listdir(gipuma_cam_folder)

    # Grab the camera intrinsics
    intrinsic = np.zeros((3, 3), dtype=float)
    with open(intrin_path) as f:
        count = 0
        for line in f:
            count = count + 1
            a = np.fromstring(line, sep=' ')
            if (count == 2):  # K matrix (line 1)
                fx = a[0]
                cx = a[2]
            if(count == 3):  # K matrix (line 2)
                fy = a[1]
                cy = a[2]

    # Loop over the cameras, project points to 3D, and append to PLY file
    ply_vec = []
    count=0
    gipuma_prefix = 'camera_data__'
    scale=50
    for camera_name in camera_names:
        
        # For debugging only
        # count=count+1
        # if (count>10):
        #     break
        if(not camera_name=="000006.extrin"):
            continue

        image_prefix = os.path.splitext(camera_name)[0]
        image_suffix = os.path.splitext(camera_name)[1]
        if (image_suffix !=".extrin"):
            continue
        print(camera_name)

        sub_depth_folder = os.path.join(
            dense_folder, gipuma_prefix+(image_prefix))
        if not os.path.isdir(sub_depth_folder):
            os.mkdir(sub_depth_folder)

        # Read in the inverse depth map
        out_depth_bin = os.path.join(sub_depth_folder, 'disp.bin')
        disp_image = read_in_bin(out_depth_bin)

        print(np.shape(disp_image))
        print(fx,cx,fy,cy)
        # Project each pixel to 3D with camera coordinate system
        for i in range(0, np.shape(disp_image)[0]): # Y axis (rows)
            for j in range(0, np.shape(disp_image)[1]): # X axis (columns)
                Z = 1.0/disp_image[i][j]
                X = (j-cx)*Z/fx
                Y = (i-cy)*Z/fy

                # Convert to global coordinate system

                # Append to PLY file
                ply_vec.append((scale*X, scale*Y, -scale*Z))

    # Write out the PLY file
    print("Writing PLY file ...")
    TT = np.array(ply_vec, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(TT, 'vertex')
    output_file = os.path.join(dense_folder, 'output.ply')
    PlyData([el], text=True).write(output_file)

    print("DONE!")

    return


if __name__ == '__main__':

    ####################################################################################################################
    #       USAGE:
    #
    #       python3  simple_inv_projection.py --dense_folder=/home/ubuntu/my_deepdso_outputs --images_folder==/home/ubuntu/image_inputs
    #
    #       NOTE:
    #
    #       "--dense_folder" path must contain subfolders:  cams, invdepthmaps
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

    # Depth map fusion with gipuma
    print("----------------------------------------------------------------")
    print('Run depth map fusion & filter')
    intrin_cam_file = os.path.join(args.dense_folder, 'intrinsic_matrix.txt')
    depth_map_fusion(args.dense_folder, args.image_folder, args.dense_folder, intrin_cam_file,
                     disp_threshold, num_consistent)
