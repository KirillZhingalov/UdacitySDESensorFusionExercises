# ---------------------------------------------------------------------
# Exercises from lesson 1 (lidar)
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.  
#
# Purpose of this file : Starter Code
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

from PIL import Image
import io
import sys
import os
import cv2
import time
import numpy as np
import zlib

## Add current working directory to path
sys.path.append(os.getcwd())

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2


# Example C1-5-1 : Load range image
def load_range_image(frame, lidar_name):
    
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    return ri


# Exercise C1-5-5 : Visualize intensity channel
def vis_intensity_channel(frame, lidar_name):

    print("Exercise C1-5-5")
    # extract range image from frame
    range_image = load_range_image(frame, lidar_name)[..., 1]
    range_image[range_image < 0.] = 0.

    # map value range to 8bit
    # intensity_image = np.amax(range_image)/2 * range_image * 255 / np.amax(range_image) 
    intensity_image = ((range_image.max() / 2) * (range_image / range_image.max()) * 255.).astype(np.uint8)
    print('Range image shape:', range_image.shape)
    
    # focus on +/- 45° around the image center
    deg45 = int(intensity_image.shape[1] / 8)
    ri_center = int(intensity_image.shape[1]/2)
    intensity_image = intensity_image[:,ri_center-deg45:ri_center+deg45]

    cv2.imshow('range_image', intensity_image)
    cv2.waitKey(0)

# Exercise C1-5-2 : Compute pitch angle resolution
def print_pitch_resolution(frame, lidar_name):

    print("Exercise C1-5-2")

    # load range image
        
    # compute vertical field-of-view from lidar calibration 
    calib_lidar = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    vfov_rad = calib_lidar.beam_inclination_max - calib_lidar.beam_inclination_min
    
    # compute pitch resolution and convert it to angular minutes
    print('Pitch resolution (in degrees):', np.degrees(vfov_rad / 64) * 60)

# Exercise C1-3-1 : print no. of vehicles
def print_no_of_vehicles(frame):

    print("Exercise C1-3-1")    

    # find out the number of labeled vehicles in the given frame
    # Hint: inspect the data structure frame.laser_labels
    num_vehicles = len([label for label in frame.laser_labels if label.type == 1])
            
    print("number of labeled vehicles in current frame = " + str(num_vehicles))