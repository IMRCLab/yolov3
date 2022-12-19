import numpy as np
import config_yolo as cfg
import rowan
import yaml
import csv
import glob
import math
from mpmath import csc
import matplotlib.pyplot as plt
import os
import pandas as pd
from time import perf_counter
from  plot import *
from functions import *

image_template = 'img_{0:05d}.jpg'

def main():
    csv1 = cfg.CSV1
    csv2 = cfg.CSV2 # remove for hovering
    T_cam_w, T_rob_w = np.eye(4), np.eye(4)
    Eucl_error = []
     
    with open(cfg.CAMERA_PARAMS_YAML) as f:
        camera_params = yaml.safe_load(f)
    fx = np.array(camera_params['camera_matrix'])[0][0]
    fy = np.array(camera_params['camera_matrix'])[1][1]
    ox = np.array(camera_params['camera_matrix'])[0][2]
    oy = np.array(camera_params['camera_matrix'])[1][2]

    cam = csv_to_dict(csv1) # dictionary keys=image names, values=time,state info
    robot = csv_to_dict(csv2)
    time_cam = np.array(list(cam.values()))[:,0]
    time_robot = np.array(list(robot.values()))[:,0]
    # pose_cam = np.array(list(cam.values()))[:,1:]
    pose_robot = np.array(list(robot.values()))[:,1:]
    # interpolate 
    x_rob_interp = np.interp(time_cam, time_robot, pose_robot[:,0])
    y_rob_interp = np.interp(time_cam, time_robot, pose_robot[:,1])
    z_rob_interp = np.interp(time_cam, time_robot, pose_robot[:,2])   
    q_interp = quaternion_interpolation(time_cam, pose_robot[:,3:], time_robot)
    start = perf_counter()
    for file in glob.glob(cfg.PREDICTION + "*.txt"):
        name = file[-13:]
        image_name = name[:9] + '.jpg' 
        index = list(cam.keys()).index(image_name)
        cam_state = cam[image_name] # time,x,y,z,qw,qx,qy,qz
        T_cam_w[:3,3] = cam_state[1:4] #skip the time
        T_rob_w[:3,3] = np.array([x_rob_interp[index],y_rob_interp[index], z_rob_interp[index] ])
        q_cam_in_w = np.array([cam_state[4], cam_state[5],cam_state[6], cam_state[7]])
        T_cam_w[:3,:3] = rowan.to_matrix(rowan.normalize(q_cam_in_w))
        T_rob_w[:3,:3] = rowan.to_matrix(rowan.normalize(q_interp[index]))
        # robot_in_cam = np.linalg.inv(T_cam_w) @ T_rob_w[:,3]
        T_rel = np.linalg.inv(T_cam_w) @ T_rob_w
        gt_xyz = T_rel[:3,3] # ground truth data, in meters.
        fileObj = open(file, "r")
        words = fileObj.read().splitlines() # get bb directly
        bb = words[0].split()
        a1 = np.array([-(int(bb[0])-ox)/fx, (((int(bb[1]) + int(bb[3]))/2)-oy)/fy, 1.])
        a2 = np.array([-(int(bb[2])-ox)/fx, (((int(bb[1]) + int(bb[3]))/2)-oy)/fy, 1.])
        a1_mag = np.linalg.norm(a1)
        a2_mag = np.linalg.norm(a2)
        angle = np.arccos(np.dot(a1,a2)/(a1_mag*a2_mag)) # radians 
        x = cfg.RADIUS*csc(angle/2) # distance
        curW = round((int(bb[0]) + int(bb[2]))/2) # center of bb is the center of CF
        curH = round((int(bb[1]) + int(bb[3]))/2)
        y = -x*(curW-oy)/fy
        z = -x*(curH-ox)/fx # CHECK ITS SIGN
        print(math.sqrt(pow((x - gt_xyz[0]), 2) + pow((y - gt_xyz[1]), 2) + pow((z - gt_xyz[2]), 2)))
        Eucl_error.append(math.sqrt(pow((x - gt_xyz[0]), 2) + pow((y - gt_xyz[1]), 2) + pow((z - gt_xyz[2]), 2)))

    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))
    # PlotHist.plot(Eucld_err,'euclidean-error-locanet.jpg')
    # fig = plt.figure()    
    # plt.hist(Eucl_error)
    # plt.suptitle('Histogram for Eucl. with Yolo in m.')
    # fig.savefig('Yolo-based-position-estimation-real.jpg')


if __name__ == "__main__":
    main()