import numpy as np
import config as cfg
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
image_template = 'img_{0:05d}.jpg   '


# def csv_dict(csv_file):
#     cf_list = []
#     dicts = {}
#     cols = pd.read_csv(csv_file, nrows=1).columns 
#     poses = np.loadtxt(csv_file, comments='#',delimiter=',',skiprows=1, usecols=[i for i in range(1, cols.shape[0])]) #np.loadtxt(csv_file,comments='#',delimiter=',',skiprows=1, usecols=(1,2,3,4,5,6,7,8)) # t, x, y, z, qw, qx, qy, qz
#     with open(csv_file, 'r') as file:
#         reader = csv.reader(file)
#         for row in reader:
#             cf_list.append(row)
#     for i in range(1, len(cf_list)):
#         dicts[cf_list[i][0]] = poses[i-1]
#     return dicts
def csv_to_dict(csv_file):
    dicts={}
    cf_list=[]
    with open(csv_file) as f:
        next(f)
        ncols = len(f.readline().split(','))
    data = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=range(1,ncols))
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            cf_list.append(row)
    for i in range(1, len(cf_list)):
        dicts[cf_list[i][0]] = data[i-1]
    return dicts

def main():
    csv1 = cfg.CSV1
    csv2 = cfg.CSV2 # remove for hovering
    T_cam, T_robot = np.eye(4), np.eye(4)
    Eucl_error = []
     
    with open(cfg.CAMERA_PARAMS_YAML) as f:
        camera_params = yaml.safe_load(f)
    fx = np.array(camera_params['camera_matrix'])[0][0]
    fy = np.array(camera_params['camera_matrix'])[1][1]
    ox = np.array(camera_params['camera_matrix'])[0][2]
    oy = np.array(camera_params['camera_matrix'])[1][2]

    pose_cam = csv_to_dict(csv1)  
    pose_robot = csv_to_dict(csv2)
    # time_cam = np.array(list(pose_cam.values()))[:,0]
    time_robot = np.array(list(pose_robot.values()))[:,0]
    start = perf_counter()
    for file in glob.glob(cfg.PREDICTION + "*.txt"):
        name = file[-13:]
        image_name = name[:9] + '.jpg   ' #name[:9] + '.jpg   '
        state_cam = pose_cam[image_name] # time,x,y,z,qw,qx,qy,qz
        time_cam = state_cam[0]
        min_time_index = np.argmin(abs(time_cam - time_robot))
        # print(min_time_index)
        # state_robot = np.array([0.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0]) # for hovering case
        state_robot = pose_robot[image_template.format(min_time_index)]
        T_cam[0,3], T_cam[1,3], T_cam[2,3] = state_cam[1], state_cam[2], state_cam[3]
        T_robot[0,3], T_robot[1,3], T_robot[2,3]= state_robot[1], state_robot[2], state_robot[3] # change for hovering - no time
        
        T_cam[:3,:3] = rowan.to_matrix([state_cam[4], state_cam[5], state_cam[6], state_cam[7]]) # in mocap
        T_robot[:3,:3] = rowan.to_matrix([state_robot[4], state_robot[5], state_robot[6], state_robot[7]]) # in mocap

        T_rel = np.linalg.inv(T_cam) @ T_robot # in camera frame
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
        Eucl_error.append(math.sqrt(pow((x - gt_xyz[0]), 2) + pow((y - gt_xyz[1]), 2) + pow((z - gt_xyz[2]), 2)))

    end = perf_counter()
    print("Time taken for test is {} min.".format((end-start)/60.))
    # PlotHist.plot(Eucld_err,'euclidean-error-locanet.jpg')
    PlotWhisker.plot(Eucl_error, 'euclidean-synth-whisker-yolov3')

    # fig = plt.figure()    
    # plt.hist(Eucl_error)
    # plt.suptitle('Histogram for Eucl. with Yolo in m.')
    # fig.savefig('Yolo-based-position-estimation-real.jpg')


if __name__ == "__main__":
    main()