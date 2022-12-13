import yolov3.config_yolo as cfg
import rowan
import yaml
import csv
import glob
import math
from mpmath import csc
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
# from plot import *
from munkres import Munkres

image_template = 'img_{0:05d}.jpg' # for 5k


def get_euclidean_err(a,b):
    return math.sqrt(pow((a[0] - b[0]), 2) + pow((a[1] - b[1]), 2) + pow((a[2] - b[2]), 2))

def find_euc(pr_list, gt_list):
    matrix = []
    for i in range(len(gt_list)):
        tmp_list = []
        for j in pr_list:
            tmp_list.append(get_euclidean_err(gt_list[i], j))
        matrix.append(tmp_list)
    return matrix
# returns indices for each row and total sum of cost
def hungarian(m,mx):
    indexes = m.compute(mx)
    total = 0
    for row, column in indexes:
        value = mx[row][column]
        total += value
    return indexes, total

def find_min_euc(pr_list, gt_list):
    f = []
    for i in range(len(pr_list)):
        min_err = 10.
        gt_index = -1
        for j in gt_list:
            temp_err = get_euclidean_err(pr_list[i], j)
            if min_err > temp_err:
                min_err=temp_err
                gt_index += 1 # 0 for the first element
        f.append(min_err) # Euclidean error, prediction Cfs list
#         gt_list.remove(min_value)
        del gt_list[gt_index]
        
        if not gt_list:
            break
    return f

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

def testing_yolo():
    csv1 = cfg.CSV1
    # csv2 = cfg.CSV2
    T_cam, T_robot = np.eye(4), np.eye(4)
    Eucl_err_sum_yolo = []
    m = Munkres()

    with open(cfg.CAMERA_PARAMS_YAML) as f:
        camera_params = yaml.safe_load(f)
    fx = np.array(camera_params['camera_matrix'])[0][0]
    fy = np.array(camera_params['camera_matrix'])[1][1]
    ox = np.array(camera_params['camera_matrix'])[0][2]
    oy = np.array(camera_params['camera_matrix'])[1][2]
    
    # pose_robot = csv_dict(csv2)
    # time_cam = np.array(list(pose_cam.values()))[:,0]
    # time_robot = np.array(list(pose_robot.values()))[:,0]


    pose_cam = csv_to_dict(csv1)
    # pose_robot = csv_to_dict(csv2)
    pose_robot = np.array([1.0, 0.0, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0]) # for hovering CF in images, needs time also

    # time_cam = np.array(list(pose_cam.values()))[:,0]
    # time_robot = np.array(list(pose_robot.values()))[:,0]

    for file in glob.glob(cfg.PREDICTION + "*.txt"):
        # for file in glob.glob(PREDICTION + "*.txt"): # get prediction txt files in labels folder
        name = file[-13:]
        image_name = name[:9] + '.png' # three space for 5k
        state_cam = pose_cam[image_name] 
        # time_cam_curr = state_cam[0]

        # min_time_index = np.argmin(abs(time_cam_curr - time_robot))
        # state_robot = pose_robot[image_template.format(min_time_index)]
        state_robot = pose_robot
        # read predicted labels
        fileObj = open(file, "r")
        words = fileObj.read().splitlines() # get bb directly
        gt_yolo, pr_yolo = [], []
        for j in range((state_robot.shape[0])//8):
            T_cam[0,3], T_cam[1,3], T_cam[2,3] = state_cam[8*j+1], state_cam[8*j+2], state_cam[8*j+3]
            T_robot[0,3], T_robot[1,3], T_robot[2,3]= state_robot[8*j+1], state_robot[8*j+2], state_robot[8*j+3] 

            T_cam[:3,:3] = rowan.to_matrix([state_cam[8*j+4], state_cam[8*j+5], state_cam[8*j+6], state_cam[8*j+7]]) # in mocp
            T_robot[:3,:3] = rowan.to_matrix([state_robot[8*j+4], state_robot[8*j+5], state_robot[8*j+6], state_robot[8*j+7]]) # in mocap

            T_rel = np.linalg.inv(T_cam) @ T_robot # in camera frame
            # get g.t
            gt_xyz = T_rel[:3,3] # ground truth data, in meters.
            gt_yolo.append(gt_xyz)
        for k in range(len(words)):
            bb = words[k].split()
            a1 = np.array([-(int(bb[0])-ox)/fx, (((int(bb[1]) + int(bb[3]))/2)-oy)/fy, 1.])
            a2 = np.array([-(int(bb[2])-ox)/fx, (((int(bb[1]) + int(bb[3]))/2)-oy)/fy, 1.])
            a1_mag = np.linalg.norm(a1)
            a2_mag = np.linalg.norm(a2)
            angle = np.arccos(np.dot(a1,a2)/(a1_mag*a2_mag)) # radians 
            x_yolo = float(cfg.RADIUS*csc(angle/2)) # distance
            curW = round((int(bb[0]) + int(bb[2]))/2) # center of bb is the center of CF
            curH = round((int(bb[1]) + int(bb[3]))/2)
            y_yolo = float(-x_yolo*(curW-oy)/fy)
            z_yolo = float(-x_yolo*(curH-ox)/fx) # CHECK ITS SIGN
            pr_yolo.append(np.array([x_yolo,y_yolo,z_yolo]))
            
        # min_err_list_yolo = find_min_euc(pr_yolo, gt_yolo) # list of min error for predicted Cfs
        # print(min_err_list_yolo)
        # Eucl_err_sum_yolo.append(sum(min_err_list_yolo))
        if (len(pr_yolo)!=0):
            matrix = find_euc(pr_yolo, gt_yolo) # matrix of Euclidean distance between all prediction vs. g-t
            _,cost = hungarian(m,matrix)
            # print(cost)
            Eucl_err_sum_yolo.append(cost)
        pr_yolo.clear()
        gt_yolo.clear()
    # PlotHist.plot(Eucl_err_sum_yolo, 'euclidean-error-mrs-yolo.jpg')
    # PlotWhisker.plot(Eucl_err_sum_yolo, 'euclidean-synth-yolov3-mrs.jpg',"Box plot for multi-Cf, synthetic data, yolov3")
    return Eucl_err_sum_yolo
    
if __name__ == "__main__":
    testing_yolo()