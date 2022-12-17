import cv2
import numpy as np
import config_yolo as cfg
import shutil
import os
import rowan
import pandas as pd
import yaml
from PIL import Image 
import csv
import random
from functions import *

def main():
    folder = cfg.DATASET_FOLDER
    csv1 = cfg.CSV1 # always assumed to be considered/camera
    csv2 = cfg.CSV2
    img_size = cfg.IMG_SIZE
    img_names = []

    with open(csv1, 'r') as file: 
        csvreader_object= csv.reader(file)
        next(csvreader_object) # skip headers
        for row in csvreader_object:
            img_names.append(row[0])

    T_cam_w, T_rob_w = np.eye(4), np.eye(4)
    corners = np.zeros([8,2])
    shutil.rmtree(folder + 'annotations', ignore_errors=True)
    shutil.rmtree(folder + 'bb', ignore_errors=True)
    os.mkdir(folder + 'annotations')
    os.mkdir(folder + 'bb') # to verify visually

    label_path = os.path.join(folder + 'annotations')
    label_file = 'img_{0:05d}.txt'
    bb_name = 'img_bb_{0:05d}.png' # to verify visually
    img_name = 'img_{0:05d}.png' # to verify visually


    with open(cfg.CAMERA_PARAMS_YAML) as f:
        camera_params = yaml.safe_load(f)

    int_mtrx = np.array(camera_params['camera_matrix'])
    dist_v = np.array(camera_params['dist_coeff'])
    t_v = np.array(camera_params['tvec'])
    r_v = np.array(camera_params['rvec'])

    x, y, z = 0.045, 0.025, 0.045 # Crazyflie physical dimensions, for SYNTHETIC DATA
    M = np.array([[x, y, -z, 1.],
                 [x, y, z, 1.],
                 [-x, y, z, 1.],
                 [-x, y, -z, 1.],
                 [x, -y, -z, 1.],
                 [x, -y, z, 1.],
                 [-x, -y, z, 1.],
                 [-x, -y, -z, 1.]])

    cols = pd.read_csv(csv1, nrows=1).columns 
    cam_in_w = np.loadtxt(csv1, comments='#',delimiter=',',skiprows=1, usecols=[i for i in range(1, cols.shape[0])])
    rob_in_w = np.loadtxt(csv2, comments='#',delimiter=',',skiprows=1, usecols=[i for i in range(1, cols.shape[0])]) 
    time = cam_in_w[:,0] 

    # interpolate the position of the robot
    x_rob_interp = np.interp(time, rob_in_w[:,0], rob_in_w[:,1])
    y_rob_interp = np.interp(time, rob_in_w[:,0], rob_in_w[:,2])
    z_rob_interp = np.interp(time, rob_in_w[:,0], rob_in_w[:,3])  
    q_interp = quaternion_interpolation(time, rob_in_w[:,4:], rob_in_w[:,0])
    for i in range(time.shape[0]): 
        # rob_state = rob_in_w  
        if q_interp[i] is None:
          continue
        cam_state = cam_in_w[i,1:] # skip the timestamp
        T_cam_w[:3,3] = cam_state[:3] # T_cam_w[1,3], T_cam_w[2,3]
        T_rob_w[:3,3] = np.array([x_rob_interp[i],y_rob_interp[i], z_rob_interp[i] ])
        q_cam_in_w = np.array([cam_state[3], cam_state[4],cam_state[5], cam_state[6]])
        # q_rob_in_w = np.array([rob_state[3], rob_state[4],rob_state[5], rob_state[6]])
        T_cam_w[:3,:3] = rowan.to_matrix(rowan.normalize(q_cam_in_w))
        T_rob_w[:3,:3] = rowan.to_matrix(rowan.normalize(q_interp[i]))

        # check if the object is behind the scene
        robot_cam = np.linalg.inv(T_cam_w) @ T_rob_w[:,3]
        xy_hom = (int_mtrx @ robot_cam[:3].T) # in image plane
        z = xy_hom[2] # FOR SYNTHETIC DATA
        if z > 0:
            T_corners_in_cam = (np.linalg.inv(T_cam_w) @ T_rob_w @ M.T).T
                
            for j in range (T_corners_in_cam.shape[0]): # get pixels with re-projection
                pixel, _ = cv2.projectPoints(T_corners_in_cam[j,:3], r_v, t_v, int_mtrx, dist_v)
                corners[j,:] = np.resize(pixel,(1,2))
            if np.isnan(np.sum(corners)):
                continue
            xmin = round(corners[:,0].min())
            ymin = round(corners[:,1].min())
            xmax = round(corners[:,0].max())
            ymax = round(corners[:,1].max())
            if xmin < 0 or ymin < 0 or xmax > cfg.IMG_SIZE[0] or ymax > cfg.IMG_SIZE[1]:
                continue

            h = ymax - ymin
            w = xmax - xmin
            x_c = round((xmin+xmax)/2)
            y_c = round((ymin+ymax)/2)

            if x_c/img_size[0] <= 0.0 or x_c/img_size[0] >= 1.0 or y_c/img_size[1] <= 0.0 or y_c/img_size[1] >= 1.0 or  w/img_size[0] <= 0.0 or  w/img_size[0] >= 1.0 or  h/img_size[1] <= 0.0 or  h/img_size[1] >= 1.0:
                continue
            with open(os.path.join(label_path, label_file.format(i)), "w") as file: 
                file.write(' {} {} {} {} {}'.format(0, x_c/img_size[0],  y_c/img_size[1],  w/img_size[0], h/img_size[1]))
                file.write('\n')   
            file.close()
            img = cv2.imread(folder + img_names[i]) 
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(folder + 'bb/', img_name.format(i)), img)
    

if __name__ == "__main__":
    main()