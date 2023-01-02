
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
from functions import *

def main():
    folder = cfg.DATASET_FOLDER
    csv1 = cfg.CSV1 # always assumed to be considered
    csv2 = cfg.CSV2
    img_size = cfg.IMG_SIZE
    img_names = []
    
    with open(csv1, 'r') as file: 
        csvreader_object= csv.reader(file)
        next(csvreader_object) # skip headers
        for row in csvreader_object:
            img_names.append(row[0])
            ncol = len(row) 
    T_cam_w, T_rob_w = np.eye(4), np.eye(4)
    corners = np.zeros([8,2])
    yolo_folder = folder + '../yolov3/'
    shutil.rmtree(yolo_folder, ignore_errors=True)
    os.mkdir(yolo_folder) # Create a folder for saving images
    shutil.rmtree(yolo_folder + 'annotations', ignore_errors=True)
    shutil.rmtree(yolo_folder + 'bb', ignore_errors=True)
    os.mkdir(yolo_folder + 'annotations')
    os.mkdir(yolo_folder + 'bb') # to verify visually

    label_path = os.path.join(yolo_folder + 'annotations')
    # label_file = 'img_{0:05d}.txt'
    # bb_name = 'img_bb_{0:05d}.png' # to verify visually

    with open(cfg.CAMERA_PARAMS_YAML) as f:
        camera_params = yaml.safe_load(f)

    int_mtrx = np.array(camera_params['camera_matrix'])
    dist_v = np.array(camera_params['dist_coeff'])
    t_v = np.array(camera_params['tvec'])
    r_v = np.array(camera_params['rvec'])

    x, y, z = 0.040, 0.040, 0.040 # synthetic x=0.040, y = 0.015, z =0.030
    M = np.array([[x, y, -z, 1.],
                 [x, y, z, 1.],
                 [-x, y, z, 1.],
                 [-x, y, -z, 1.],
                 [x, -y, -z, 1.],
                 [x, -y, z, 1.],
                 [-x, -y, z, 1.],
                 [-x, -y, -z, 1.]])

    # cols = pd.read_csv(csv1, nrows=1).columns 
    cam_in_w = np.loadtxt(csv1, comments='#',delimiter=',',skiprows=1, usecols=[i for i in range(1, ncol)])
    rob_in_w = np.loadtxt(csv2, comments='#',delimiter=',',skiprows=1, usecols=[i for i in range(1, ncol)]) 
    time = cam_in_w[:,0] 
    x_rob_interp, y_rob_interp, z_rob_interp, q_interp = [], [], [], []

    for j in range(cam_in_w.shape[1]//8): # for each robot in images
        x_rob_interp.append(np.interp(time, rob_in_w[:,8*j+0], rob_in_w[:,8*j+1]))
        y_rob_interp.append(np.interp(time, rob_in_w[:,8*j+0], rob_in_w[:,8*j+2]))
        z_rob_interp.append(np.interp(time, rob_in_w[:,8*j+0], rob_in_w[:,8*j+3]))
        q_interp.append(quaternion_interpolation(time, rob_in_w[:,8*j+4:8*j+8], rob_in_w[:,8*j+0]))

    for i in range(time.shape[0]): 
        img = cv2.imread(folder + img_names[i]) 
        success = 0
        # file = open(os.path.join(label_path, label_file.format(i)), "w") # for real-images should be changed
        file = open(os.path.join(label_path, img_names[i][:-4] + '.txt'), "w") # for real-images should be changed
        for j in range(len(q_interp)): # for each robot
            if q_interp[j][i] is None:
                continue
            cam_state = cam_in_w[i,8*j+1:8*j+8]
            T_cam_w[:3,3] = cam_state[:3] 
            T_rob_w[:3,3] = np.array([x_rob_interp[j][i],y_rob_interp[j][i], z_rob_interp[j][i]])
            q_cam_in_w = np.array([cam_state[3], cam_state[4],cam_state[5], cam_state[6]])
            T_cam_w[:3,:3] = rowan.to_matrix(rowan.normalize(q_cam_in_w))
            T_rob_w[:3,:3] = rowan.to_matrix(rowan.normalize(q_interp[j][i]))
            robot_j_in_cam = np.linalg.inv(T_cam_w) @ T_rob_w[:,3]
            if robot_j_in_cam[0]>0: # x for the distance
                T_corners_in_cam = (np.linalg.inv(T_cam_w) @ T_rob_w @ M.T).T 
                for j in range (T_corners_in_cam.shape[0]): # get pixels with re-projection
                    pixel, _ = cv2.projectPoints(T_corners_in_cam[j,:3], r_v, t_v, int_mtrx, dist_v)
                    corners[j,:] = np.resize(pixel,(1,2))

                xmin = round(corners[:,0].min())
                ymin = round(corners[:,1].min())
                xmax = round(corners[:,0].max())
                ymax = round(corners[:,1].max())

                h = ymax - ymin
                w = xmax - xmin
                x_c = round((xmin+xmax)/2)
                y_c = round((ymin+ymax)/2)

                if x_c/img_size[0] < 0.0 or x_c/img_size[0] > 1.0 or y_c/img_size[1] < 0.0 or y_c/img_size[1] > 1.0 or  w/img_size[0] < 0.0 or  w/img_size[0] > 1.0 or  h/img_size[1] < 0.0 or  h/img_size[1] > 1.0:
                    continue

                success += 1
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                file.write(' {} {} {} {} {}'.format(0, x_c/img_size[0],  y_c/img_size[1],  w/img_size[0], h/img_size[1]))
            file.write('\n')   
        file.close()   
        if success == 0:
            os.remove(os.path.join(label_path, img_names[i][:-4] + '.txt'))

        else: 
            cv2.imwrite(os.path.join(yolo_folder + 'bb/', img_names[i]), img)


if __name__ == "__main__":
    main()
