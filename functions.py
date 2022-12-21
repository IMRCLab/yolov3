import rowan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import math
import yolov3.config_yolo as cfg_yolo
import yaml
from mpmath import csc

def quaternion_interpolation(time, q, t):
    # time - interpolate q with this time
    q_i = []
    for i in range(0, len(time)):
        if time[i] < t[0] or time[i] > t[-1]:
            q_i.append(None)
            continue
        min_val_index = np.where(t - time[i] >= 0)[0][0] # if t = t[i]
        t1, q_1 = t[min_val_index-1], q[min_val_index-1]
        t2, q_2 = t[min_val_index], q[min_val_index]

        interpolation_parameter = (time[i]-t1)/(t2-t1)
        q_inter = rowan.interpolate.slerp(q_1, q_2, interpolation_parameter)
        q_i.append(q_inter[0]) # interpolated quaternion
    return q_i

def get_euler(q): # list of array
    # q = [i for i in q if i is not None]
    roll, pitch, yaw = [], [], []
    for i in range(len(q)):
        Euler = rowan.to_euler(q[i])
        roll.append(Euler[2])
        pitch.append(Euler[1])
        yaw.append(Euler[0])
    return roll, pitch, yaw

class Plot:
    figure_counter = 0
    @staticmethod
    def plot(r, p, y, t, ri, pi, yi, t_high):
        Plot.figure_counter += 1
        plt.figure(Plot.figure_counter, figsize=(20, 12))
        plt.margins(0.1)

        gs = gridspec.GridSpec(1, 3)
        ax = plt.subplot(gs[0, 0])
        # ax.set_title('roll')

        plt.plot(t, np.degrees(r), '-o', label='Roll')
        plt.plot(t_high, np.degrees(ri), '-x')
        plt.legend()

        ax = plt.subplot(gs[0, 1])
        # ax.set_title('y')

        plt.plot(t, np.degrees(p), '-o', label='Pitch')
        plt.plot(t_high, np.degrees(pi), '-x')
        plt.legend()

        ax = plt.subplot(gs[0, 2])
        # ax.set_title('z')

        plt.plot(t, np.degrees(y), '-o', label='Yaw')
        plt.plot(t_high, np.degrees(yi), '-x')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        plt.suptitle('Quaternion interpolation')
    
        plt.show()

def cf_dict_time(csv_file):
    cf_list = []
    dicts = {}
    poses = np.loadtxt(csv_file,comments='#',delimiter=',',skiprows=1, usecols=(1,2,3,4,5,6,7,8))
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            cf_list.append(row)
    for i in range(1, len(cf_list)):
        dicts[cf_list[i][0]] = poses[i-1]
    return dicts

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

def xyz_from_bb(bb):
    fx,fy,ox,oy = get_camera_parameters()
    a1 = np.array([-(int(bb[0])-ox)/fx, (((int(bb[1]) + int(bb[3]))/2)-oy)/fy, 1.])
    a2 = np.array([-(int(bb[2])-ox)/fx, (((int(bb[1]) + int(bb[3]))/2)-oy)/fy, 1.])
    a1_mag = np.linalg.norm(a1)
    a2_mag = np.linalg.norm(a2)
    angle = np.arccos(np.dot(a1,a2)/(a1_mag*a2_mag)) # radians 
    x = cfg_yolo.RADIUS*csc(angle/2) # distance
    curW = round((int(bb[0]) + int(bb[2]))/2) # center of bb is the center of CF
    curH = round((int(bb[1]) + int(bb[3]))/2)
    y = -x *(curW-oy)/fy
    z = -x *(curH-ox)/fx # SIGN ?
    return x,y,z

def get_camera_parameters():
    with open(cfg_yolo.CAMERA_PARAMS_YAML) as f:
        camera_params = yaml.safe_load(f)
    fx = np.array(camera_params['camera_matrix'])[0][0]
    fy = np.array(camera_params['camera_matrix'])[1][1]
    ox = np.array(camera_params['camera_matrix'])[0][2]
    oy = np.array(camera_params['camera_matrix'])[1][2]
    return fx,fy,ox,oy