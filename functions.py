import rowan
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv

def quaternion_interpolation(time, q, t):
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