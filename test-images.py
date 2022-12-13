import os
import cv2
import numpy as np

def load_annotations(annot_path):
        with open(annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
            return annotations

image_name = 'img_{0:05d}.jpg'
directory = "yolo-5k"
parent_dir = '/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/'
path = os.path.join(parent_dir, directory)
os.mkdir(path)
annotations_test = load_annotations('/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/5000/test.txt')
DATASET_FOLDER = '/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/5000/'
for i in range(len(annotations_test)):
    ann = annotations_test[i]
    line = ann.split()
    image_path = DATASET_FOLDER + line[0]    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # <320, 320>
    image = image[..., np.newaxis] # <320, 320, 1>
    image = image.astype('float32')
    # cv2.imwrite(os.path.join(path, image_name.format(i)), image)
    cv2.imwrite(os.path.join(path, line[0]), image)