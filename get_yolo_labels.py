
import cv2
import shutil
import os
import yaml
import argparse
import glob
from os.path import normpath, basename
# Converts dataset.yaml into yolov3 training format. Saves all images with/without robot, and labels for images with no robot is empty.
# python3 get_yolo_labels.py 'PATH-TO-MAIN-FOLDER
def run(main_data_folder , img_size, img_ext, train_data_percentage):

    synchronized_data_folder = main_data_folder + 'Synchronized-Dataset/'
    yaml_path = synchronized_data_folder + 'dataset.yaml'
    yolo_folder = main_data_folder + 'yolov3/'
    shutil.rmtree(yolo_folder, ignore_errors=True)
    os.mkdir(yolo_folder) # Create a folder for saving images
    shutil.rmtree(yolo_folder + 'annotations', ignore_errors=True)
    shutil.rmtree(yolo_folder + 'bb', ignore_errors=True)
    os.mkdir(yolo_folder + 'annotations')
    os.mkdir(yolo_folder + 'bb') # to verify visually
   # Prepare training, validation, testing data
    images_path = yolo_folder + 'images/'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = yolo_folder + 'labels/'
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    yolo_folders = ['train', 'val', 'test']
    # train, test, validate folders
    for k in range(len(yolo_folders)): 
        os.mkdir(images_path + yolo_folders[k] + '/')
        os.mkdir(labels_path + yolo_folders[k] + '/')
    annotation_path = os.path.join(yolo_folder + 'annotations')
    with open(yaml_path, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)

    for folder in [f.path for f in os.scandir(synchronized_data_folder) if f.is_dir()]: # for each subfolder 0,1,2 etc.
        robot_number = basename(normpath(folder)) # get the folder with robot number
        total_imgs = sorted(filter( os.path.isfile, glob.glob(folder + '/' + img_ext) ) )
        for i in range(len(total_imgs)):
            img = cv2.imread(total_imgs[i]) 
            file = open(os.path.join(annotation_path, total_imgs[i].split("/")[-1][:-4] + '.txt'), "w") # iamge name without jpg
        
            for j in range(len(synchronized_data['images'][str(robot_number) + '/' + total_imgs[i].split("/")[-1]]['visible_neighbors'])):
                bb = synchronized_data['images'][str(robot_number) + '/' + total_imgs[i].split("/")[-1]]['visible_neighbors'][j]['bb'] # xmin,ymin,xmax,ymax
                xmin,ymin,xmax,ymax = bb[0],bb[1],bb[2],bb[3]
                h = ymax - ymin
                w = xmax - xmin
                x_c = round((xmin+xmax)/2)
                y_c = round((ymin+ymax)/2)
                
                # if x_c/img_size[0] <= 0. or x_c/img_size[0] >= 1.0 or y_c/img_size[1] <= 0. or y_c/img_size[1] >= 1.0 or w/img_size[0] <= 0. or w/img_size[0] >= 1.0 or h/img_size[1] <= 0. or h/img_size[1] >= 1.0:
                #     continue
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                file.write(' {} {} {} {} {}'.format(0, x_c/img_size[0],  y_c/img_size[1],  w/img_size[0], h/img_size[1]))
                file.write('\n')   
            cv2.imwrite(os.path.join(yolo_folder + 'bb/', total_imgs[i].split("/")[-1]), img)
            file.close()  
        indices = list(range(0, len(total_imgs))) 
        numImgTrain = round(train_data_percentage/100*len(total_imgs))
        training_idx, val_idx, test_idx = indices[:numImgTrain], indices[numImgTrain:numImgTrain+int((len(total_imgs)-numImgTrain)/2)], indices[numImgTrain+int((len(total_imgs)-numImgTrain)/2):]
        idx = [training_idx,val_idx,test_idx]  
        for k in range(len(idx)): 
            target_img_path = images_path + yolo_folders[k] + '/'
            target_label_path = labels_path + yolo_folders[k] + '/'
            for t in idx[k]:
                src_image = folder + '/' + total_imgs[t].split("/")[-1]
                src_label = yolo_folder + 'annotations/' + total_imgs[t].split("/")[-1][:-4] + '.txt'
                shutil.copy(src_image, target_img_path)                           
                shutil.copy(src_label, target_label_path)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('foldername', help="dataset.yaml file")
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=(320,320), help='image size w,h')
    parser.add_argument('--img_ext', type=str, default= '*.jpg', help="image extension") # png for real and jpg for synthetic images
    parser.add_argument('--training_data', type=int, default=90, help='training data percentage')

    args = parser.parse_args()
    return args

def main(args):
    run(args.foldername, args.imgsz, args.img_ext, args.training_data)


if __name__ == "__main__":
    args = parse_opt()
    main(args)
