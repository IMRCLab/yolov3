import yaml
import argparse
import itertools
from pathlib import Path
import shutil, os, cv2
# python3 get_yolo_labels.py -f args_1 -f args_2
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str, nargs='+', action='append', help='file list')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=(320,320), help='image size w,h')
    parser.add_argument('--img_ext', type=str, default= '*.jpg', help="image extension") # png for real and jpg for synthetic images
    parser.add_argument('--training_data_percentage', type=int, default=90, help='training data percentage')
    args = parser.parse_args()
    
    args = parser.parse_args()
    data = args.file  # list of files
    img_size = args.imgsz
    train_data_percentage = args.training_data_percentage
    
    yolo_folder = str(Path(data[0][0]).parent.parent.parent / "yolov3")
    shutil.rmtree(yolo_folder, ignore_errors=True)
    os.mkdir(yolo_folder) # Create a folder for saving images

    ann_folder = yolo_folder + '/annotations'
    shutil.rmtree(ann_folder, ignore_errors=True)
    os.mkdir(yolo_folder + '/annotations')
    bb_folder = yolo_folder + '/bb'
    shutil.rmtree(bb_folder, ignore_errors=True) 
    os.mkdir(yolo_folder + '/bb') # to verify visually
   # Prepare training, validation, testing data
    images_path = yolo_folder + '/images/'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = yolo_folder + '/labels/'
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    yolo_folders = ['train', 'val']
    # train, test, validate folders
    for k in range(len(yolo_folders)): 
        os.mkdir(images_path + yolo_folders[k] + '/')
        os.mkdir(labels_path + yolo_folders[k] + '/')
    
    for i in range(len(data)): 
        main_folder =  Path(data[i][0]).parent
        yaml_file = data[i][0]
        numImgTrain = round(train_data_percentage/100*int(data[i][1]))
        data_cnt = 0
        with open(yaml_file, 'r') as stream:
            synchronized_data = yaml.safe_load(stream)
        training_val_data = dict(itertools.islice(synchronized_data['images'].items(), int(data[i][1])))
        for image, value in training_val_data.items():
            file = open(os.path.join(ann_folder, image[:-4] + '.txt'), "w") 
            neighbors = value['visible_neighbors']
            if len(neighbors) > 0:
                img = cv2.imread(str(main_folder / image))
                for neighbor in neighbors:
                    xmin, ymin, xmax, ymax = neighbor['bb']
                    h = ymax - ymin
                    w = xmax - xmin
                    x_c = round((xmin+xmax)/2)
                    y_c = round((ymin+ymax)/2)     
                    if x_c/img_size[0] <= 0. or x_c/img_size[0] >= 1.0 or y_c/img_size[1] <= 0. or y_c/img_size[1] >= 1.0 or w/img_size[0] <= 0. or w/img_size[0] >= 1.0 or h/img_size[1] <= 0. or h/img_size[1] >= 1.0:
                        continue
                    cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                    file.write(' {} {} {} {} {}'.format(0, x_c/img_size[0],  y_c/img_size[1],  w/img_size[0], h/img_size[1]))
                    file.write('\n')                     
                cv2.imwrite(os.path.join(bb_folder, image), img) # just imgs with robot
            file.close()
            data_cnt += 1
            src_img_path = str(main_folder / image)
            src_file_path = str(ann_folder + '/' + image[:-4] + '.txt')
            if data_cnt <= numImgTrain:
                target_img_path = images_path + yolo_folders[0] + '/'
                target_label_path = labels_path + yolo_folders[0] + '/'
                shutil.copy(src_img_path, target_img_path)                           
                shutil.copy(src_file_path, target_label_path)
            else:
                target_img_path = images_path + yolo_folders[1] + '/'
                target_label_path = labels_path + yolo_folders[1] + '/'
                shutil.copy(src_img_path, target_img_path)                           
                shutil.copy(src_file_path, target_label_path)
          
            

if __name__ == "__main__":
    main()
