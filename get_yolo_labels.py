import yaml
import argparse
from pathlib import Path
import shutil, os, cv2
# python3 get_yolo_labels.py -f args_1

def get_yolo_labels(dataset_yaml, output_folder, mode, training_data_percentage=90):

    main_folder = Path(output_folder)
    img_size = (320,320)
    train_data_percentage = training_data_percentage
    
    yolo_folder = main_folder / "yolov3"
    shutil.rmtree(yolo_folder, ignore_errors=True)
    os.mkdir(yolo_folder) # Create a folder for saving images

    ann_folder = yolo_folder / 'annotations'
    shutil.rmtree(ann_folder, ignore_errors=True)
    os.mkdir(ann_folder)
    
    bb_folder = yolo_folder / 'bb'
    shutil.rmtree(bb_folder, ignore_errors=True) 
    os.mkdir(bb_folder) # to verify visually
   
    # Prepare training, validation, testing data
    images_path = yolo_folder / 'images'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = yolo_folder / 'labels'
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    if mode == 'test':
        data_folders = ['test']
    else:
        data_folders = ['val','train']
    # train, test, validate folders
    for k in range(len(data_folders)): 
        os.mkdir(images_path / data_folders[k])
        os.mkdir(labels_path / data_folders[k])
    
    yaml_file = dataset_yaml
    with open(yaml_file, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)
    num_images = len(synchronized_data['images'])

    numImgTrain = round(train_data_percentage/100*num_images)
    data_cnt = 0
    filename_to_dataset_key = dict()
    for image_name, entry in synchronized_data['images'].items():

        new_image_name = str(Path(image_name).name)
        if new_image_name in filename_to_dataset_key:
            # already there -> resolve duplicates
            i = 0
            while True:
                new_image_name = str(Path(image_name).stem) + "_" + str(i) + str(Path(image_name).suffix)
                if new_image_name not in filename_to_dataset_key:
                    break
                i += 1
        filename_to_dataset_key[new_image_name] = image_name

        src_file_path = ann_folder / Path(new_image_name).with_suffix(".txt")
        file = open(src_file_path, "w")
        neighbors = entry['visible_neighbors']
        img = cv2.imread(str(Path(dataset_yaml).parent / image_name))
        for neighbor in neighbors:
            xmin, ymin, xmax, ymax = neighbor['bb']
            h = ymax - ymin
            w = xmax - xmin
            x_c = round((xmin+xmax)/2)
            y_c = round((ymin+ymax)/2)     
            # if x_c/img_size[0] <= 0. or x_c/img_size[0] >= 1.0 or y_c/img_size[1] <= 0. or y_c/img_size[1] >= 1.0 or w/img_size[0] <= 0. or w/img_size[0] >= 1.0 or h/img_size[1] <= 0. or h/img_size[1] >= 1.0:
            #     continue
            file.write(' {} {} {} {} {}'.format(0, x_c/img_size[0],  y_c/img_size[1],  w/img_size[0], h/img_size[1]))
            file.write('\n')
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
        file.close()

        if len(neighbors) > 0:
            cv2.imwrite(str(bb_folder / new_image_name), img)

        data_cnt += 1
        src_img_path = Path(dataset_yaml).parent / image_name

        idx = 0
        if data_cnt <= numImgTrain and mode =='train':
            idx = 1
        target_img_path = images_path / data_folders[idx] / new_image_name
        target_label_path = labels_path / data_folders[idx]
        shutil.copy(src_img_path, target_img_path)
        shutil.copy(src_file_path, target_label_path)

    
    with open(yolo_folder / "filename_to_dataset_mapping.yaml", "w") as f:
        yaml.dump(filename_to_dataset_key, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file', type=str, help='dataset.yaml file')
    parser.add_argument('--training_data_percentage', type=int, default=90, help='training data percentage')
    parser.add_argument('-mode',help='train or test')
    
    args = parser.parse_args()

    main_folder =  Path(args.file).parent.parent

    get_yolo_labels(args.file, main_folder, args.mode)

if __name__ == "__main__":
    main()
