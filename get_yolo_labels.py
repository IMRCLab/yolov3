
import cv2
import shutil
import os
import yaml
import argparse
import fnmatch
# Converts dataset.yaml into yolov3 training format
# python3 get_yolo_labels.py --yaml 'PATH-TO-SYNCHRONIZED-DATASET/dataset.yaml'
def run(yaml_path, img_size, img_ext, train_data_percentage):

    synchronized_data_folder = yaml_path[:-12]
    yolo_folder = synchronized_data_folder + '../yolov3/'
    shutil.rmtree(yolo_folder, ignore_errors=True)
    os.mkdir(yolo_folder) # Create a folder for saving images
    shutil.rmtree(yolo_folder + 'annotations', ignore_errors=True)
    shutil.rmtree(yolo_folder + 'bb', ignore_errors=True)
    os.mkdir(yolo_folder + 'annotations')
    os.mkdir(yolo_folder + 'bb') # to verify visually

    annotation_path = os.path.join(yolo_folder + 'annotations')
    with open(yaml_path, 'r') as stream:
        synchronized_data = yaml.safe_load(stream)

    total_imgs = fnmatch.filter(os.listdir(synchronized_data_folder), img_ext)
    for i in range(len(total_imgs)):
        img = cv2.imread(synchronized_data_folder + total_imgs[i])   
        file = open(os.path.join(annotation_path, total_imgs[i][:-4] + '.txt'), "w")
        for j in range(len(synchronized_data['images'][total_imgs[i]]['visible_neighbors'])):
            bb = synchronized_data['images'][total_imgs[i]]['visible_neighbors'][j]['bb'] # xmin,ymin,xmax,ymax
            xmin,ymin,xmax,ymax = bb[0],bb[1],bb[2],bb[3]
            h = ymax - ymin
            w = xmax - xmin
            x_c = round((xmin+xmax)/2)
            y_c = round((ymin+ymax)/2)
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            file.write(' {} {} {} {} {}'.format(0, x_c/img_size[0],  y_c/img_size[1],  w/img_size[0], h/img_size[1]))
        file.write('\n')   
        cv2.imwrite(os.path.join(yolo_folder + 'bb/', total_imgs[i]), img)

    file.close()  

# Prepare training, validation, testing data
    #create folders
    images_path = yolo_folder + 'images/'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = yolo_folder + 'labels/'
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    indices = list(range(0, len(total_imgs))) 
    numImgTrain = round(train_data_percentage/100*len(total_imgs))
    training_idx, val_idx, test_idx = indices[:numImgTrain], indices[numImgTrain:numImgTrain+int((len(total_imgs)-numImgTrain)/2)], indices[numImgTrain+int((len(total_imgs)-numImgTrain)/2):]
    idx = [training_idx,val_idx,test_idx]
    folders = ['train', 'val', 'test']
    
    for k in range(len(idx)): 
        target_img_path = images_path + folders[k] + '/'
        target_label_path = labels_path + folders[k] + '/'
        os.mkdir(target_img_path)
        os.mkdir(target_label_path)
        for t in idx[k]:
            src_image = synchronized_data_folder + total_imgs[t]
            src_label = yolo_folder + 'annotations/' + total_imgs[t][:-4] + '.txt'
            shutil.copy(src_image, target_img_path)                           
            shutil.copy(src_label, target_label_path)
        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', help="dataset.yaml file")
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=(320,320), help='image size w,h')
    parser.add_argument('--img_ext', type=str, default= '*.png', help="image extension") # png for real and jpg for synthetic images
    parser.add_argument('--training_data', type=int, default=90, help='training data percentage')

    args = parser.parse_args()
    return args

def main(args):
    run(args.yaml, args.imgsz, args.img_ext, args.training_data)


if __name__ == "__main__":
    args = parse_opt()
    main(args)