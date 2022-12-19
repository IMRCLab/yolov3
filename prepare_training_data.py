import os, shutil, random
import config_yolo as cfg

# preparing the folder structure for YOLOv3
# creates images/train-val-test, labels/train-val-test ready for yolo training

def main():
    full_data_path = cfg.DATASET_FOLDER
    ann_data_path = full_data_path + '../annotations/'
    training_data_path = full_data_path
    extension_allowed = '.jpg'

    #create folders
    images_path = training_data_path + 'images/'
    if os.path.exists(images_path):
        shutil.rmtree(images_path)
    os.mkdir(images_path)

    labels_path = training_data_path + 'labels/'
    if os.path.exists(labels_path):
        shutil.rmtree(labels_path)
    os.mkdir(labels_path)

    training_images_path = images_path + 'train/'
    validation_images_path = images_path + 'val/'
    testing_images_path = images_path + 'test/'

    training_labels_path = labels_path + 'train/'
    validation_labels_path = labels_path +'val/'
    testing_labels_path = labels_path +'test/'

    os.mkdir(training_images_path)
    os.mkdir(validation_images_path)
    os.mkdir(testing_images_path)
    os.mkdir(training_labels_path)
    os.mkdir(validation_labels_path)
    os.mkdir(testing_labels_path)

    print("copying training data")
    with open(full_data_path + '../train.txt') as f:
        for line in f:
            img_name = line[:9]
            annotation_file = img_name + '.txt'
            src_label = ann_data_path + annotation_file
            print(src_label)
            if os.path.isfile(src_label):
                src_image = full_data_path + img_name + extension_allowed
                shutil.copy(src_image, training_images_path)                          
                shutil.copy(src_label, training_labels_path) 

    print("copying validation data")
    with open(full_data_path + '../val.txt') as f:
        for line in f:
            img_name = line[:9]
            annotation_file = img_name + '.txt'
            src_label = ann_data_path + annotation_file
            if os.path.isfile(src_label):
                src_image = full_data_path + img_name + extension_allowed
                shutil.copy(src_image, validation_images_path)                          
                shutil.copy(src_label, validation_labels_path)

    print("copying testing data")
    with open(full_data_path + '../test.txt') as f:
        for line in f:
            img_name = line[:9]
            annotation_file = img_name + '.txt'
            src_label = ann_data_path + annotation_file
            if os.path.isfile(src_label):
                src_image = full_data_path + img_name + extension_allowed
                shutil.copy(src_image, testing_images_path)                          
                shutil.copy(src_label, testing_labels_path)

if __name__ == "__main__":
    main()
