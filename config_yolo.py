DATASET_FOLDER = '/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/real/yolov3/yolov3-real/'
CSV1 = "{}cameraLabel.csv".format(DATASET_FOLDER) # cf with camera
# CSV2 = "{}robotLabel.csv".format(DATASET_FOLDER)
CAMERA_PARAMS_YAML = '/home/akmaral/tubCloud/Shared/cvmrs/calibration_real_a7.yaml' 
IMG_SIZE = (320, 320)
RADIUS = 0.045 # in meters
PREDICTION = '/home/akmaral/IMRS/cv-mrs/baselines/yolov3/runs/detect/yolo_gn_real_3/labels/'