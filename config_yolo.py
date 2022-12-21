DATASET_FOLDER = '/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/synth/single/1k/raw-images/'
CSV1 = "{}../cameraLabel.csv".format(DATASET_FOLDER) # cf with camera
CSV2 = "{}../robotLabel.csv".format(DATASET_FOLDER)
CAMERA_PARAMS_YAML = '/home/akmaral/tubCloud/Shared/cvmrs/calibration_virtual.yaml' 
IMG_SIZE = (320, 320)
RADIUS = 0.045 # in meters
# PREDICTION = '/home/akmaral/tubCloud/Shared/cvmrs/training_dataset/synth/single/1k/yolov3/images/detections/synth-1k/labels/'
WEIGHTS = '/home/akmaral/tubCloud/Shared/cvmrs/trained-models/yolov3/synth-1k-best.pt'
DETECTION_FOLDER = "{}../yolov3/det/".format(DATASET_FOLDER)
TEST_IMAGES = "{}../yolov3/images/test/**/*.jpg".format(DATASET_FOLDER)
INFERENCE_FILE = "{}../yolov3/inference_yolo.yaml".format(DATASET_FOLDER)