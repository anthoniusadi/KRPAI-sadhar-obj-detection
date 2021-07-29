######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/20/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier and uses it to perform object detection on a webcam feed.
# It draws boxes, scores, and labels around the objects of interest in each frame
# from the webcam.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'saved_model.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# !rescale 
video = cv2.VideoCapture(0)
def rescale_frame(frame,percent=75):
    width=int(frame.shape[1]* percent/100)
    height=int(frame.shape[0]* percent/100)
    dim = (width,height)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
def detect(frame):
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_HSV = np.array([0, 40, 0], dtype = "uint8")
    max_HSV = np.array([25, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    skinHSV = cv2.bitwise_and(frame, frame, mask = skinRegionHSV)    
    return skinHSV
while(True):
    ret, fr = video.read()
    image = fr
    frame = rescale_frame(fr,percent=100)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
  #  print(boxes,classes)

#    cropped_image = tf.image.crop_to_bounding_box(frame, yminn, xminn, ymaxx - yminn, xmaxx - xminn)    
    ## Roi pada object yang di inginkan
    (frame_height, frame_width) = image.shape[:2]
##    for i in range(len(np.squeeze(scores))):
##        
##        ymin = int((np.squeeze(boxes)[i][0]*frame_height))
##        xmin = int((np.squeeze(boxes)[i][1]*frame_width))
##	
##        ymax = int((np.squeeze(boxes)[i][2]*frame_height))
##        xmax = int((np.squeeze(boxes)[i][3]*frame_width))
##
    # ! DRAW 
    temp,cat=vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.50)
    #! mengambil koordinat pada bounding box
    box = np.squeeze(boxes)
  #  print("cat = ",cat)
    if (cat == "hand"):
        for i in range(len(boxes)):
            ymin = (int(box[i,0]*frame_height))
            xmin = (int(box[i,1]*frame_width))
            ymax = (int(box[i,2]*frame_height))
            xmax = (int(box[i,3]*frame_width))
         #   print(ymin,xmin,ymax,xmax)
            #! crop ROI image 
            roi =image[ymin:ymax,xmin:xmax].copy()
            
            segmented = detect(roi)
            cv2.imshow('roi',roi)
            cv2.imshow('segmented',segmented)
    else:
        cv2.imshow("kosong",frame)
    
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)
# Roi object
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

