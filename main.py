import object_detection_webcam as obj
import numpy as np

import cv2
import tensorflow as tf
# import com
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
from utils import visualization_utils as vis_util

model_name = '/root/krpai/inference_graph_model1/saved_model'
detection_model = obj.load_model(model_name)

#! menunggu sinyal arduino
cap = cv2.VideoCapture(2) 
obj.run_inference(detection_model,cap)
