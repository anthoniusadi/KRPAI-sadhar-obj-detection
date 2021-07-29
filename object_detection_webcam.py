
import cv2
import tensorflow as tf
# import com
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
from utils import visualization_utils as vis_util
koordinat=[]
image_width = 640
image_height =480
label=""
import numpy as np
def kirim_koordinat():
    pass

def detect(frame):
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_HSV = np.array([0, 40, 0], dtype = "uint8")
    max_HSV = np.array([25, 255, 255], dtype = "uint8")
    skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    skinHSV = cv2.bitwise_and(frame, frame, mask = skinRegionHSV)    
    return skinHSV
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model
# cap = cv2.VideoCapture(0) # or cap = cv2.VideoCapture("<video-path>")/0 ,1,2 port webcam


PATH_TO_LABELS = '/root/krpai/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
model_name = '/root/krpai/inference_graph_model1/saved_model'
detection_model = load_model(model_name)

print(detection_model.signatures['serving_default'].inputs)
detection_model.signatures['serving_default'].output_dtypes
detection_model.signatures['serving_default'].output_shapes




def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis]
    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
        
    return output_dict
def run_inference(model, cap):
    global luas
    
    while cap.isOpened():
        
        ret, image_np = cap.read()
        # image_np = cv2.flip(image_np,1)
        # Actual detection.
        (frame_height, frame_width) = image_np.shape[:2]
        output_dict = run_inference_for_single_image(model, image_np)
        # Visualization of the results of a detection.
        temp,cat=vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            max_boxes_to_draw=1,
            min_score_thresh=.70,
            agnostic_mode=False,
            line_thickness=8)
        
        boxes = output_dict['detection_boxes']
        box = np.squeeze(boxes)
        for index, score in enumerate(output_dict['detection_scores']):
            if score < 0.8:
                label=""
                continue
            else:
                label = category_index[output_dict['detection_classes'][index]]['name']
                ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
                koordinat.append((label, int(xmin * image_width), int(ymin * image_height),int(xmax * image_width), int(ymax * image_height)))
                #!ambil koordinat
                ymin = (int(ymin * image_height))
                xmin = (int(xmin * image_width))
                xmax = (int(xmax * image_width))
                ymax = (int(ymax * image_height))
                panjangx = xmax-xmin
                panjangy = ymax-ymin
                luas = panjangx*panjangy
                
                center_x=xmin+((xmax-xmin)//2)
                center_y=ymin+((ymax-ymin)//2)
                print('X = {} Y = {} ,luas : {} '.format(center_x,center_y,luas))
                #! crop ROI image 
                roi =image_np[ymin:ymax,xmin:xmax].copy()
                cv2.imshow('roi',roi)
                image_np=cv2.circle(image_np,(center_x,center_y),2,(10, 10, 220),8)
        #!kirim koordinat lewat komunikasi Serial
                # com.kirim(center_x,center_y)
                if(luas>=155300):
                    # com.kirim(center_x,center_y)
                    print("selesai")
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                    
        # if (cat == "human"):
        #     for i in range(len(boxes)):
        #         ymin = (int(box[i,0]*frame_height))
        #         xmin = (int(box[i,1]*frame_width))
        #         ymax = (int(box[i,2]*frame_height))
        #         xmax = (int(box[i,3]*frame_width))
        #         print(ymin,xmin,ymax,xmax)
        #         #! crop ROI image 
        
            
            roi =image_np[ymin:ymax,xmin:xmax].copy()
        # segmented = detect(image_np)
            cv2.imshow('roi',roi)
        # cv2.imshow('segmented',segmented)
        cv2.imshow('object_detection', cv2.resize(image_np, (640, 480)))
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

# run_inference(detection_model, cap)
