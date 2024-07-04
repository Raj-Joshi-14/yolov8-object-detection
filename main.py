import cv2
from ultralytics import YOLO
import collections
import time

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Using a smaller model for faster inference

# Maintain a history of detected objects
label_history = collections.defaultdict(lambda: collections.deque(maxlen=5))

def non_maximum_suppression(boxes, confidences, iou_threshold):
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=iou_threshold)
    if len(indices) > 0:
        indices = indices.flatten()  # Flatten the indices list
    return indices

def detect_objects(image, conf_threshold=0.5, nms_threshold=0.4):
    results = model(image, conf=conf_threshold, iou=nms_threshold)
    
    # Get bounding boxes, class names, and confidences
    boxes = []
    confidences = []
    class_ids = []
    
    for result in results:
        for detection in result.boxes:
            box = detection.xyxy[0].cpu().numpy().astype(int).tolist()  # Bounding box
            confidence = float(detection.conf[0].cpu().numpy())  # Confidence score
            class_id = int(detection.cls[0].cpu().numpy())  # Class ID
            
            boxes.append(box)
            confidences.append(confidence)
            class_ids.append(class_id)
    
    indices = non_maximum_suppression(boxes, confidences, nms_threshold)
    
    final_boxes = [boxes[i] for i in indices]
    final_confidences = [confidences[i] for i in indices]
    final_class_ids = [class_ids[i] for i in indices]
    
    return final_boxes, final_confidences, final_class_ids

def draw_boxes(image, boxes, confidences, class_ids):
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[class_id]}: {confidence:.2f}'
        
        # Add current label to history
        label_history[class_id].append((label, confidence))
        
        # Calculate the most frequent label in the history
        labels = [lbl for lbl, _ in label_history[class_id]]
        most_frequent_label = max(set(labels), key=labels.count)
        
        color = (255, 0, 0)  # Blue color for bounding box
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Put the most frequent label above the bounding box
        cv2.putText(image, most_frequent_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image

def process_image(image_path):
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not read image.")
        return
    
    # Resize image to maintain aspect ratio
    height, width = image.shape[:2]
    aspect_ratio = width / height
    new_width = 640
    new_height = int(new_width / aspect_ratio)
    image = cv2.resize(image, (new_width, new_height))

    # Detect objects
    boxes, confidences, class_ids = detect_objects(image)
    # Draw bounding boxes
    image = draw_boxes(image, boxes, confidences, class_ids)
    
    # Print the detected objects after NMS
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        print(f'Detected {model.names[class_id]} with confidence {confidence:.2f}')
    
    cv2.imshow('YOLOv8 Object Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Resize frame to maintain aspect ratio
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = 990
        new_height = int(new_width / aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))

        # Detect objects
        boxes, confidences, class_ids = detect_objects(frame)
        # Draw bounding boxes
        frame = draw_boxes(frame, boxes, confidences, class_ids)
        
        # Print the detected objects after NMS
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            print(f'Detected {model.names[class_id]} with confidence {confidence:.2f}')
        
        cv2.imshow('YOLOv8 Object Detection', frame)
        
        # Calculate processing time and sleep to maintain frame rate
        processing_time = time.time() - start_time
        frame_rate = 30  # Target frame rate (frames per second)
        delay = max(1, int((1 / frame_rate - processing_time) * 1000))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def process_real_time(ip_address):
    cap = cv2.VideoCapture(ip_address)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # Resize frame to maintain aspect ratio
        height, width = frame.shape[:2]
        aspect_ratio = width / height
        new_width = 640
        new_height = int(new_width / aspect_ratio)
        frame = cv2.resize(frame, (new_width, new_height))

        # Detect objects
        boxes, confidences, class_ids = detect_objects(frame)
        # Draw bounding boxes
        frame = draw_boxes(frame, boxes, confidences, class_ids)
        
        # Print the detected objects after NMS
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            print(f'Detected {model.names[class_id]} with confidence {confidence:.2f}')
        
        cv2.imshow('YOLOv8 Object Detection', frame)
        
        # Calculate processing time and sleep to maintain frame rate
        processing_time = time.time() - start_time
        frame_rate = 30  # Target frame rate (frames per second)
        delay = max(1, int((1 / frame_rate - processing_time) * 1000))
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
# Process an image file
# process_image('img.jpg')

# Process a video file
process_video('videoplayback.webm')

# Process real-time video from webcam
#process_real_time('http://192.168.1.4:8080/video')
