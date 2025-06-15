import os       #Interacts with the operating system
import sys      #Provides functions and variables that manipulate different parts of the Python runtime environment
import argparse     #Provides argument parsing functionality    
import glob     #Searches for file path names that match a specific pattern
import time     #Counts the time and frames during use
import csv      #Creates the csv files for the detection_log
from pathlib import Path        # Used for creating and pathing files/ folders

import cv2      #This library might have a warning listed next to it if opened in an IDE. Don't worry, as this script will be run in a CLI 
import numpy as np      #This library might have a warning listed next to it if opened in an IDE. Don't worry, as this script will be run in a CLI 
from ultralytics import YOLO        #This library might have a warning listed next to it if opened in an IDE. Don't worry, as this script will be run in a CLI 


# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/Warehouse_Model.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), index of a USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, the script will match the source resolution',
                    default=None)
parser.add_argument('--screenshot_interval', help='Set the screenshot interval in frames for making a new dataset (e.g. 60)', type=int, default=60)

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
screen_inter = args.screenshot_interval

#Screenshot settings for creating future datasets for embodied learning functions
screenshot_dir = 'Screenshots'
os.makedirs(screenshot_dir, exist_ok=True)

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get the labelmap 
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if the image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

#Uses of parsers to determine what input source is being used, and sets the variable
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified the display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])


# Load or initialize image or video source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

# Set bounding box colors (using the Tableu 10 color scheme) There are 7 classes created, however there is room for a total of ten different colours, (You can easily add more).
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

#csv Logging for embodied learning
log_file = open('detection_log.csv', mode= 'w', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Frame', 'FPS'] + list(labels.values()))
frame_index = 0


# Begin the inference loop
while True:
    #Calculates the FPS for each frame
    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame (YOLO model processes)
    results = model(frame, verbose=False)

    # Extract results (Detects the results)
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    class_counts = {}

    #Csv logging (creates a dictionary to count objects of each class in the current frame)
    class_counts = {label: 0 for label in labels.values()}

    # Go through each detection and get the bounding box coordinates, confidence, and class
    for i in range(len(detections)):

        # Get the bounding box coordinates
        # Ultralytics returns results in a Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in the CPU memory (This is because NumPy can't work on GPU, so this block converts it to CPU)
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to a Numpy array and flattens it (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract the individual coordinates and convert them to int so that OpenCv can draw it

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Get the bounding box confidence
        conf = detections[i].conf.item()

        # Draw the bounding box if confidence threshold is high enough
        if conf > 0.5:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1) # Get the font size (The Hershey Duplex font is included within the OpenCV Library)
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw a white box to put the label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1) # Draw the label text

            # Count the objects by their classes in the image
            class_counts[classname] = class_counts.get(classname, 0) + 1

    # Calculate and display the framerate to the screen (This is if you're using a video input or a camera)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10,20), cv2.FONT_HERSHEY_DUPLEX, .7, (0,255,255), 2) # Display the framerate to the screen
    
    # Display the detection results
    y_offset = 40
    for classname, count in class_counts.items():
        cv2.putText(frame, f'{classname}: {count}', (10, y_offset), cv2.FONT_HERSHEY_DUPLEX, .7, (0,255,255), 2) #Display the detection results: FPS and Object class counts
        y_offset += 20
    #Csv logging for embodied learning
    row = [frame_index, avg_frame_rate]
    for class_name in labels.values():
        row.append(class_counts.get(class_name, 0))
    csv_writer.writerow(row)
    frame_index += 1

    #Executes the set interval for the screenshot capture for embodied learning
    if frame_index % screen_inter == 0:
        print(f"[DEBUG] Trying to save screenshot at frame {frame_index}...") #Wrote in this debug because there were a lot of issues when saving images for the embodied learning function

        if frame is None:
            print(f"[ERROR] Frame is None at frame {frame_index}, skipping screenshot.")
        else:
            screenshot_path = os.path.join(screenshot_dir, f'frame_{frame_index:05d}.png')
            success = cv2.imwrite(screenshot_path, frame)

            if success:
                print(f"Screenshot saved: {screenshot_path}")
            else:
                print(f"[ERROR] Failed to save screenshot at: {screenshot_path}")

    cv2.imshow('YOLO detection results',frame) # Display the image
   

    # If inferencing on individual images, wait for the user keypress before moving to the next image. Otherwise, wait 5ms before moving to the next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # You can use the 'q' key to stop the inference
        break
    elif key == ord('s') or key == ord('S'): # You can use the 's' key to pause the inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # You can use the 'p' key to save a picture of results on this frame manually
        cv2.imwrite('capture.png',frame)
    
    # Calculate the FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append the FPS result to frame_rate_buffer (for finding the average FPS across several frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate the average FPS for previous frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
#csv logging
log_file.close()
#closing
cv2.destroyAllWindows() 
