import cv2
import numpy as np

from collections import defaultdict



def classify_motion(angle_t, mag_t, avg_magnitude, average_angle, prev_classifications, fps):
    # angle_threshold = 1.8  # Increase angle sensitivity for differentiating turns
    # magnitude_threshold = 1.2  # Adjust for more accurate movement detection
    
    # Define a threshold for detecting consistent movement over time
    consistent_frames_threshold = 5 # At least 0.5 seconds of consistent movement
    # Initialize default class as forward movement
    # Initialize default class as forward movement
    motion_class = 'forward'
    
    # Use moving averages for smoother classification of turns
    if avg_magnitude > mag_t:
        if average_angle > angle_t:
            # Detect a right turn if the motion is consistent over time
            if len(prev_classifications) >= consistent_frames_threshold and \
               all(clas == 'left turn' for clas in prev_classifications[-consistent_frames_threshold:]):
                motion_class = 'left turn'
            else:
                # Otherwise, classify as right lane change
                motion_class = 'left turn'
        elif average_angle < -angle_t:
            # Detect a left turn if the motion is consistent over time
            if len(prev_classifications) >= consistent_frames_threshold and \
               all(clas == 'right turn' for clas in prev_classifications[-consistent_frames_threshold:]):
                motion_class = 'right turn'
            else:
                # Otherwise, classify as left lane change
                motion_class = 'right turn'
        else:
            motion_class = 'forward'
    else:
        motion_class = 'forward'

    # print( avg_magnitude, average_angle, motion_class)
    return motion_class

def rle_encode_label_matrix(label_matrix):
    # Flatten row-major (C-order)
    flat = label_matrix.flatten()

    # Initialize lists
    labels = []
    lengths = []

    # Initialize first run
    prev_label = flat[0]
    count = 1

    for label in flat[1:]:
        if label == prev_label:
            count += 1
        else:
            labels.append(prev_label)
            lengths.append(count)
            prev_label = label
            count = 1

    # Append last run
    labels.append(prev_label)
    lengths.append(count)

    # Return as list of tuples
    return list(zip(labels, lengths))

def convert_ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray_to_list(i) for i in obj]
    else:
        return obj


def summarize_optical_flow(angle_t, mag_t, video_path):
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    motion_summary = []
    prev_classifications = []
    video_id = video_path.split('/')[-1]

    ########### Writing optical flow video ######################
    height, width, _ = prev_frame.shape
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine the start frame for the last one second of the video
    # start_frame = total_frames - 10 #for brain for cars
    start_frame = total_frames - 10 #for DAAD Front view for cars
    # start_frame = 0

    # If the video is shorter than one second, start from the beginning
    if start_frame < 0:
        start_frame = 0
    
    # Set the current frame position to the start frame for the last second
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    out = cv2.VideoWriter('/scratch/sai/optical_flow/optical_flow_output.mp4', fourcc, fps, (width, height))

    indx=0
    dtop[video_id] = dict()

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # if i%20!=0:
        #     continue
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        H, W , _ = flow.shape

        # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        direction_labels = np.empty((H, W), dtype=object)

        # Masks for movement
        moving_right_mask = flow[..., 0] > 0
        moving_left_mask = flow[..., 0] < 0
        no_motion_mask = flow[..., 0] == 0

        # Assign labels
        direction_labels[moving_right_mask] = 'right'
        direction_labels[moving_left_mask] = 'left'
        direction_labels[no_motion_mask] = 'none'

        # Compute magnitude and angle of 2D flow vectors\
        
        ph, pw = 50, 50

        patch_h = H // ph
        patch_w = W // pw
        dominant_labels = np.empty((patch_h, patch_w), dtype=object)

        # Loop through patches and compute dominant label
        for i in range(patch_h):
            for j in range(patch_w):
                patch = direction_labels[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                flat_patch = patch.flatten()
                counts = Counter(flat_patch)
                dominant = counts.most_common(1)[0][0]
                dominant_labels[i, j] = dominant


      
        
        # rle = rle_encode_label_matrix(direction_labels)
        indx+=1

        frame = f'frame_{indx}'

        dtop[video_id][frame] = dominant_labels

        # import pdb;pdb.set_trace()
        # avg_magnitude = np.mean(mag)
        # avg_angle = np.mean(ang)

        ############# Visualize the optical flow #####################
        # hsv = np.zeros_like(frame)
        # hsv[..., 1] = 255  # Full saturation for better color
        # hsv[..., 0] = ang * 180 / np.pi / 2  # Hue representing direction (angle of flow)
        # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Value represents magnitude of flow

        # # Convert HSV image to BGR for visualization
        # flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # # Write the frame to the output video
        # out.write(flow_bgr)

        # # Calculate the average direction angle more robustly
        # avg_angle_x = np.mean(np.cos(avg_angle))  # Horizontal component
        # avg_angle_y = np.mean(np.sin(avg_angle))  # Vertical component
        # average_angle = np.arctan2(avg_angle_y, avg_angle_x)  # Average direction of movement

        # # Classify the movement based on magnitude, direction, and consistency over time
        # motion_class = classify_motion(angle_t, mag_t, avg_magnitude, average_angle, prev_classifications, fps)
        
        # motion_summary.append(motion_class)
        # prev_classifications.append(motion_class)

        # prev_gray = gray_frame  # Update the previous frame for the next iteration
    # import pdb;pdb.set_trace()
    
    cap.release()
    out.release()

    return motion_summary


import pandas as pd
from collections import Counter
import json

# file = pd.read_csv('/scratch/sai/brain4cars_data/train.csv')
file = pd.read_csv('/scratch/sai/train.csv')


angle_thresholds = [1.0, 1.2, 1.5, 1.8, 2.0]#
magnitude_thresholds = [0.8, 1.0, 1.2, 1.4, 1.6]# 

a = set()

dtop = dict()
optical_flow_output = open('optical_flow_cropped.json', 'w')

for angle_t in angle_thresholds:
    for mag_t in magnitude_thresholds:
        # classes = {0:'right turn', 1:'right lane change', 2:'left turn', 3:'left lane change', 4:'forward'}
        # count = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
        # total = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
        classes = {0:'straight', 1:'slow down', 2:'left turn', 3:'left lane change', 4:'right turn', 5:'right lane change', 6:'u turn'}
        count = {'straight':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
        total = {'straight':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
        att = dict()
        for index, row in file.iterrows():
            filename = row['filename'].split('/')[-1]  # Access 'filename' column
            class_label = row['class']  # Access 'class' column
            if '.mp4' not in filename:
                continue
            
            
            #only required for brain4cars
            # filename = filename.split('/')[-1]
            # filename = filename.split('.')[0]
            if class_label==1 or class_label==0 or class_label==3 or class_label==5 or class_label==6:
                continue

            # if class_label==0 or class_label==1 or class_label==3 or class_label==5 or class_label==6:
            #     continue
            # if '0b75f188-f15e-446c-a83e-91ebc6ae7941' not in filename:
            #     continue
            
            import os
            if not os.path.exists(f'/scratch/sai/front_view_cropped/{filename}'):
                continue
            
            print(f'''/scratch/sai/front_view_cropped/{filename}''')
            motion = summarize_optical_flow(angle_t, mag_t, f'''/scratch/sai/front_view_cropped/{filename}''')
            if motion == []:
                continue
            most_common_action = Counter(motion).most_common(1)[0][0]
            if classes[class_label] == most_common_action:
                count[most_common_action]+=1
                att[filename+'.mp4'] = most_common_action
            else:
                i=0
                print(f'Ground Truth: {classes[class_label]} , predcitied: {most_common_action}')

            total[classes[class_label]]+=1
            a.add(filename)
            

        # predictions = get_predictions(angle_t, mag_t)  # Your logic here
        # ac    c = compute_accuracy(predictions, ground_truth)
        dtop_clean = convert_ndarray_to_list(dtop)
        json.dump(dtop_clean , optical_flow_output)

        acc = (sum(count.values()))/(sum(total.values()))
        import pdb;pdb.set_trace()
        print(f"Angle: {angle_t}, Mag: {mag_t} => Accuracy: {acc}")
        


import pdb;pdb.set_trace()