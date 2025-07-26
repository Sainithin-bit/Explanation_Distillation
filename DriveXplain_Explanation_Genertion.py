import transformers
import torch

import pandas as pd
import csv
import re
import json

from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

##################################################DAAD################################

with open('testing_set_inputs/DAAD_LLAMA/videollama_daad_front_view.txt', 'r') as file:
    file_content = file.read()
    lines = file_content.strip().splitlines()

current_video = None
sentences = []
video_responses = {}
# Loop through the lines of the input data
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespaces
    if line.endswith(".avi"):  # If the line is a video filename
        if current_video and sentences:
            # Store the previous video and its corresponding sentences in the dictionary
            current_video = current_video.split('/')[-1]
            video_responses[current_video] = sentences
        
        # Start a new video entry
        current_video = line
        sentences = []  # Reset the sentences list for the new video
    elif line and current_video:  # If the line is a sentence and there's a current video
        sentences.append(line)  # Add the sentence to the list



classes = {0:'forward', 1:'slow down', 2:'left turn', 3:'left lane change', 4:'right turn', 5:'right lane change', 6:'u turn'}
count = {'forward':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
count_1 = {'forward':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
count_2 = {'forward':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
global_count = {'forward':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}
total = {'forward':0, 'slow down':0, 'left turn':0, 'left lane change':0, 'right turn':0, 'right lane change':0, 'u turn':0}

import pandas as pd
from collections import defaultdict
import re 
import json

gd = pd.read_csv('/scratch/sai/train.csv')
d = defaultdict(list)
caption_dict = dict()

optical_flow = json.load(open('testing_set_inputs/DAAD_LLAMA/optical_flow_output_daad.json', 'r'))
lane_change = json.load(open('testing_set_inputs/DAAD_LLAMA/lane_change_daad_patch.json', 'r'))
narratives = json.load(open('testing_set_inputs/DAAD_LLAMA/detections_daad.json', 'r'))

groundtruth, predicted = [] ,  []
ans='left turn'
indx = 0

# file_path = '/scratch/sai/LLAVA_DAAD_Front_view.txt'
file_path = 'testing_set_inputs/DAAD_LLAMA/LLAVA_DAAD_Front_view.txt'
# file_path = '/scratch/sai/output_brain4cars_llava.txt'


file_data = defaultdict(list)
file_id = None


with open(file_path, 'r') as file:
    file_content = file.read()
    lines = file_content.strip().splitlines()
    for line in lines:
        line = line.strip()
        
        if '/home/egoexo_anno/front_view_frames' in line:  # Identify the file path (ID)
            file_id = line.split('/')[-1] + '.mp4'
            if file_id not in file_data:
                file_data[file_id] = []  # Create a list to hold captions
        elif line.startswith("<s>"):  # If line contains caption text
            if file_id:
                file_data[file_id].append('Frame caption :' + line)



for filename, response in video_responses.items():

    if any(gd['filename']==filename):
        print(filename)
        flag = False
        ans = classes[gd[gd['filename']==filename]['class'].to_list()[0]]
    else:
        continue
    # if total[ans]>=50:
    #     print(total)
    #     continue

    
    
    new_rep = []
    for res in response:
        if ".txt" in res:
            continue
        else:  
            res = re.sub(r'\d+: #C', '', res)
            new_rep.append(res)

    input_prompt = ''.join(new_rep)

    filename = filename.split('/')[-1]
    
    try:
        if file_data[filename]==[]:
           continue
    except:
        continue
    
    video_id = filename.split('.')[0]

    try:
        if lane_change[video_id]==[]:
            continue
    except:
        continue

    # if indx>20:
    #     break
    indx+=1
    ############################### Phase1 ######################################################
    try:
        if optical_flow[filename] and narratives[filename]:
            messages_1 = [
    {
        "role": "system",
        "content": """

        You are an expert tasked with classifying driving maneuvers based on provided inputs.
        Your goal is to classify the input_text into the correct driving maneuver.

        Use the full context provided,  paying attention to direction, speed, and action verbs to select the most appropriate label. Be concise and only output the label.
        Avoid unnecessary explanations or additional information. Your task is to analyze multimodal driving data from a driving video and generate both:

        1. A high-level **driving maneuver label** that best summarizes the ego vehicle's most significant action.
        2. A detailed **natural language explanation** of why this label was chosen, based on available evidence.

        {
        "label": "<One of: forward, slow down, left turn, left lane change, right turn, right lane change, u turn>",
        "explanation": "A detailed explanation justifying the selected label."
        }
"""
    },
    {
        "role": "user",
        "content": f"""


Basic Instruction: Analyze the Frame-wise Captions, Video-wide Caption, Optical Flow  Context, Surrounding context and Frame-wise Lane Segmentation Context to classify the described driving maneuver. Then provide Label, Explantion1, Explantion2, Explantion3, Explantion4, Explantion5 different detailed explanation justifying your choice.



Task and Label Descriptions:
- 'forward': The vehicle continues moving forward without turning or changing lanes.
- 'slow down': The vehicle decelerates or prepares to stop.
- 'left turn': The vehicle turns sharply or significantly to the left.
- 'left lane change': The vehicle moves into the left lane.
- 'right turn': The vehicle turns sharply or significantly to the right.
- 'right lane change': The vehicle moves into the right lane.
- 'u turn': The vehicle makes a U-shaped turn.

Constraints:
- Only reply with one of the following labels: forward, slow down, left turn, left lane change, right turn, u turn, right lane change.



### Example 1

**Context**  
"frame_0": "others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane road road lane lane road", 
"frame_60": "others others others others others others others others others others others others others others others road road road road road others lane lane road road road road road", 
"frame_160": "others others others others others others others others others others others others others others others others lane lane road lane lane others lane lane road road road road", 
"frame_240": "others others others others others others others others others others others others others others others lane lane lane others lane lane lane lane road road lane lane road", 
"frame_320": "others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane road road lane road others", 
"frame_380": "others others others others others others others others others others others others others others others lane lane lane lane lane lane road road road road road road lane",
"frame_400": "others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane road road road road lane"
**Label**  — right lane change


### Example 2

**Context**  
"frame_0": "others others others others others others others others others others others others others others others others road lane lane others others road road road road lane lane others", 
"frame_60": "others others others others others others others others others others others others others others others road road lane lane others others road road road road lane lane lane", 
"frame_180": "others others others others others others others others others others others others others others road lane lane lane lane others others road road road road lane lane lane", 
"frame_320": "others others others others others others others others others others others others others others lane lane lane lane lane others others lane lane lane road lane lane lane", 
"frame_420": "others others others others others others others others others others others others others others others others lane lane lane lane others others others road road lane lane lane", 
"frame_500": "others others others others others others others others others others others others others others lane lane lane lane lane lane others road road road road lane lane lane", 
"frame_560": "others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane lane road lane lane lane"
**Label** - left lane change


### Example 3

**Context**  
"frame_0": "others others others others others others others others others others others others others others others others others lane lane lane others lane lane lane lane lane lane lane", 
"frame_80": "others others others others others others others others others others others others others others others others others lane lane others others lane lane lane lane lane lane lane", 
"frame_160": "others others others others others others others others others others others others others others others others others lane road others others lane lane lane lane lane lane lane", 
"frame_240": "others others others others others others others others others others others others others others others others road road road others others lane lane lane road road lane lane", 
"frame_300": "others others others others others others others others others others others others others others others others road road lane lane road road road road road road road lane", 
"frame_360": "others others others others others others others others others others others others others others road road road road road lane lane road road road road road lane lane", 
"frame_420": "others others others others others others others others others others others others others others road lane road road road road road lane road road road road road road", 
"frame_480": "others others others others others others others others others others others others others others others others road road road road road road road road road road road road", 
"frame_540": "others others others others others others others others others others others others others others others others road lane lane others others road road road road lane road road", 
"frame_560": "others others others others others others others others others others others others others others others others others lane lane others others others road road road lane lane road"
**Label** - left lane change



---

Here is the multimodal input for the video:

- **Frame-wise Lane Segmentation Context**  
{lane_change[video_id]}

- **Frame-wise Captions**  
{file_data[filename]}

- **Video-wide Caption**  
{input_prompt}

- **Surrounding context**  
{narratives[filename]}


### Example 1

**Context**  

"frame_6": [["left", "left", "left", "left", "right", "right", "right", "right", "right"], ["left", "left", "left", "left", "left", "right", "right", "right", "left"], ["left", "left", "left", "left", "left", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"]], 
"frame_7": [["left", "left", "left", "left", "right", "right", "right", "right", "right"], ["left", "left", "left", "left", "left", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"]], 
"frame_8": [["left", "left", "left", "left", "right", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"]], 
"frame_9": [["left", "left", "left", "left", "right", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"]], 
"frame_10": [["left", "left", "left", "left", "right", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "left"], ["left", "left", "right", "left", "left", "left", "left", "left", "left"], ["right", "right", "left", "left", "left", "left", "left", "left", "left"], ["right", "left", "right", "left", "left", "left", "left", "left", "left"]]

**Label** — left turn

### Example 2

**Context**  
"frame_6": [["right", "right", "right", "right", "right", "right", "left", "right", "left"], ["right", "right", "right", "right", "right", "right", "left", "left", "left"], ["right", "right", "right", "right", "right", "left", "right", "right", "left"], ["right", "right", "right", "right", "left", "left", "left", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"]], 
"frame_7": [["right", "right", "right", "right", "right", "right", "left", "right", "left"], ["right", "right", "right", "right", "right", "right", "right", "left", "left"], ["right", "right", "right", "right", "right", "left", "right", "right", "left"], ["right", "right", "right", "right", "left", "left", "left", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"]], 
"frame_8": [["right", "right", "right", "right", "right", "right", "left", "right", "left"], ["right", "right", "right", "right", "right", "right", "right", "left", "left"], ["right", "right", "right", "right", "right", "left", "right", "right", "left"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"]], 
"frame_9": [["right", "right", "right", "right", "right", "right", "left", "right", "left"], ["right", "right", "right", "right", "right", "right", "right", "left", "left"], ["right", "right", "right", "right", "right", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"]], 
"frame_10": [["right", "right", "right", "right", "right", "right", "left", "right", "left"], ["right", "right", "right", "right", "right", "right", "right", "right", "left"], ["right", "right", "right", "right", "right", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"]]
**Label** — left turn

### Example 3

**Context**  
"frame_6": [["right", "right", "right", "right", "left", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "left"], ["right", "right", "right", "right", "right", "right", "right", "left", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "left"]], 
"frame_7": [["right", "right", "right", "right", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "left", "left"], ["right", "right", "right", "right", "right", "right", "left", "left", "left"]], 
"frame_8": [["right", "right", "right", "left", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "left"]], 
"frame_9": [["right", "right", "right", "left", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "left", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "left"]], 
"frame_10": [["right", "right", "right", "left", "left", "left", "right", "right", "right"], ["right", "right", "right", "left", "left", "right", "right", "right", "right"], ["right", "right", "right", "right", "left", "right", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "right"], ["right", "right", "right", "right", "right", "right", "left", "left", "left"]]
**Label** — right turn        

### Example 4

**Context**  
"frame_6": [["left", "left", "left", "right", "left", "right", "right", "left", "left"], ["left", "left", "left", "right", "left", "right", "left", "left", "left"], ["left", "left", "right", "right", "left", "left", "right", "left", "right"], ["right", "right", "right", "right", "left", "left", "right", "right", "right"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"]], 
"frame_7": [["left", "left", "left", "right", "left", "right", "right", "left", "left"], ["left", "left", "left", "right", "left", "right", "left", "left", "left"], ["left", "left", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "left", "left", "right", "left", "left"], ["right", "right", "right", "right", "right", "right", "right", "right", "right"]], 
"frame_8": [["left", "left", "left", "right", "left", "right", "right", "left", "left"], ["left", "left", "left", "right", "left", "right", "left", "left", "left"], ["left", "left", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "right", "right", "right", "left", "left"]], 
"frame_9": [["left", "left", "left", "right", "left", "right", "right", "left", "left"], ["left", "left", "left", "right", "left", "right", "left", "left", "left"], ["left", "left", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "right", "left", "left", "left", "left"]], 
"frame_10": [["left", "left", "left", "right", "left", "right", "right", "left", "left"], ["left", "left", "left", "right", "left", "right", "right", "left", "left"], ["left", "left", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "left", "left", "left", "left", "left"], ["right", "right", "right", "right", "right", "left", "left", "left", "left"]]
**Label** — right turn        


- **Optical Flow  Context**  
{optical_flow[filename]}


Output:
Label: <one of the 7 maneuver labels>  
Explanation1: <Justification>
Explanation2: <Justification>
Explanation3: <Justification>
Explanation4: <Justification>
Explanation5: <Justification>

"""
    }
]
    outputs = pipeline(
    messages_1,
    max_new_tokens=256,
    )

    caption_2 = outputs[0]["generated_text"][-1]['content']

    label_prefix = 'Label'
    explanation_prefix = 'Explanation'

    

    label_start = caption_2.find(label_prefix)
    explanation_start = caption_2.find(explanation_prefix)

    # Extract label
    temp_label = caption_2[label_start + len(label_prefix):explanation_start].strip()

    # Extract explanation
    explanation = caption_2[explanation_start :].strip()
    label = ''
    for _, clss in classes.items():
        if clss in temp_label:
            label = clss
            break
    

    if any(gd['filename']==filename):
        flag = False
        ans = classes[gd[gd['filename']==filename]['class'].to_list()[0]]
        if ans in label.lower():
            count[ans] += 1
            flag = True
            predicted.pop()
            groundtruth.pop()
            caption_dict[filename] = caption_2
            predicted.append(label.lower())
            groundtruth.append(ans)

        else:
            print(label, ans)
        
        d['filename'].append(filename) 
        d['Predcited'].append(label)
        d['Ground Truth'].append(ans)
        d['Match'].append(flag)

global_count['forward'] = max([count_1['forward'], count['forward']])
global_count['left lane change'] =max([count_1['left lane change'], count['left lane change'], count_2['left lane change']]) 
global_count['right lane change'] = max([count_1['right lane change'], count['right lane change'], count_2['right lane change']])
global_count['left turn'] = count['left turn']
global_count['right turn'] = count['right turn']

df = pd.DataFrame(d)

# Step 2: Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)

import json
with open('captions_daad.json', 'w') as caption_file:
    json.dump(caption_dict, caption_file, indent=4)



class_names = ['forward', 'slow down', 'left turn', 'left lane change', 'right turn', 'right lane change', 'u turn' ]

cm = confusion_matrix(groundtruth, predicted, labels=class_names)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix as an image file
plt.savefig('confusion_matrix.png')  # Save as PNG
plt.close()  # Close the plot



acc = (sum(global_count.values()))/(sum(total.values()))
print(total)
print(count)
print(count_1)
print(count_2)
print(acc)

from sklearn.metrics import f1_score
f1 = f1_score(groundtruth, predicted, average='weighted')

print(f1)

import pdb; pdb.set_trace()



####################################################Barin4Cars##################################


with open('testing_set_inputs/B4c_LLAMA/videollama.txt', 'r') as file:
    file_content = file.read()
    lines = file_content.strip().splitlines()

current_video = None
sentences = []
video_responses = {}
# Loop through the lines of the input data
for line in lines:
    line = line.strip()  # Remove leading/trailing whitespaces
    if line.endswith(".avi"):  # If the line is a video filename
        if current_video and sentences:
            # Store the previous video and its corresponding sentences in the dictionary
            current_video = current_video.split('/')[-1]
            video_responses[current_video] = sentences
        
        # Start a new video entry
        current_video = line
        sentences = []  # Reset the sentences list for the new video
    elif line and current_video:  # If the line is a sentence and there's a current video
        sentences.append(line)  # Add the sentence to the list

optical_flow = json.load(open('testing_set_inputs/B4c_LLAMA/optical_flow_output_b4c.json', 'r'))
narratives = json.load(open('testing_set_inputs/B4c_LLAMA/detections_centerTrack.json', 'r'))
lane_change = json.load(open('testing_set_inputs/B4c_LLAMA/lane_change_patch_latest.json', 'r'))

# file_path = '/scratch/sai/LLAVA_DAAD_Front_view.txt'
# file_path = '/scratch/sai/DAAD_LLAMA/LLAVA_DAAD_Front_view.txt'
file_path = 'testing_set_inputs/B4c_LLAMA/output_brain4cars_llava.txt'


file_data = defaultdict(list)
file_id = None


with open(file_path, 'r') as file:
    file_content = file.read()
    lines = file_content.strip().splitlines()
    for line in lines:
        line = line.strip()
        
        if '/home/egoexo_anno/front_view_frames' in line:  # Identify the file path (ID)
            file_id = line.split('/')[-1] + '.mp4'
            if file_id not in file_data:
                file_data[file_id] = []  # Create a list to hold captions
        elif line.startswith("<s>"):  # If line contains caption text
            if file_id:
                file_data[file_id].append('Frame caption :' + line)


# detections = []
# with open('/scratch/sai/CenterTrack/src/detections.json', 'r') as f:
#     for line in f:
#         line = line.strip()
#         import pdb;pdb.set_trace()
#         if line:
#             detections.append(json.loads(line))



d = defaultdict(list)

with open('/scratch/sai/BDD-Instruct-desc.json', 'r') as gt_file:
    gt = json.load(gt_file)

gt_ = {item['video_id']: item['QA']['q'] for item in gt}
caption_dict = dict()

predicted = []
groundtruth = []
global_set = set()




classes = {0:'right turn', 1:'right lane change', 2:'left turn', 3:'left lane change', 4:'forward'}
global_count_1 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
count_1 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
count_2 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
count_3 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
total_1 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
total_2 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }
total_3 = {'forward':0, 'left lane change':0, 'left turn':0, 'right lane change':0, 'right turn':0 }

gd = pd.read_csv('/scratch/sai/brain4cars_data/train.csv')
gd['filename']=gd['filename'].apply(lambda x: x.split('/')[-1])
caption_file = open('captions.txt', 'w')

captions = json.load(open('/scratch/sai/vlms/captions_llava_next_b4c.json', 'r'))



d = defaultdict(list)
# d['filename'] =
# d['Predcited']=''
# d['Ground Truth']=''

with open('/scratch/sai/BDD-Instruct-desc.json', 'r') as gt_file:
    gt = json.load(gt_file)

gt_ = {item['video_id']: item['QA']['q'] for item in gt}
caption_dict = dict()

predicted = []
groundtruth = []
global_set = set()

import pdb;pdb.set_trace()

for filename, response in video_responses.items():

    new_rep = []
    print(filename)
    
    for res in response:
        if ".txt" in res:
            continue
        else:  
            res = re.sub(r'\d+: #C', '', res)
            new_rep.append(res)

    input_prompt = ''.join(new_rep)

    filename = filename.split('/')[-1]
    video_id = filename.split('.')[0]

    
    try:
        # if file_data[filename]==[]:
        #    continue

        # import pdb;pdb.set_trace()
        try:
            if optical_flow[filename] and narratives[filename] and lane_change[video_id]:
                messages_1 = [
        {
            "role": "system",
            "content": """

    You are an expert in interpreting driving behavior. Your task is to classify the ego vehicle’s most significant maneuver by analyzing how it behaves — including movement patterns, interaction with surrounding objects, and changes in speed or position.

    Your responsibilities:
    1. Assign a **driving maneuver label** that best summarizes the ego vehicle’s dominant action.
    2. Write a **detailed, human-like explanation** that clearly supports your choice based on how the vehicle behaves and **why it behaves that way** — e.g., reacting to traffic, preparing to turn, overtaking, merging, etc.

    You must NOT mention any input source such as:
    - Captions (video-wise or frame-wise)
    - Optical flow
    - Lane-change context
    - Sensor or textual data
    - Frame-by-frame or analysis process

    Instead, your explanation should sound like a direct observation, grounded in reasoning based on:
    - Vehicle motion (e.g., turning, slowing down, lane shifting)
    - Interaction with surrounding vehicles, road structure, signs, or traffic
    - Implicit intent (e.g., avoiding a car, merging, preparing to stop)

    Your output must be in this **exact JSON format**:

    ```json
    {
    "label": "<one of: forward, slow down, left turn, left lane change, right turn, right lane change, u turn>",
    "Explanation": "<Detailed and human-like reasoning about what the vehicle is doing and why — based solely on behavior and context, without mentioning input sources>"
    }

    """
        },
        {
            "role": "user",
            "content": f"""


    Basic Instruction: Analyze the Frame-wise Captions and video-wise captions to classify the described driving maneuver. Then provide Label, Explantion1, Explantion2, Explantion3, Explantion4, Explantion5 different detailed explanation justifying your choice.

    Task and Label Descriptions:
    - 'right turn': The vehicle turns sharply or significantly to the right.
    - 'right lane change': The vehicle moves into the right lane.
    - 'left turn': The vehicle turns sharply or significantly to the left.
    - 'left lane change': The vehicle moves into the left lane.
    - 'forward': The vehicle completes a maneuver or comes to a stop (e.g., parking, stopping at a red light, or halting after a task).

    Constraints:
    - Only reply with one of the following labels: forward, left turn, left lane change, right turn, right lane change.
    - Use only the given information to form your judgment.  
    - Return your answer in the exact format shown below.
    - You must NOT mention any input source such as:
        - Captions (frame-wise and video-wise captions)
        - Optical flow
        - Lane-change context
        - Sensor or textual data
        - Frame-by-frame or analysis process

        
    Output Format:
    Label: <one of the 7 maneuver labels>  
    Explanation: <Justification>


    ### Example 1

    **Context**  
    "frame_80": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane road road others others lane road lane lane lane lane lane road lane lane road road lane lane lane lane lane lane lane road road road road road road road lane road road road road"
    "frame_100": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane others others others lane road lane lane lane lane lane lane lane lane road road lane lane lane lane lane lane lane lane lane road road road road road lane road road road road"
    "frame_120": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane road road others others lane road lane lane lane lane lane road lane lane road road lane lane lane road road road lane lane lane road road road road road lane road road road road"
    "frame_140": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane others others others road lane lane lane lane lane lane lane lane road road lane road lane lane lane lane road road lane road road road road road lane road road road"

    **Label** — left lane change

    
    ### Example 2

    **Context**  
    "frame_80": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane others others others others others others lane lane lane lane lane lane others others others others road road lane road lane others others others others others others road road road others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others", 
    "frame_100": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane others others others others others lane lane lane lane lane lane lane lane lane others others others road road lane others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others", 
    "frame_120": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane others others others others others others others lane lane lane lane lane lane others others lane lane others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others", 
    "frame_140": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane others others others others others others others road road lane lane lane lane others others others others others others road others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others"
    **Label** — left lane change


    ---

    ### Example 3

    **Context**  
    "frame_80": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane road lane lane others lane lane lane lane road road road lane lane road lane lane lane road road road road road road others others others others others"
    "frame_100": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane others lane lane lane lane road road lane lane road lane lane lane road road road road road road road others others others others others"
    "frame_120": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane others lane lane lane road road road lane lane road lane lane lane road road road road road road road others others others others others"
    "frame_140": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane lane others lane lane lane road road lane road road road lane lane lane road road road road road road road others others others others others"

    **Label** — right lane change

    ---

    ### Example 4

    **Context**  
    "frame_80": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane lane others others lane road road lane lane road lane lane lane lane lane lane others others others others others others others others others others others others"
    "frame_100": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane lane others others lane road road lane road lane lane lane lane lane lane lane others others others others others others others others others others others others"
    "frame_120": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane lane others others lane road lane lane road lane lane lane lane lane lane lane others others others others others others others others others others others others"
    "frame_140": "others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others others lane lane lane lane lane lane lane lane lane others others others road lane lane road road lane road lane lane lane lane lane others others others others others others others others others others others others"

    **Label** — right lane change

    ---

    Here is the multimodal input for the video:

    - **Frame-wise Lane Segmentation Context**  
    {lane_change[video_id]}

    - **Frame-wise Captions**  
    {file_data[filename]}

    - **Video-wide Caption**  
    {input_prompt}

    - **Surrounding context**  
    {narratives[filename]}



    ### Example 1

    **Context**  
    "frame_6": "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "left", "right", "right", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "right", "left", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "right", "left", "left", "left", "right", "right", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "right", "right", "left", "left", "left", "left" 
    "frame_7": "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "right", "left", "left", "left", "left", "left" 
    "frame_8": "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right", "right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "right", "right", "right", "right", "right", "left", "right", "right", "right", "left", "right", "right", "left", "left", "left", "right", "left", "right", "left", "right", "left"
    "frame_9": "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "left", "left", "left", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "right", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "right", "right", "left", "left", "right", "right", "right", "right", "right", "right", "left", "right", "right", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "right", "left"
    "frame_10": "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "right", "left", "left", "left", "right", "right", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "right", "right", "left", "left", "right", "right", "right", "right", "right", "left", "left", "right", "right", "right", "right", "right", "left", "left", "left", "right", "left", "right", "left", "right", "left"

    **Label** — right turn

    ### Example 2

    **Context**  

    "frame_6": "right", "right", "right", "left", "right", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right", "left", "right", "left", "right", "right", "right", "right", "right", "right", "left", "left", "right", "left", "right", "right", "right", "right", "right", "left", "right", "right", "right", "left", "right", "right", "left", "left", "left", "right", "right", "right", "left", "left", "left", "right", "right", "left", "right", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "right", "right", "right", "left", "right", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left"
    "frame_7": "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "left", "left", "left", "right", "right", "right", "right", "left", "left", "right", "left", "left", "left", "right", "right", "left", "left", "right", "right", "right", "right", "left", "left", "left", "right", "right", "left", "left", "right", "left", "left", "right", "right", "right", "right", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left", "right", "left", "left", "right", "left", "right", "right", "left", "left", "left", "left", "left", "right", "left", "right", "left", "left", "right", "left", "right", "right", "right", "right", "right", "right", "left", "right", "right", "right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left"
    "frame_8": "right", "right", "right", "right", "left", "right", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "right", "right", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "right", "right", "left", "left", "right", "left", "left", "right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "right", "right", "left", "right", "right", "right", "right", "left", "left", "left", "right", "right", "right", "left", "left", "left", "right", "right", "left", "right", "right", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "left", "right", "right", "right", "left"
    "frame_9": "right", "right", "right", "right", "left", "right", "right", "right", "left", "left", "right", "right", "left", "left", "left", "right", "right", "left", "left", "left", "right", "left", "right", "left", "left", "right", "left", "left", "left", "right", "right", "left", "right", "left", "right", "right", "left", "right", "right", "left", "right", "left", "right", "right", "right", "right", "right", "left", "right", "right", "left", "right", "right", "left", "right", "left", "right", "right", "right", "right", "right", "left", "left", "right", "right", "right", "left", "right", "left", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left"
    "frame_10": "right", "right", "right", "right", "right", "right", "right", "right", "left", "right", "left", "left", "left", "right", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "right", "left", "right", "left", "right", "right", "right", "right", "left", "left", "right", "left", "left", "right", "right", "right", "right", "left", "right", "left", "right", "right", "right", "left", "right", "left", "left", "right", "right", "right", "right", "left", "left", "right", "left", "left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right"

    **Label** — left turn

    ### Example 3

    **Context**  

    "frame_5": [["right", "left", "left", "right", "left", "left", "left", "left", "right", "right", "left", "left", "left", "right"], ["right", "left", "left", "left", "right", "right", "left", "left", "right", "right", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "left", "left", "left", "right", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "right"], ["left", "left", "right", "right", "left", "right", "right", "right", "right", "right", "right", "left", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right"], ["left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "right", "right", "right"]], 
    "frame_6": [["right", "left", "left", "right", "left", "left", "left", "left", "right", "right", "left", "left", "left", "right"], ["right", "left", "left", "left", "right", "right", "left", "right", "right", "right", "left", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "left", "left", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "left", "left", "right", "right", "right", "left", "left", "right"], ["left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "left", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right", "right"], ["left", "left", "left", "left", "left", "right", "right", "right", "right", "left", "left", "left", "right", "right"]], 
    "frame_7": [["right", "left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "left", "left", "left"], ["right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "left", "left", "right"], ["left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "left", "right", "right"], ["left", "left", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right"], ["left", "left", "right", "right", "right", "right", "right", "left", "right", "left", "left", "left", "right", "right"], ["left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right"], ["left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "right", "left"]], 
    "frame_8": [["right", "left", "right", "right", "left", "right", "left", "right", "right", "right", "right", "left", "left", "right"], ["right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "right", "left", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "left", "right", "right"], ["left", "left", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right"], ["left", "left", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right", "right"], ["left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "right", "right"]], 
    "frame_9": [["right", "left", "right", "right", "left", "left", "left", "right", "right", "right", "right", "right", "left", "right"], ["right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "right", "left", "left", "right", "right", "right", "right", "right", "left", "left"], ["left", "left", "right", "right", "right", "left", "left", "right", "right", "right", "right", "right", "right", "right"], ["left", "left", "right", "right", "right", "right", "left", "right", "right", "right", "left", "left", "right", "right"], ["left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "left", "right"], ["right", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "right", "right"], ["left", "left", "left", "left", "left", "right", "right", "left", "left", "left", "left", "left", "left", "right"]], 
    "frame_10": [["right", "left", "right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "left", "right"], ["right", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "right", "left", "left"], ["left", "left", "left", "left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "left", "left", "right", "left", "left", "right", "right", "right", "right", "left", "left", "left"], ["left", "left", "right", "right", "right", "right", "left", "right", "right", "right", "right", "left", "left", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "right"], ["left", "right", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right"], ["left", "left", "right", "right", "right", "right", "right", "right", "left", "left", "left", "left", "left", "right"], ["left", "left", "left", "left", "left", "right", "right", "right", "left", "left", "left", "left", "left", "right"]]

    **Label** — left turn


    - **Optical Flow  Context**  
    {optical_flow[filename]}


    Output:
    Label:
    Explanation1:
    Explanation2:
    Explanation3:
    Explanation4:
    Explanation5:

    """
        }
    ]

    outputs = pipeline(
    messages_1,
    max_new_tokens=256,
    )

    temp = outputs[0]["generated_text"][-1]['content']
    

    label_prefix = 'Label'
    explanation_prefix = 'Explanation1'

    label_start = temp.find(label_prefix)
    explanation_start = temp.find(explanation_prefix)

    # Extract label
    temp_label = temp[label_start + len(label_prefix):explanation_start].strip()

    # Extract explanation
    explanation = temp[explanation_start :].strip()
    label = ''
    for _, clss in classes.items():
        if clss in temp_label:
            label = clss
            break


    if any(gd['filename']==filename):
        ans = classes[gd[gd['filename']==filename]['class'].to_list()[0]]
        flag = False
        if ans in label:
            count_1[ans] += 1
            flag = True
            captions[filename] = temp
        else:
            print(label, ans)
        
        predicted.append(label)
        groundtruth.append(ans)
        
        temp = outputs[0]["generated_text"][-1]
        d['filename'].append(filename) 
        d['Predcited'].append(temp['content'])
        d['Ground Truth'].append(ans)
        d['Match'].append(flag)
        total_1[ans] += 1
        caption_file.write(f'''ground truth : {ans} , label : {temp}\n''')


df = pd.DataFrame(d)



# Step 2: Save the DataFrame to a CSV file
df.to_csv('output.csv', index=False)



with open('captions.json', 'w') as caption_file_json:
    json.dump(caption_dict, caption_file_json, indent=4)


class_names = ['forward', 'left lane change', 'left turn', 'right lane change', 'right turn']

cm = confusion_matrix(groundtruth, predicted, labels=class_names)

# Plot confusion matrix using seaborn heatmap
plt.figure(figsize=(8,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix as an image file
plt.savefig('confusion_matrix.png')  # Save as PNG
plt.close()  # Close the plot


global_count_1['forward'] = max([count_3['forward'], count_2['forward']])
global_count_1['left lane change'] = max([count_1['left lane change'], count_2['left lane change'], count_3['left lane change']])
global_count_1['right lane change'] = max([count_1['right lane change'], count_2['right lane change'], count_3['right lane change']])
global_count_1['left turn'] = count_2['left turn']
global_count_1['right turn'] = count_2['right turn']

acc = (sum(global_count_1.values()))/(sum(total_3.values()))
print(acc)
from sklearn.metrics import f1_score
f1 = f1_score(groundtruth, predicted, average='weighted')

print(f1)

import pdb; pdb.set_trace()


#################################AIDE###################################################

