import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor

model_id = "DAMO-NLP-SG/VideoLLaMA3-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)


processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# define a chat history and use `apply_chat_template` to get correctly formatted prompt
# Each value in "content" has to be a list of dicts with types ("text", "image", "video") 
conversation = [
    {

        "role": "user",
        "content": [
            {"type": "text", "text": '''  
    
    Basic Instruction: Classify the driving maneuver based on the video.

    Task and Label Descriptions:
    - 'right turn': The vehicle turns sharply or significantly to the right.
    - 'right lane change': The vehicle moves into the right lane.
    - 'left turn': The vehicle turns sharply or significantly to the left.
    - 'left lane change': The vehicle moves into the left lane.
    - 'forward': The vehicle completes a maneuver or comes to a stop (e.g., parking, stopping at a red light, or halting after a task).

    Constraints: 
    - Only reply with one of the following labels: right turn, right lane change, left turn, left lane change, forward
'''},
            {"type": "video"},
            ],
    },
]

import json
import pandas as pd
import os

##Open the file
captions = dict()
file = open('Descriptions.json', 'w')

#Replace the path of the folder which contains videos (Brain4Cars - /scratch/sai/brain4cars_data/road_cam)
for video_id in os.listdir('/scratch/sai/brain4cars_data/road_cam'):

    if '.avi' not in video_id:
        continue
        
    if video_id in gd:
        print(video_id)
    else:
        continue

    # if any(gd['filename']==video_id):
    #     print(video_id)
    #     flag = False
    # else:
    #     continue


    video_path = f'/scratch/sai/brain4cars_data/road_cam/{video_id}'
    container = av.open(video_path)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # sample uniformly 8 frames from the video, can sample more for longer videos
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    clip = read_video_pyav(container, indices)
    inputs_video = processor(text=prompt, videos=clip, padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    temp = processor.decode(output[0][2:], skip_special_tokens=True)
    temp = temp.split('ASSISTANT:')[1]
    print(temp.lower())
    captions[video_id] = temp.lower()

json.dump(captions, file, indent=4)


