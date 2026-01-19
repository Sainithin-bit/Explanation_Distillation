# Video Description Generation

## Overview
This module generates natural language descriptions for input videos. The generated descriptions are designed to be directly used as inputs for **DriveXplain Generation**.

## Usage
- Provide the input videos in the required format and directory structure.
- Run the script to process the videos.
```bash
python3 videollama.py
```
- The script automatically generates descriptive captions for each video.
- The generated descriptions are saved and can be directly fed into the **DriveXplain Generation** pipeline.

## Output
- Textual descriptions corresponding to each input video.
- Outputs are compatible with downstream DriveXplain-based reasoning and generation tasks.
