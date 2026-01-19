## Training Pipeline

- After generating **explanations and labels** using the **DriveXplain Explanation Generation** module (with all required inputs), use the generated outputs to train the models provided in this repository.
- The training process leverages video-level descriptions and explanations to learn driving-relevant reasoning and prediction tasks.
- Once training is complete, the models can be evaluated to obtain the final results.

## Dependencies and Reference Models

This work builds upon the following open-source vision–language models:

- **VideoLLaMA**  
  https://github.com/DAMO-NLP-SG/Video-LLaMA

- **Qwen2.5**  
  https://github.com/QwenLM/Qwen2.5

Please refer to the respective repositories for installation instructions, pretrained checkpoints, and usage details.
