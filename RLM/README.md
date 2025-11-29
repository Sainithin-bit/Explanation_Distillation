# HybridNets: End2End Perception Network

We use HybridNets to obtain road and lane (RLM) segmentation context. Below are the code changes mentioned to be added in the HybridNets file for getting RLM context in json files.


## Getting Started [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dpMrjgJs3qKxaKR833RJDyZTh2O1-Wyn?usp=sharing#forceEdit=true&sandboxMode=true)
### Installation
The project was developed with [**Python>=3.7**](https://www.python.org/downloads/) and [**Pytorch>=1.10**](https://pytorch.org/get-started/locally/).
```bash
git clone https://github.com/datvuthanh/HybridNets
cd HybridNets
pip install -r requirements.txt
```
 
### Demo
```bash
# Download end-to-end weights
curl --create-dirs -L -o weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth

# Image inference
python hybridnets_test.py -w weights/hybridnets.pth --source demo/image --output demo_result --imshow False --imwrite True

# Video inference
python hybridnets_test_videos.py -w weights/hybridnets.pth --source demo/video --output demo_result

# Result is saved in a new folder called demo_result
```

## Usage
### Data Preparation
Recommended dataset structure:
```bash
HybridNets
└───datasets
    ├───imgs
    │   ├───train
    │   └───val
    ├───det_annot
    │   ├───train
    │   └───val
    ├───da_seg_annot
    │   ├───train
    │   └───val
    └───ll_seg_annot
        ├───train
        └───val
```
Update your dataset paths in `projects/your_project_name.yml`.

For BDD100K: 
- [imgs](https://bdd-data.berkeley.edu/)
- [det_annot](https://drive.google.com/file/d/1Ge-R8NTxG1eqd4zbryFo-1Uonuh0Nxyl/view)
- [da_seg_annot](https://drive.google.com/file/d/1xy_DhUZRHR8yrZG3OwTQAHhYTnXn7URv/view)
- [ll_seg_annot](https://drive.google.com/file/d/1lDNTPIQj_YLNZVkksKM25CvCHuquJ8AP/view)

### Training
#### 1) Edit or create a new project configuration, using bdd100k.yml as a template. Augmentation params are here.
```python
# mean and std of dataset in RGB order
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# bdd100k anchors
anchors_scales: '[2**0, 2**0.70, 2**1.32]'
anchors_ratios: '[(0.62, 1.58), (1.0, 1.0), (1.58, 0.62)]'

# BDD100K officially supports 10 classes
# obj_list: ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle', 'traffic light', 'traffic sign']
obj_list: ['car']
obj_combine: ['car', 'bus', 'truck', 'train']  # if single class, combine these classes into 1 single class in obj_list
                                               # leave as empty list ([]) to not combine classes

seg_list: ['road',
          'lane']
seg_multilabel: false  # a pixel can belong to multiple labels (i.e. lane line + underlying road)

...
```

