# HybridNets: End2End Perception Network

We Use 


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

#### 2) Train
```bash
python train.py -p bdd100k        # your_project_name
                -c 3              # coefficient of effnet backbone, result from paper is 3
            OR  -bb repvgg_b0     # change your backbone with timm
                -n 4              # num_workers
                -b 8              # batch_size per gpu
                -w path/to/weight # use 'last' to resume training from previous session
                --freeze_det      # freeze detection head, others: --freeze_backbone, --freeze_seg
                --lr 1e-5         # learning rate
                --optim adamw     # adamw | sgd
                --num_epochs 200
```
Please check `python train.py --help` for cheat codes.

**~~IMPORTANT~~ (deprecated):** If you want to train on multiple gpus, use `train_ddp.py`. Tested on NVIDIA DGX with 8xA100 40GB.  
Why didn't we combine DDP into the already existing `train.py` script?
1. Lots of if-else.
2. Don't want to break functioning stuffs.
3. Lazy. 

**Update 24/06/2022:** `train_ddp.py` broke because we have a lot of things changed. Therefore, we decided to write a merged `train.py` with DDP support for easier maintainance. In the meantime, please clone [this commit](https://github.com/datvuthanh/HybridNets/tree/ecc835ca1f68b17c9d1deb926f9e7bbe8455ccee) with a working `train_ddp.py` script if you really have to.

#### 3) Evaluate
```bash
python val.py -w checkpoints/weight.pth
```
Again, check `python val.py --help` for god mode.

**Validation process got killed! What do I do?**
=> This is because we use a default confidence threshold of 0.001 to compare with other networks. So when calculating metrics, it has to handle a large amount of bounding boxes, leading to out-of-memory, and finally exploding the program before the next epoch.

That being said, there are multiple ways to circumvent this problem, choose the best that suit you:

- Train on a high-RAM instance (RAM as in main memory, not VRAM in GPU). For your reference, we can only val the combined `car` class with 64GB RAM.
- Train with `python train.py --cal_map False` to not calculate metrics when validating. This option will only print validation losses. When the losses seem to flatten and the weather is nice, rent a high-RAM instance to validate the best weight with `python val.py -w checkpoints/xxx_best.pth`. We actually did this to save on cost.
- Reduce the confidence threshold with `python train.py --conf_thres 0.5` or `python val.py --conf_thres 0.5`, depending on your application and end goals. You don't have to get best recall unless you're either helping us by experimenting :smiling_face_with_three_hearts: or competing with us :angry:.

#### 4) Export
```bash
python export.py -w checkpoints/weight.pth --width 640 --height 384
```
This automatically creates an ONNX weight and an `anchor_{H}_{W}.npy` file to use in postprocessing. Refer to the ROS section for usage example.

For more information on this refer to HybridNets Repo.
