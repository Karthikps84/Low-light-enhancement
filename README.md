
# Towards Light-Agnostic Real-Time Visual Perception

This repository provides a information about installation and running the configs to train and reproduce the results.
## Table of Contents
- [Installation](#installation)
- [Configuration Changes Before Training](#configuration-changes-before-training)
- [Datasets](#datasets)
- [Train Model](#train-model)
- [Results](#results)
## Installation
```
sh install.sh
```
## Configuration Changes Before Training
Before starting the training process, please make the following changes to the configuration:
### Step 1: Set the Data Path
Set the data path to your data directory
### Step 2: Change the Work Directory
Change the work directory

## Datasets

Datasets can be donwloaded from the following paths:

|Dataset | Link |
|:-------:|:----------:|
| Exdark | [Link](https://drive.google.com/file/d/1X_zB_OSp_thhk9o26y1ZZ-F85UeS0OAC/view) |
| COCO | [Link](https://cocodataset.org/#download)|
| LIS | [Link](https://drive.google.com/drive/folders/1KpC82G_H1CI35lmnB2LYr9aK3FQcahAC) |
| Darkface | [Link](https://drive.google.com/file/d/1DuwSRvsYzDpOHdRYG5bk7E45IMDdp5pQ/view) |
| Widerface |[Link](http://shuoyang1213.me/WIDERFACE/) |

## Train Model
After setting the datapath, train the model with the below command.
```
python tools/train.py <CONFIG_FILE>
```

## Evaluate Performance

Run the Test Script: Use the tools/test.py script to evaluate the model. For single-GPU testing, execute:
```
python tools/test.py <CONFIG_FILE> <CHECKPOINT_FILE> 

```

## Results:

### Object Detection
| Settings | Model |Dataset | Backbone| mAP50  | Config | Checkpoint         |
|:---------------:|:---------------:|:---------------:|:-----:|:-----:|:------:|:-----------------------:|
|Low-light| YOLOV3 | Exdark | Darknet | 80.1 | [config](configs/yolo/yolov3_d53_exdark.py) | [Model](https://drive.google.com/file/d/1zxGNWBZE2DNZWMaCgokTZa14oI94n8qh/view?usp=sharing) |
|Light-Agnostic| YOLOV3 | Exdark-COCO | Darknet | 64 |  [config](configs/yolo/yolov3_d53_exdark_coco.py) | [Model](https://drive.google.com/file/d/193sfLPALWBm_-TDdAfhLUy3HmhvyXHzG/view?usp=sharing) |

### Face Detection
| Settings | Model |Dataset | Backbone| mAP50  | Config | Checkpoint         |
|:---------------:|:---------------:|:---------------:|:-----:|:-----:|:------:|:-----------------------:|
|Low-light| RetinaNet | Darkface | ResNet-50 | 53.7 | [config](configs/retinanet/darkface_retinanet.py)  | [Model](https://drive.google.com/file/d/13Nob4E4Q5yh3h8FPJOd-0ULWUE_eqJGj/view?usp=sharing) |
|Low-light| RetinaNet | Darkface_widerface | ResNet-50 | 59 | [config](configs/yolo/yolov3_d53_darkface_widerface.py)  | [Model](https://drive.google.com/file/d/1JX0yi5gCOVbdTVZbR1_lm8igSRPDElXN/view?usp=sharing) |

### Instance Segmentation
| Settings | Model |Dataset | Backbone| mAP | SegAP | Config | Checkpoint         |
|:---------------:|:---------------:|:---------------:|:-----:|:-----:|:-----:|:------:|:-----------------------:|
|Low-light| RTMDet-tiny | LIS |  CSPNext | 61.1 | 53.6 | [config](configs/rtmdet/lis-rtmdet-ins_tiny.py) | [Model](https://drive.google.com/file/d/14rZTzMV8Wb5JuiroH2xHCrqK6nayD0B6/view?usp=sharing) |
|Low-light| RTMDet-tiny | LIS-COCO |  CSPNext | 45.2 | 36.6 | [config](configs/rtmdet/coco_lis-rtmdet-ins_tiny_torch.py) | [Model](https://drive.google.com/file/d/1zFdRhumLbPkWn_LWBvDAP_gKQBpf_ZGz/view?usp=sharing) |


