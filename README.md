# pytorch-video-recognition

<table>
   <tr>
       <td><img src="assets/demo1.gif" frame=void rules=none></td>
       <td><img src="assets/demo2.gif" frame=void rules=none></td>
   </tr>
</table>

## Introduction
This repo contains several models for video action recognition,
including C3D, R2Plus1D, R3D, inplemented using PyTorch (0.4.0).
Currently, we train these models on UCF101 and HMDB51 datasets.
**More models and datasets will be available soon!**

## Installation
The code was tested with Anaconda and Python 3.5. After installing the Anaconda environment:

0. Clone the repo:
    ```Shell
    git clone https://github.com/jfzhang95/pytorch-video-recognition.git
    cd pytorch-video-recognition
    ```

1. Install dependencies:

    For PyTorch dependency, see [pytorch.org](https://pytorch.org/) for more details.

    For custom dependencies:
    ```Shell
    conda install opencv
    pip install tqdm scikit-learn tensorboardX
    ```

2. Download pretrained model from [BaiduYun](https://pan.baidu.com/s/1saNqGBkzZHwZpG-A5RDLVw).
   Currently only support pretrained model for C3D.

3. Configure your dataset and pretrained model path in
[mypath.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/mypath.py).

4. You can choose different models and datasets in
[train.py](https://github.com/jfzhang95/pytorch-video-recognition/blob/master/train.py).

    To train the model, please do:
    ```Shell
    python train.py
    ```

## Experiments
These models were trained in machine with NVIDIA TITAN X 12gb GPU. Note that I splited
train/val/test data for each dataset using sklearn. If you want to train models using
official train/val/test data, you can look in the repo in the folder dataloaders for a
file called dataset.py, and modify it to your needs.

Currently, I only trained C3D model in UCF and HMDB datasets. The train/val/test
accuracy and loss curves for each experiment are shown below:

- **UCF101**

<p align="center"><img src="assets/ucf101_results.png" align="center" width=900 height=auto/></p>

- **HMDB51**

<p align="center"><img src="assets/hmdb51_results.png" align="center" width=900 height=auto/></p>

Experiments for other models will be updated soon ...