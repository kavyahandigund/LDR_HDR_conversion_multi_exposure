FHDR: Generating HDR image From differnt multiexposure LDR image Using FHDR model
========================================
[![arXiv](https://img.shields.io/badge/cs.cv-arXiv%3A1912.11463-42ba94.svg)](https://arxiv.org/abs/1912.11463v1)

This repository contains the code for our FHDR work accepted at [GlobalSIP](http://2019.ieeeglobalsip.org).

<p >
    <p>Multi-Exposure Image Combined into
 Single LDR Using Simple Averaging.
</p>
<img src="https://github.com/kavyahandigund/LDR_HDR_conversion_multi_exposure/blob/main/mp2Multi1.JPG" width="70%" left="-20%">
<img src="https://github.com/kavyahandigund/LDR_HDR_conversion_multi_exposure/blob/main/mp2multi2.JPG" width="70%" left="-20%">
</p>
</p>

    
Table of contents:
-----------

- [Abstract](#abstract)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Pretrained models](#pretrained-models)
- [Evaluation](#evaluation)


Abstract
------------

> High dynamic range (HDR) image generation from a single exposure low dynamic range (LDR) image has been made possible due to the recent advances in Deep Learning. Various feed-forward Convolutional Neural Networks (CNNs) have been proposed for learning LDR to HDR representations. <br><br>
To better utilize the power of CNNs, we exploit the idea of feedback, where the initial low level features are guided by the high level features using a hidden state of a Recurrent Neural Network. Unlike a single forward pass in a conventional feed-forward network, the reconstruction from LDR to HDR in a feedback network is learned over multiple iterations. This enables us to create a coarse-to-fine representation, leading to an improved reconstruction at every iteration. Various advantages over standard feed-forward networks include early reconstruction ability and better reconstruction quality with fewer network parameters. We design a dense feedback block and propose an end-to-end feedback network- FHDR for HDR image generation from a single exposure LDR image. Qualitative and quantitative evaluations show the superiority of our approach over the state of-the-art methods.

Setup
-----

### Pre-requisites

- Python3
- GPU, CUDA
- [OpenCV](https://opencv.org)
- [PIL](https://pypi.org/project/Pillow/)
- [Numpy](https://numpy.org/)
- [scikit-image](https://scikit-image.org/)
- [tqdm](https://pypi.org/project/tqdm/)

**`requirements.txt`** has been provided for installing Python dependencies.

```sh
pip install -r requirements.txt
```

Dataset
--------

The dataset is to comprise of LDR (input) and HDR (ground truth) image pairs. The network is trained to learn the mapping from LDR images to their corresponding HDR ground truth counterparts.

The dataset should follow the following folder structure - 

```
> dataset

    > train

        > LDR

            >LDR_exposure_+2
              >LDR_001.jpg
              >LDR_002.jpg
            >LDR_exposure_-2
              >LDR_001.jpg
              >LDR_002.jpg
            >LDR_exposure_0
              >LDR_001.jpg
              >LDR_002.jpg

        > HDR

            > hdr_image_1.hdr
            > hdr_image_2.hdr
            .
            .

    > test

```

- Sample test datasets can be downloaded here - 
    - [256x256 size images]

- For evaluating on this dataset, download and unzip the folder, replace it with the `test` directory in the `dataset` folder, and refer to [Pretrained models](#pretrained-models) and [Evaluation](#evaluation).

**Note:** The pre-trained models were trained on 256x256 size images.

Training
--------

After the dataset has been prepared, the model can be trained using the **`train.py`** file.

```sh
python3 train.py
```

The corresponding parameters/options for training have been specified in the **`options.py`** file and can be easily altered. They can be logged using -

```sh
python3 train.py --help
```
- **`--iter`** param is used to specify the number of feedback iterations for global and local feedback mechanisms (refer to paper/architecture diagram)
- Checkpoints of the model are saved in the **`checkpoints`** directory. (Saved after every 2 epochs by default)
- GPU is used for training. Specify GPU IDs using **`--gpu_ids`** param.
- The iter-1 model takes around 2.5 days to train on a dataset of 12k images on an RTX 2070 SUPER GPU.

Pretrained models
---------------------------

Pre-trained models can be downloaded from the below-mentioned links. 

These models have been trained with the default options, on 256x256 size images for 200 epochs, in accordance with the paper.

- [3-Iterations model](https://drive.google.com/drive/folders/1aJkGCpSN2T96vfQoh2OFwIMZVVzN3C6F)
- [2-Iterations model](https://drive.google.com/drive/folders/1j6QkshoLHfovfva9YbJuVUjFk14wT_OC)
- [1-Iterations model](https://drive.google.com/drive/folders/1chMTUfzu6946K4KfitTGXTPOvlWSy7bw) 

Here is a graph plotting the performance vs iteration count. 

<img src="https://github.com/kavyahandigund/LDR_HDR_conversion_multi_exposure/blob/main/mp2multi3.JPG" width="40%">

Evaluation
----------

The performance of the network can be evaluated using the **`test.py`** file - 
SSIM and PSNR are used as a evaluation matrics.



This research was supported by the Science and Engineering Research Board (SERB) Core Research Grant.

