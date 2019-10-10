# Manipulating Attributes of Natural Scenes via Hallucination
<p align="center">
  <b>Levent Karacan, Zeynep Akata, Aykut Erdem, Erkut Erdem</b></span>
</p>

This is the implementation of [ Manipulating Attributes of Natural Scenes via Hallucination ](#) (Accepted for publication in ACM Transactions on Graphics, 2019).
See our [ website ](https://hucvl.github.io/attribute_hallucination/) for more information about the model!

<div align="center">
  <img src="https://hucvl.github.io/attribute_hallucination/images/teaser.jpg" style="float:left" width="100%">
</div>

## Bibtex

```
@ARTICLE{attribute_hallucination,
author={L. {Karacan} and Z. {Akata} and A. {Erdem} and E. {Erdem}},
journal={ACM Transactions on Graphics},
title={Manipulating Attributes of Natural Scenes via Hallucination},
year={2019},
volume={},
number={},
pages={1-1},
month={},}
```

## Requirements
- Linux or OSX
- NVIDIA GPU + CUDA
- We tested our code on python2.7 and python3.6
- You need to install Tkinter module to use interactive scene editing tool.

## Dataset

- We arranged dataset from [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and [Transiet Attributes](http://transattr.cs.brown.edu/) datasets.
- For our work, we curated a new dataset, which we call ALS18K, using images and annotations from [ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) and [Transiet Attributes](http://transattr.cs.brown.edu/) datasets.
- You can the download the ALS18K dataset from this [link](https://drive.google.com/drive/folders/1JWoswk9lIi7T7cvWZFsslDwUx-8m0wAk?usp=sharing)
- Please refer to the project pages of the aforementioned works for more details.
  - Laffont et al., [Transient Attributes for High-Level Understanding and Editing of Outdoor Scenes](http://transattr.cs.brown.edu/), SIGGRAPH 2014
  - Zhou et al., [Scene Parsing through ADE20K Dataset](https://groups.csail.mit.edu/vision/datasets/ADE20K/), CVPR 2017

## Coarse Model Training
- We follow similar multiscale strategy with [Pix2pixHD model](https://github.com/NVIDIA/pix2pixHD) in our code. Some parts of code adapted from Pix2pixHD's official code.
- Download "sceneparsing" models from the [link](https://drive.google.com/drive/folders/1HP34R6aqw7qdWk-477cJGFXW1SyDWXYN?usp=sharing).


```
python train_coarse.py --img_root ./data/ADE20K_TA_Dataset/  --save_filename ./model/sgn_coarse --scene_parsing_model_path ./sceneparsing/resnet34_dilated8/ --batch_size 16  --num_epoch 100
```

## Enhancer Model Training

```
python train_enhancer.py --img_root ./data/ADE20K_TA_Dataset/ --coarse_model ./model/sgn_coarse_G_latest  --save_filename ./model/sgn_hd --scene_parsing_model_path ./sceneparsing/resnet34_dilated8/ --batch_size 8 --num_epoch 100 --isEnhancer
```

## Test Coarse Model
- Download pre-trained models from the [link](https://drive.google.com/open?id=1Tm-iIvTdvRgpF0AmGqZS-kwkoH-JNmVh).

```
python test.py --img_root ./data/ADE20K_TA_Dataset/ --model_path ./model/sgn_coarse_G_latest --save_dir ./results
```

## Test Enhancer Model

```
python test.py --img_root ./data/ADE20K_TA_Dataset/ --model_path ./model/sgn_enhancer_G_latest --save_dir ./resultsHD --isEnhancer
```

## Interactive Scene Editing Demo
- Install Tkinter module.
- Photo style transfer and semantic segmentation codes taken directly from the following works and adapted to editing tool.
 - Zhou et al., [Scene Parsing through ADE20K Dataset](https://github.com/CSAILVision/semantic-segmentation-pytorch), CVPR 2017
 - Li et al., [A Closed-form Solution to Photorealistic Image Stylization](https://github.com/NVIDIA/FastPhotoStyle), ECCV 2018
 - Mechrez et al.,[Photorealistic Style Transfer with Screened Poisson Equation](https://github.com/roimehrez/photorealism),BMVC 2017
 - Yoo et al.,[Photorealistic Style Transfer via Wavelet Transforms](https://github.com/clovaai/WCT2), ICCV 2019
- If you do not want to try the codes to adapt, you can download semantic_segmentation_pytorch, fastphoto, photorealism, WCT2 [link](https://drive.google.com/drive/folders/18E__lGg5V4x792RsI8xYeQbSMhQcE2FN?usp=sharing)
- Place those codes under the editing_tool folder.

```
cd editing_tool
python main.py --model_path ./pretrained_models/sgn_enhancer_G_latest --isEnhancer --image_size 512
```
