# Manipulating Attributes of Natural Scenes via Hallucination
<p align="center">
  <b>Levent Karacan, Zeynep Akata, Aykut Erdem, Erkut Erdem</b></span>
</p>

This is the implementation of [ Manipulating Attributes of Natural Scenes via Hallucination ](#) (Accepted for publication in ACM Transactions on Graphics, 2019)
See our [ website ](https://hucvl.github.io/attribute_hallucination/)  for more information about the model!

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
- We tested our code on python2.7 and python3.6
- See [`requirements.txt`](requirements.txt) for the required python packages and run `pip install -r requirements.txt` to install them.

## Coarse Model Training

python train_coarse.py --img_root ./data/ADE20K_TA_Dataset/  --save_filename ./model/sgn_coarse --scene_parsing_model_path ./sceneparsing/resnet34_dilated8/ --batch_size 16  --num_epoch 100

## Enhancer Model Training

python train_enhancer.py --img_root ./data/ADE20K_TA_Dataset/ --coarse_model ./model/sgn_coarse_G_latest  --save_filename ./model/sgn_hd --scene_parsing_model_path ./sceneparsing/resnet34_dilated8/ --batch_size 8 --num_epoch 100 --isEnhancer


## Test Coarse Model

python test.py --img_root ./data/ADE20K_TA_Dataset/ --model_path ./model/SGN/sgn_coarse_G_latest --save_dir ./results

## Test Enhancer Model

python test.py --img_root ./data/ADE20K_TA_Dataset/ --model_path ./model/SGN/sgn_enhancer_G_latest --save_dir ./resultsHD --isEnhancer


## Interactive Scene Editing Demo
Download semantic_segmentation_pytorch, fastphoto, photorealism, WCT2
cd editing_tool
python main.py --model_path ./pretrained_models/sgn_enhancer_G_latest --isEnhancer --image_size 512
