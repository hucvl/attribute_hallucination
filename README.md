# attribute_hallucination



#Coarse Model Training

python3 train_coarse.py --img_root ./data/ADE20K_TA_Dataset/  --save_filename ./model/sgn_coarse --scene_parsing_model_path ./sceneparsing/resnet34_dilated8/ --batch_size 16  --num_epoch 100

#Enhancer Model Training

python3 train_enhancer.py --img_root ./data/ADE20K_TA_Dataset/ --coarse_model ./model/sgn_coarse_G_latest  --save_filename ./model/sgn_hd --scene_parsing_model_path ./sceneparsing/resnet34_dilated8/ --batch_size 8 --num_epoch 100 --isEnhancer


#Test Coarse Model

python36 test.py --img_root ./data/ADE20K_TA_Dataset/ --model_path ./model/SGN/sgn_coarse_G_latest --save_dir ./results

#Test Enhancer Model

python36 test.py --img_root ./data/ADE20K_TA_Dataset/ --model_path ./model/SGN/sgn_enhancer_G_latest --save_dir ./resultsHD --isEnhancer


#Interactive Scene Editing Demo
cd editing_tool
python main.py --model_path ./pretrained_models/sgn_enhancer_G_latest --isEnhancer --image_size 512
