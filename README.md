# Real Time Instance Segmentation using Modified YOLACT

Developed a custom loss function that takes into consideration various strategies to address the challenges posed by occlusions, pose changes, and deformation. Specifically, explored incorporating object context or shape information, known as contextual loss, to improve the model's performance. This approach takes advantage of the spatial arrangement and co-occurrence of objects, making it more robust to occlusions and varying poses. Furthermore, improved the model by tuning the hyperparameters in YOLACT. 

Implemented channel attention model for instance segmentation, which leverages channel attention to capture feature dependencies and improve the segmentation in complex scenarios. The aim was to increase the mean average precision (mAP) of the model.

#  Channel Attention Model

The channel attention model used is inspired and referenced from the paper "Squeeze-and-Excitation Networks" by Jie Hu, Li Sh en, and Gang Sun.
The SE block offers a straightforward and light weight attention mechanism that can be easily implemented into different convolutional neural network (CNN) designs to increase their performance. It is meant to adaptively recalibrate feature maps in a channel-wise way.

There are four primary stages in the YOLACT architecture, which uses a modified ResNet as its backbone, and they are layer 1, layer 2, layer 3, and layer 4. The phases of the ResNet architecture are made to gradually extract higher-level features from the input image. The decision to add four channel attention tiers was primarily made to match the ResNet backbone's stage count.

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/19ccc810-52bc-44d0-bf9a-b4026b169da2)

The following steps were followed while creating this model:

● Feature maps were taken as input and Global average and max pooling were applied on the feature maps.

● The output of GAP and GMP were passed through separate fully connected layers followed by ReLU activation. This is then passed through another FC layer to restore the dimension.

● The results are added from GAP and GMP and a sigmoid activation is applied to obtain channel wise attention weights and is multiplied with the feature maps to obtain better output feature maps.

# Contextual Loss

To design a contextual loss function for YOLACT we leveraged the ideas from the paper "Unsupervised Feature Learning via Nonparametric Instance-level Discrimination”, we have modified the mask prediction head to produce feature embeddings instead of masks. These embeddings were learned using the non-parametric instance-level discrimination approach described in the paper.

We computed the contextual loss between the feature embeddings of adjacent pixels to encourage the embeddings to encode contextual information. The contextual loss is defined as the cosine distance between the feature embeddings of adjacent pixels in the same instance. This loss encourages the network to learn embeddings that capture the spatial relationships between pixels in an instance.

This technique determines the Contextual Loss between two input feature maps, feature 1 and feature 2. The actions are:
a. L2 normalization should be used to normalize the input feature maps.

b. The distance matrix between the normalized feature maps should be calculated.

c. Fix the values of the distance matrix between -1 and 1.

d. After applying a Gaussian function to the distance matrix, calculate the kernel matrix.

e. Based on the kernel matrix, determine the contextual loss.

f. The mean contextual loss should be returned.

This code calculates the contextual loss between two feature maps (feature_map1 and feature_map2) and adds it to the total loss.

# Results

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/1ce6c4c2-0f59-4b2f-8ec9-c2aca877b3ce)

![DL Results](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/dc7a5851-b3c7-4387-865e-7ea96988b1ff)


***Achieved a higher mAP compared to the traditional YOLACT, but obtained a lower fps***

The Contexual loss function helped our model train better, increasing the mAP. While the channel attention model led to better segmentation reults in the images and the model could predict with more confidence. The before and after results are as shown below: 

# BEFORE

![before1 CV](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/167c05be-7264-469d-98a6-84b538c74b4c)

![before2 CV](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/dc5f010b-f35d-447e-9acb-b8b5485d9481)

![before3 CV](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/5e6da129-d4ea-451c-b1a0-49ab9e517baf)


# AFTER

![WPI CV](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/55999f35-e92c-45e1-afdc-54fb855f2e86)

![London CV](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/93562c25-d9c0-4ee2-9deb-a5c7920c1bdd)

![Night CV](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/beda0698-3add-4bd4-b5e8-7551e3713e1d)

# Real Time Instance Segmentation on Videos:

***Achieved 18.1 fps on videos***

***We acknowledge that this video has been sourced from the public domain and has been used by us to test our instance segmentation network. We do not claim any ownership or credit for its content.***

![ezgif com-video-to-gif](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/1bef30a0-8df2-401e-ab22-1303e70ea9dc)

# Installation
 - Clone this repository and enter it:
   ```Shell
   git clone https://github.com/ankushsingh999/Real-time-instance-segmentation.git
   cd yolact
   ```
 - Set up the environment using one of the following methods:
   - Using [Anaconda](https://www.anaconda.com/distribution/)
     - Run `conda env create -f environment.yml`
   - Manually with pip
     - Set up a Python3 environment (e.g., using virtenv).
     - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
     - Install some other packages:
       ```Shell
       # Cython needs to be installed before pycocotools
       pip install cython
       pip install opencv-python pillow pycocotools matplotlib 
       ```
 - If you'd like to train YOLACT, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into `./data/coco`.
   ```Shell
   sh data/scripts/COCO.sh
   ```
 - If you'd like to evaluate YOLACT on `test-dev`, download `test-dev` with this script.
   ```Shell
   sh data/scripts/COCO_test.sh
   ```
 - If you want to use YOLACT++, compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)).
   Make sure you have the latest CUDA toolkit installed from [NVidia's Website](https://developer.nvidia.com/cuda-toolkit).
   ```Shell
   cd external/DCNv2
   python setup.py build develop
   ```
# Evaluation


| Image Size | Backbone      | FPS  | mAP  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Resnet101-FPN | 18.1 | 28.69 | [yolact_base_54_800000.pth](https://drive.google.com/file/d/1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_/view?usp=sharing)      | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EYRWxBEoKU9DiblrWx2M89MBGFkVVB_drlRd_v5sdT3Hgg)


## Quantitative Results on COCO
```Shell
# Quantitatively evaluate a trained model on the entire validation set. Make sure you have COCO downloaded as above.
# This should get 29.92 validation mask mAP last time I checked.
python eval.py --trained_model=weights/yolact_base_54_800000.pth

# Output a COCOEval json to submit to the website or to use the run_coco_eval.py script.
# This command will create './results/bbox_detections.json' and './results/mask_detections.json' for detection and instance segmentation respectively.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json

# You can run COCOEval on the files created in the previous command. The performance should match my implementation in eval.py.
python run_coco_eval.py

# To output a coco json file for test-dev, make sure you have test-dev downloaded from above and go
python eval.py --trained_model=weights/yolact_base_54_800000.pth --output_coco_json --dataset=coco2017_testdev_dataset
```
## Qualitative Results on COCO
```Shell
# Display qualitative results on COCO. From here on I'll use a confidence threshold of 0.15.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --display
```
## Benchmarking on COCO
```Shell
# Run just the raw model on the first 1k images of the validation set
python eval.py --trained_model=weights/yolact_base_54_800000.pth --benchmark --max_images=1000
```
## Images
```Shell
# Display qualitative results on the specified image.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png

# Process an image and save it to another file.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --image=input_image.png:output_image.png

# Process a whole folder of images.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --images=path/to/input/folder:path/to/output/folder
```
## Video
```Shell
# Display a video in real-time. "--video_multiframe" will process that many frames at once for improved performance.
# If you want, use "--display_fps" to draw the FPS directly on the frame.
python eval.py --trained_model=weights/yolact_base_54_800000.pth --score_threshold=0.15 --top_k=15 --video_multiframe=4 --video=my_video.mp4

# Training
By default, we train on COCO. Make sure to download the entire dataset using the commands above.
 - To train, grab an imagenet-pretrained model and put it in `./weights`.
   - For Resnet101, download `resnet101_reducedfc.pth` from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   
    - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an `*_interrupt.pth` file at the current iteration.
   - All weights are saved in the `./weights` directory by default with the file name `<config>_<epoch>_<iter>.pth`.
```Shell
# Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

# Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

# Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

# Use the help option to see a description of all available command line arguments
python train.py --help
```
# Citation

```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```
