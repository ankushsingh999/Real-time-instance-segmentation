# Real-time-instance-segmentation

Real Time Instance Segmentation using Modified YOLACT

Developed a custom loss function that takes into consideration various strategies to address the challenges posed by occlusions, pose changes, and deformation. Specifically, explored incorporating object context or shape information, known as contextual loss, to improve the model's performance. This approach takes advantage of the spatial arrangement and co-occurrence of objects, making it more robust to occlusions and varying poses. Furthermore, improved the model by tuning the hyperparameters in YOLACT. 

Implemented channel attention model for instance segmentation, which leverages channel attention to capture feature dependencies and improve the segmentation in complex scenarios. The aim was to increase the mean average precision (mAP) of the model.

#  Channel Attention Model

The channel attention model used is inspired and referenced from the paper "Squeeze-and-Excitation Networks" by Jie Hu, Li Sh en, and Gang Sun.
The SE block offers a straightforward and lightweight attention mechanism that can be easily implemented into different convolutional neural network (CNN) designs to increase their performance. It is meant to adaptively recalibrate feature maps in a channel-wise way.

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

**Before**

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/ee210f8d-13c8-41fb-9b1e-931651471589)

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/3c9c132a-f997-46f1-ab70-f13d68350b0f)

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/1b525d4a-3ae3-42b4-83d9-e7c72db8cfda)


**After**

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/44df7328-e6c2-4fc2-9fc2-932b4c767225)

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/5a31b221-79c9-4a0b-ad3d-6df2b403f3c9)

![image](https://github.com/ankushsingh999/Real-time-instance-segmentation/assets/64325043/1941e46e-e68c-4c04-8694-d0f8a0a6bbfe)


