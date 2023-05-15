# Real-time-instance-segmentation

Real Time Instance Segmentation using Modified YOLACT

Developed a custom loss function that takes into consideration various strategies to address the challenges posed by occlusions, pose changes, and deformation. Specifically, explored incorporating object context or shape information, known as contextual loss, to improve the model's performance. This approach takes advantage of the spatial arrangement and co-occurrence of objects, making it more robust to occlusions and varying poses. Furthermore, improved the model by tuning the hyperparameters in YOLACT. 

Implemented channel attention model for instance segmentation, which leverages channel attention to capture feature dependencies and improve the segmentation in complex scenarios. The aim was to increase the mean average precision (mAP) of the model.

