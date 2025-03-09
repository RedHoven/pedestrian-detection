# Pedestrian Detection

<!-- ![alt text](Isolated.png "Title") -->

<!-- why interesting or why people should care. -->

## Why Pedestrian Detection is Important and Why You Should Care

Pedestrian detection is a critical task in computer vision with significant real-world implications. It is a fundamental component of autonomous driving, traffic monitoring, and smart surveillance systems. As urban areas become more congested and the number of vehicles on the road increases, ensuring pedestrian safety through intelligent vision systems is more important than ever. Pedestrian detection algorithms are essential for preventing accidents, improving traffic flow, and enhancing the safety of vulnerable road users.

However, building a robust pedestrian detection system comes with its challenges. Variations in lighting, weather conditions, occlusions, and adversarial attacks pose significant challenges. Understanding and addressing them effectively is needed to developing reliable traffic systems.

## Our Approach:

In our research we focus on three main areas:

### 1. Transfer Learning

Training deep learning models from scratch is computationally expensive and requires vast amounts of labeled data. Instead, we utilize transfer learning, where a pre-trained model (e.g., a convolutional neural network trained on ImageNet) is fine-tuned for pedestrian detection. This approach significantly reduces training time and improves performance, especially when labeled pedestrian data is limited. We also test the limits in how much data and computational power is needed to change the detection task to new classes.

### 2. Robustness

Pedestrian detection models can be sensitive to variations in image data, including changes in lighting, weather conditions, image quality, and differences in human appearance. We evaluate how these models perform across diverse settings and explore strategies to increase their robustness.

### 3. Explainability

A common criticism of deep learning models is their "black-box" nature. To address this, we analyze and visualize the layers of our pedestrian detection model to understand how it makes decisions.

### Research questions:

1. How can pedestrian detection models be made more robust to variations in lighting, weather, and the environment?

2. How can knowledge from pedestrian detection in road scenes be effectively transferred to other settings and object categories, such as road signs?

3. How can layers visualization techniques improve the interpretability and explainability of convolutional networks for pedestrian detection models?

<!-- 346 WORDS -->

## Literature Review

Pedestrian detection has evolved significantly with deep learning, transitioning from traditional methods **(HOG+SVM, DPM)** do not use short version for the first time to CNN-based models. Two-stage detectors like Faster R-CNN (Ren et al., 2015) introduced Region Proposal Networks (RPNs) **cite?,** significantly improving accuracy. However, their computational cost led to the rise of one-stage detectors such as **YOLO** full name (Redmon et al., 2016) and **SSD** again full name , which prioritize speed while maintaining accuracy.

### CNN-Based Detection Models

Faster R-CNN established a strong foundation for pedestrian detection, with innovations like Feature Pyramid Networks (FPN) enhancing its ability to detect small objects. RetinaNet introduced focal loss, a technique that addresses class imbalance by down-weighting easy, correctly classified examples, thereby focusing training on harder, misclassified ones. This advancement enabled one-stage detectors to achieve accuracy levels comparable to two-stage models. The YOLO family models further improved detection efficiency, making real-time pedestrian detection more practical. This is made possible by introducing anchor boxes for better handling of various object sizes and shapes, and by adopting advanced backbone networks like Darknet-53 to improve feature extraction and detection accuracy.

### Transformer-Based Models

DETR (Detection Transformer) introduced by Carion et al. (2020) presented an end-to-end object detection framework utilizing transformers, eliminating the need for anchor boxes and post-processing steps like Non-Maximum Suppression (NMS). However, challenges such as slow convergence and difficulties in detecting small objects led to the development of Deformable DETR by Zhu et al. (2021), which incorporated multi-scale attention mechanisms to address these issues. Subsequent hybrid models, including the Swin Transformer, enhanced pedestrian detection by integrating hierarchical vision features. To tackle high computational overheads and latency in transformer-based detectors, Real-Time Detection Transformer (RT-DETR) employs a hybrid encoder that efficiently processes multi-scale features and utilizes IoU-aware query selection to improve object query initialization. As a result, RT-DETR offers promising real-time performance while retaining the global context modeling advantages of transformers.

## Datasets and State-of-the-Art Models

A range of pedestrian detection datasets capture diverse conditions, from urban traffic to low-light or thermal imaging. The table below highlights key benchmarks and their current best-performing models:

- **Caltech** → _LSFM_  
  A pioneering large-scale dataset from urban driving scenarios, commonly used as a standard benchmark.

- **CityPersons** → _DIW Loss_  
  Derived from Cityscapes, emphasizing dense crowds and significant occlusions.

- **LLVIP** → _MMPedestron_  
  Focuses on low-light/nighttime scenes, necessitating specialized approaches.

- **DVTOD** → _YOLOv6 (Thermal)_  
  Infrared/thermal dataset showing how YOLO variants adapt to non-RGB domains.

- **TJU-Ped-traffic** → _LSFM_  
  Heavy-traffic settings with frequent occlusions, demanding robust detection.

- **TJU-Ped-campus** → _EGCL_  
  Campus-based scenarios testing generalization to semi-controlled environments.

- **CVC14** → _CFT_  
  Smaller, varied dataset evaluating adaptability across different conditions.

- **MMPD-Dataset** → _MMPedestron_  
  Emphasizes robust, specialized pedestrian detection across multiple challenges.

These datasets and top-performing models illustrate the breadth of current pedestrian detection challenges and innovative solutions in the field.

<!--
### Related research that we build upon:

how it's done now: what current typical approach(es)
Yolo v8 finetune becase:
https://arxiv.org/pdf/2404.08081

Fine tune DETR:
https://arxiv.org/abs/2005.12872
(here they user RL-DETR-L, we are going to use rtdetr_r50vd)

PED is pedestrian DETR:
https://arxiv.org/pdf/2012.06785

Add the prompt (trainable vector):
https://arxiv.org/abs/2203.12119


### For literature review:
MMPedestron:
https://arxiv.org/pdf/2407.10125v1
this uses IR pictures which are not always available - not good

https://arxiv.org/abs/2404.19299
The method enhances pedestrian detection, especially in challenging scenarios like small-scale or heavily occluded pedestrians.

https://arxiv.org/pdf/2304.03135
The paper proposes constructing a versatile pedestrian knowledge bank by extracting generalized pedestrian features from large-scale pretrained models

- https://github.com/hasanirtiza/Pedestron Pedestron
- https://arxiv.org/pdf/1703.06870 MaskRCNN
- https://arxiv.org/abs/1802.02611 DeepLabV3+
- https://arxiv.org/abs/1908.07919 HRNet
- https://arxiv.org/pdf/1506.02640 Yolo - train from scratch based on the paper
- - https://arxiv.org/pdf/2004.10934v1 Yolov4
- - https://yolov8.com/ Yolov8
- - https://arxiv.org/html/2502.12524v1 Yolov12
- https://arxiv.org/pdf/2211.07636v2 EVA
- https://arxiv.org/abs/1912.06218 YoLACAST++
- https://arxiv.org/abs/2001.00309 BlendMask

what is missing; what the problem is, and what consequences this problem has

explainability - layers vizualization
computationally heavy - trasfer learning trying to make it quick
advesial attacs - trying to make it robust

Other:
Integrate promts to existing architecture

what you propose (e.g. explanation of what you're gonna implement but in words)

# Plan
1. fine-tune YOLOv8 on our dataset
2. Answer research questions
3. Attempt a bigger model -->

## Dataset

We selected the validation part of the [EuroCity Persons (ECP)](https://eurocity-dataset.tudelft.nl/) benchmark as the data for fine-tuning and testing our models. ECP features street-level city images from a driver's point of view. The entire dataset contains images from all seasons and from 12 cities, with various weather conditions and from different parts of a day. The dataset includes both crowds and heavy occlusions. We decided to use the validation part because of its reasonable size for our experiments: 10 GB and 4266 image-label pairs. We randomly split the images and corresponding annotations from all the cities into train, validation, and test sets in 70-10-20 ratio.

## Training

We decided to fine-tune a pre-trained a YOLO model which show competitive predictive performance in pedestrian detection on images from city traffic cameras [link](https://arxiv.org/pdf/2404.08081). Additionally, we chose [what?] because it' s blablabla.

### YOLO

We used `YOLOv8l` model pre-trained on [COCO](https://cocodataset.org/#home) dataset for object detection. The model is vailable in [Ulitralitics PyPI package](https://pypi.org/project/ultralytics/). While newer versions were proposed, the 8th version still remains a strong baseline for pedestrian detection.

We trained the model on 50 epochs with batch size of 16. To speed up the training, the images were resized to fit a 640x640 pixels square with a preserved aspect ratio. The rest of the image was padded with grey pixels.

### Other Model? Why?
