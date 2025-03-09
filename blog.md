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

Faster R-CNN became the baseline for pedestrian detection, with enhancements like Feature Pyramid Networks (FPN) improving small-object detection. Meanwhile, YOLO and SSD revolutionized real-time detection. RetinaNet (Lin et al., 2017) introduced **focal loss** which is?, enabling one-stage detectors to match two-stage accuracy. **Later versions like** later than? YOLOv4, YOLOv5, and YOLOv7 improved detection efficiency, making real-time pedestrian detection feasible. Recently, YOLOv8 further **refines** gpt word the YOLO family by adopting advanced backbone architectures and streamlined training procedures for improved accuracy-speed tradeoffs, demonstrating competitive results on pedestrian benchmarks. **last sentence is very gpt**

### Transformer-Based Models

DETR (Carion et al., 2020) introduced an end-to-end approach for pedestrian detection with transformers, removing the need for anchor boxes and post-processing steps like **NMS** full name. However, slow convergence and difficulty with small objects led to improvements such as Deformable DETR (Zhu et al., 2021), which uses multi-scale attention. Hybrid models like the Swin Transformer enhanced pedestrian detection by integrating hierarchical vision features. Real-Time DETR (RT-DETR) **addresses high computational overheads and latency in transformer-based detectors by employing lightweight attention and efficient decoder modules.** gpt As a result, RT-DETR offers promising real-time performance while retaining transformers' global context modeling advantages.

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

and this is why:
https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.pdf

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

We decided to fine-tune pre-trained YOLOv8 and RT-DETR large models which show competitive predictive performance in pedestrian detection on images from city traffic cameras [link](https://arxiv.org/pdf/2404.08081).

### YOLO

We fine-tuned a `YOLOv8l` model pre-trained on [COCO](https://cocodataset.org/#home) dataset for object detection. The model is vailable in [Ulitralitics PyPI package](https://pypi.org/project/ultralytics/). While newer versions were proposed, the 8th version still remains a strong baseline for pedestrian detection. Morover, it was mentioned in the paper.

We trained the model on 50 epochs with batch size of 16. To speed up the training, the images were resized to fit a 640x640 pixels square with a preserved aspect ratio. The rest of the image was padded with grey pixels.

### RT-DETR

## Results 

In this section, we compare the predictive performance of `YOLOv8l` and `RT-DETR-l` models using standard object detection evaluation metrics. The results below summarize their strengths in terms of precision, recall, and mean average precision (mAP). The best results per metric are highlighted in bold.

| Metric                                      | YOLOv8  | RT-DETR |
|---------------------------------------------|--------|---------|
| Average Precision (AP)                  | 0.4546 | **0.4551**  |
| AP at IoU = 0.50                         | 0.7273 | **0.7628**  |
| Mean Average Precision (mAP)            | 0.4546 | **0.4551**  |
| mAP at IoU = 0.50                        | 0.7273 | **0.7628**  |
| mAP at IoU = 0.75                        | **0.4776** | 0.4679  |
| Mean AP for Different IoU Thresholds     | 0.4546 | **0.4551**  |
| Mean Precision                           | **0.8129** | 0.8033  |
| Precision                                | **0.8129** | 0.8033  |
| Recall                                   | 0.6514 | **0.6706**  |

RT-DETR achieves a slightly higher AP and mAP, particularly at IoU = 0.50, suggesting better overall detection performance. However, YOLO performs slightly better at IoU = 0.75, indicating stronger localization accuracy under stricter overlap conditions. In terms of recall, RT-DETR detects more true positives (0.6706 vs. 0.6514), which can reduce the number of missed detections. On the other hand, YOLO maintains a slightly higher precision (0.8129 vs. 0.8033), meaning it produces fewer false positives compared to RT-DETR.  

Overall, RT-DETR demonstrates better recall and consistency across different IoU thresholds, while YOLO maintains strong precision and localization accuracy at higher thresholds. The choice between these models depends on the specific task requirements—whether prioritizing detection coverage or minimizing false positives is more important.