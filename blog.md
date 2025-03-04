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

TODO 

<!-- 
### Related research that we build upon:

how it's done now: what current typical approach(es)
Yolo v8 finetune becase: 
https://arxiv.org/pdf/2404.08081 

Fine tune DETR:
https://arxiv.org/abs/2005.12872

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


## Training 

### YOLO 

TODO 

### DETR 

