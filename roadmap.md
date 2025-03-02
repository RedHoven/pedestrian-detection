# Pedestrian - the data in the traffic startup
-Comparing finetuning training 
-Adversarial attacks 
-People detecting race bias  
-Transfer learning 
-Layers visualization 
-Object detection vs segmentation - what is useful
-how object detection can improve the segmentation of the model 

-Datasets:
https://cocodataset.org/#explore
https://universe.roboflow.com/citypersons-conversion/citypersons-woqjq/dataset/9
https://eurocity-dataset.tudelft.nl/eval/user/login?_next=/eval/downloads/detection (behind login)
https://synscapes.on.liu.se/ (realistic but synthetic street data)

-Papers/models:
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

A nice bench with different SOTA models for pedestrian detection: https://www.sciencedirect.com/science/article/pii/S2667379723000293 and different datasets

-Metrics:
- object detection
    - Mean Average Precision (MAP): AP for a range of IoU thresholds from 0.05 to 0.95. Very strong
- segmentation
    - Intersection over union (IoU) aka Jacard
    - Dice

Project steps:
- literature review -> find sota architecture + some clever modification
- train a phat model / fine-tunning
- RQs [choose 3 from below]
    - using a buffer between pre-trained model parts
    - visuaizing layers
    - in the wild + weather + lighting + inside / outside built up area
    - racial bias
    - transfer learning
        - people on the road -> people in other setting
        - people -> road signs
    - adversarial attacks
- web blog on GitHub

before Friday
sota
- EVA
- YOLOv8...v12
    - Plenty of models: https://leaderboard.roboflow.com/
- MaskRCNN
- HRNet
what we cat train / architectures
- EVA / HRNet as a backbone feature extractor pre-trained on ImageNet
- Feature pyramid network FPN to convert arbitrary sized input to a proportionally sized featur map  (i.e. Neck)
    - We can use FPN from Mask R-CNN
- YOLO for BB prediction v8-v12 with convolutions, YOLOCAST++ with attention. (i.e. Head)
    - fine-tune existing ones or train from scratch
- Loss depending on a task
    - For detection, smooth L1 loss with focal loss (https://paperswithcode.com/method/focal-loss
    - Smooth L1 loss for detection: https://medium.com/@abhishekjainindore24/smooth-l1-loss-in-object-detection-faf8efd4569a#:~:text=Smooth%20L1%20Loss%20is%20widely,%2Ch)%20of%20bounding%20boxes.
(optional: segmentation)
- MaskR-CNN/DeepLabV3+/BlenMask for detailed segmentation
    - We can use FPN from Mask R-CNN
    - training both detection and segmention may require multi-task loss function
what modification we can apply
- Train on COCO -> Fine-tune on CityPersons
- Increase robustness by addind malicious training data, data augmentation
- classify pedestrians: 
- oclusion is one of the problems
    - we can test for it with Occlusion Sensitivity Score (OSS)
on Friday
- select favourite
- have a meeting with Ana to settle on the best option

model ideas
- Research
    - Pedestron https://github.com/hasanirtiza/Pedestron
    - YOLO as a baseline 
    - fine-tuining YOLO
- Training task
    - take the best YOLO used for pedestrian detection, identify the dataset, the metrics, etc.
    - fine-tune the chosen YOLO (or some other architecture) on a different from the paper dataset, compare it with the paper YOLO
    - improve the chosen YOLO arch and test
    - compare the results of these 3 models
- Research questions
- Vizualization of the last layer plus testing a hypothesis about how the model detect people
- Find a dataset with weather/ distortions or create an augmented dataset
- Transfer learning
    - people on the road -> people in other setting
    - people -> road signs
    - LORA

Plan:
1. fine-tune YOLOv8 on our dataset
2. Answer research questions
3. Attempt a bigger model

Minimum: Compare model performance on different research questoins


<!--  # Project Ideas
- classification
    - https://www.isic-archive.com/ https://challenge.isic-archive.com/data/
    - http://ludo17.free.fr/mitos_2012/dataset.html detection of cancer
    - Edible vs. Toxic Mushroom Classifier
    - AI vs Human created content
 
most often 
    - Emotion recognition
    - Detect hidden objects (weapons, contraband, or anomalies) in X-ray images, similar to airport security scanners. are there datasets??

    - Music Genre Prediction from Album Covers
    - https://github.com/AlexOlsen/DeepWeeds weeds detection
    - Chess Move Predictor Chessboard datasets, Lichess API

 
Denoising 
    - https://paperswithcode.com/sota/single-image-deraining-on-raincityscapes
    - 
    

    
- Other:
    - Transfer learning (model fine-tunned to segment football matches to segment basketball) (or build on general and fine-tune on more specific)
    - Generative approaches -->
