# YOLO26 + UNET

**Reference:** Sheraz, H., Khan, Z. A., & Awais, M. (2025). Human instance segmentation 
based on omega-shape using deep learning. [Link](https://www.researchgate.net/publication/389842912_Human_Instance_Segmentation_Based_on_Omega-_Shape_Using_Deep_Learning)

---

## In This Paper 

Sheraz, Khan, and Awais looked at human instance segmentation in crowded, cluttered 
environments, where most full-body segmentation approaches break down due to occlusion. 
Instead of segmenting the whole body, they proposed detecting just the "omega-shape" 
region — the head and shoulders — since this part stays visible even when the rest of 
a person is blocked by crowding or overlap. The goal was tighter, cleaner masks with 
less background clutter, useful for downstream tasks like tracking.

They used the Pascal-Part dataset, a publicly available set of additional annotations 
built on top of PASCAL VOC 2010, narrowed to 3,503 human-class images split 70/30 
between training and validation/testing. Two architectures were compared:

- **Mask R-CNN** — a ResNet-101 + FPN backbone with pre-trained weights and transfer 
  learning, extended with a segmentation mask branch that generates pixel-level masks 
  per detected instance
- **YOLO+UNET** — a custom two-stage pipeline where a modified YOLOv3 proposes 
  candidate regions, ROIAlign crops and standardizes each region to a fixed size, 
  and a reduced 14-layer UNET generates the final segmentation mask per crop

Their results showed Mask R-CNN achieving 92.6% accuracy at 6 FPS, and YOLO+UNET 
reaching 88.4% accuracy at 29 FPS — nearly five times faster. They framed this as 
a meaningful accuracy/speed trade-off, positioning YOLO+UNET as the more practical 
choice for real-time applications.

---

## Main Concept

In their modified YOLO+UNET model, YOLO detects the bounding boxes of each instance then ROI-Align crops these instances from the feature vector.
UNET takes each cropped instance one by one as input and gives the segmented output.

---

## The Math Behind the Approach

### Stage 1 — Detection (YOLO)

YOLO (You Only Look Once) frames object detection as a single regression problem. 
The input image is divided into an S×S grid. Each grid cell predicts B bounding boxes, 
a confidence score per box, and class probabilities. The confidence score reflects 
both the likelihood that an object exists and how accurate the predicted box is:

$$\text{Confidence} = P(\text{Object}) \times \text{IoU}^{\text{truth}}_{\text{pred}}$$

Each bounding box is defined by five values: center coordinates (x, y), width (w), 
height (h), and confidence. Non-Maximum Suppression (NMS) removes duplicate boxes 
by keeping only the highest-confidence box where multiple boxes overlap significantly.

In the YOLO+UNET pipeline, YOLO's role is purely detection — it finds where each 
person is in the image and produces a bounding box around them. It does not segment.

### Stage 2 — Crop Standardization (ROIAlign)

Once bounding boxes are produced, ROIAlign extracts and resizes each detected region 
to a fixed spatial size. ROIAlign uses bilinear interpolation to help preserve spatial accuracy at region 
boundaries — important for feeding clean, consistently-sized crops into the segmentation 
stage without misalignment artifacts.

### Stage 3 — Segmentation (UNET)

UNET is a fully convolutional encoder-decoder architecture. The encoder progressively 
downsamples the input through convolutional and max-pooling layers, extracting feature 
maps at multiple scales. The decoder progressively upsamples back to the original 
resolution using transposed convolutions. Skip connections between matching encoder 
and decoder layers preserve spatial detail lost during downsampling. 
The output is a binary mask the same size as the input crop, where each pixel is 
classified as person **(1)** or background **(0)**. The loss function used is a combination of 
Binary Cross-Entropy (BCE) and Dice Loss:

$$\mathcal{L} = \mathcal{L}_{BCE} + \mathcal{L}_{Dice}$$

$$\mathcal{L}_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]$$

$$\mathcal{L}_{Dice} = 1 - \frac{2 \sum y_i \hat{y}_i + \epsilon}{\sum y_i + \sum \hat{y}_i + \epsilon}$$

The BCE penalizes per-pixel prediction errors. Dice Loss directly optimizes the overlap 
between predicted and ground truth masks, making it more robust when person pixels 
are sparse relative to background pixels.

---

## How This Experiment Differs

While the core pipeline structure (detect, 
crop, and segment) remains the same, the differences are shown in the table below:

| Aspect | Sheraz et al. (2025) | This experiment |
|---|---|---|
| Detection model | Modified YOLOv3 | YOLO26 small (pre-trained) |
| Segmentation model | Custom 14-layer UNET (trained from scratch) | UNET, ResNet-50 encoder (ImageNet pre-trained), decoder fine-tuned |
| Target region | Omega-shape (head + shoulders) | Full body |
| Training dataset | Pascal-Part (3,503 human images) | MADS + COCO person subset + LIP subset + Penn-Fudan (3936)  |
| Evaluation dataset | Pascal-Part (held-out test split) | Penn-Fudan, LIP subset, COCO-person subset, MADS |
| Environment |  | AWS SageMaker (ml.c5.2xlarge, CPU) |
| Evaluation metrics | Accuracy, FPS | mIoU, Pixel Accuracy, Precision, Recall |

**Architectural difference**: Their UNET operated on a fixed 
28×28 input (omega-shape crops), while my finetuned UNET receives 
full-body crops of varying aspect ratios, resized to 256×256. This introduces 
aspect-ratio distortion on tall, narrow person crops — a limitation discussed further 
in the results section.

---

## Datasets

The choice of datasets was driven by accessibility and relevance to human segmentation 
across diverse conditions.

| Dataset | Images | What makes it useful |
|---|---|---|
| **LIP** | [2000 images from Human Parsing Dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset) | 19-class part-level annotations; diverse poses, viewpoints, occlusion, and low resolution — binarized to person vs background for this experiment |    
| **COCO 2017** | [2000 images of person-class subset of COCO 2017 validation set](https://cocodataset.org) | Large-scale, real-world scenes; people at varying scales, heavily occluded, in crowds |    
| **MADS** | [1192 images from Martial Arts, Dancing and Sports dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset) | Dynamic full-body motion; martial arts, dancing, sports activities; |    
| **Penn-Fudan** | [170 images from Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/) | Street-scene pedestrian images; |


All masks were standardized to binary format (0 = background, 255 = person) before 
training and evaluation. Ground truth masks from each dataset were binarized at load 
time using `mask > 0`, regardless of the original label format.

---

## Pipeline
- **Step 1**: Input image       
- **Step 2**: YOLO26 (small) for person bounding boxes (class 0, conf ≥ 0.4)       
- **Step 3**: For each detected box, crop and resize to 256×256       
- **Step 4**: UNET (ResNet-50 encoder) - binary mask (256×256)     
- **Step 5**: Resize mask back to original crop dimensions        
- **Step 6**: Place mask into full-image canvas        
- **Step 7**: Final full-image binary mask - evaluate against ground truth    

---

## Evaluation Metrics

Each model is evaluated using five metrics computed per image, then averaged:

| Metric | What it measures |
|---|---|
| **mIoU** | Overlap between predicted and ground truth mask |
| **Pixel Accuracy** | Proportion of correctly classified pixels |
| **Precision** | Of all predicted person pixels, how many were correct |
| **Recall** | Of all actual person pixels, how many were found |
| **FPS** | Full pipeline throughput per image |

**mIoU** and **Recall** are the primary metrics. the accuracy results were misleading at a point since most 
pixels in an image are background, a model predicting nothing could still achieve very high accuracy. 
This is demonstrated in the results section.

---

## Phases of Testing

The UNET decoder was fine-tuned in three separate experiments, each using a 
different training dataset, to understand how training data choice affects 
generalization. YOLO26 remained fixed across all three.

### Phase 1 — MADS-Trained UNET

The UNET decoder was first fine-tuned on 800 randomly selected MADS images for 8 epochs. 
MADS contains dynamic, full-body, single-subject action sequences with clean binary 
masks, a controlled conditions relative to the other datasets.

| Dataset | mIoU | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Penn-Fudan | 0.744 | 0.942 | 0.783 | 0.940 |
| LIP | 0.803 | 0.946 | 0.843 | 0.944 |
| COCO-person | 0.557 | 0.923 | 0.728 | 0.678 |

**Notes:** LIP performed best; its images are mostly single, centered, 
upright subjects, closely resembling the type of clean full-body shots in MADS. 
COCO-person showed the weakest performance, with recall dropping to 0.678. This 
indicated the model was frequently missing people in COCO's more occluded and real-world scenes
— conditions very different from what it was trained on. 

A diagnostic check confirmed that 7% of the COCO validation images received zero 
detections from YOLO26 (small) at the 0.4 confidence threshold, contributing to the 
low recall. However, even on images where YOLO did detect a person (93% of cases), 
the mean crop-level IoU was only 0.507 — confirming the segmentation stage itself, 
not just detection, was underperforming on COCO-style images.

---

### Phase 2 — COCO-Trained UNET

To test whether weakness on the COCO dataset was a domain-coverage gap or something more 
fundamental, the UNET decoder was fine-tuned from on 
800 COCO-person images for 24 epochs, with validation checks every 3 epochs on 
a 200-image held-out validation slice.

Training loss decreased steadily from 0.918 (epoch 1) to 0.180 (epoch 24). 
Validation mIoU peaked at epoch 6 (0.6987) and showed no meaningful improvement 
for the remaining 18 epochs despite training loss continuing to fall. 
The best checkpoint (epoch 6) was saved automatically.

| Dataset | mIoU | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Penn-Fudan | 0.788 | 0.954 | 0.810 | 0.967 |
| LIP | 0.865 | 0.965 | 0.893 | 0.963 | 
| COCO-person (held-out) | 0.659 | 0.947 | 0.767 | 0.774 |

**Notes:** COCO-person mIoU improved from 0.557 to 0.659 and recall 
from 0.678 to 0.774, a substantial gain confirming the Phase 1 weakness was 
primarily a domain-coverage gap. More surprisingly, Penn-Fudan and LIP also 
improved despite neither being used in training. This suggested that COCO's 
greater visual diversity (varied poses, occlusion, clutter) produced more 
generalizable learned features than MADS's controlled sequences.

---

### Phase 3 — Combined-Trained UNET (Best Model)

The combined training experiment was motivated by the Phase 1 and Phase 2 findings: 
training data diversity consistently mattered more than single-domain specialization. 
UNET was fine-tuned again on a stratified combination of all four 
datasets, using a 75/10/15 train/validation/test split applied within each dataset 
before combining, ensuring every dataset was proportionally represented in all 
three splits.

| Split | MADS | COCO | LIP | Penn-Fudan | Total |
|---|---|---|---|---|---|
| Train (75%) | 894 | 1500 | 1500 | 127 | 4021 |
| Validation (10%) | 119 | 200 | 200 | 17 | 536 |
| Test (15%) | 179 | 300 | 300 | 26 | 805 |

Training ran for 10 epochs on AWS SageMaker (ml.c5.2xlarge, CPU). Final evaluation 
was run on the held-out test splits, reported separately per dataset and as a 
combined aggregate.

| Dataset | Images | mIoU | Accuracy | Precision | Recall |
|---|---|---|---|---|---|
| MADS + COCO + LIP + Penn-Fudan | 847 | 0.821 | 0.970 | 0.886 | 0.899 |

**Notes:** the combined-trained UNET produced the strongest results 
across all three training experiments, confirming that training on a diverse pool 
of human images generalizes better than any single-domain training 
approach.

---

## Sample Results  
 ![YOLO26 + UNET Sample](../../results/sample/yolo26_unet.png) 

 ---

## Accuracy Is a Misleading Metric In This Case — Example

In some number of COCO-person validation images, the model produced an entirely empty prediction 
(no person detected) yet reported accuracy of 0.9999. mIoU, Precision, and Recall 
were all 0.0.

| Metric | Value |
|---|---|
| Accuracy | 0.9999 |
| mIoU | 0.000 |
| Precision | 0.000 |
| Recall | 0.000 |

This happens because the annotated person occupied only a tiny fraction of the image's 
total pixels. A model that detects nothing still classifies the overwhelming majority 
of pixels (background) correctly. 

---

## Conclusion

The YOLO26 + UNET pipeline demonstrates that a two-stage detect-then-segment 
approach is a viable and practical method for human instance segmentation across 
diverse image conditions. Across three phases of fine-tuning, the most consistent 
finding was that training data diversity drives generalization more than any single 
architectural choice — a UNET decoder fine-tuned on a combined pool of four datasets 
outperformed single-domain fine-tuning on every dataset tested.

The pipeline's primary bottleneck across all phases was the segmentation stage, 
not detection. YOLO26 reliably located people in the majority of images (93% 
detection rate on COCO-person at a 0.4 confidence threshold). The remaining 
performance gap was driven by the UNET decoder's ability to produce precise 
masks within each detected crop: particularly on harder cases involving occlusion, 
unusual poses, and partial visibility.

## What Could Lead to Better Results

**Better detection:**
A larger YOLO variant (medium or large) on a GPU-enabled environment would likely 
recover some of the 7% zero-detection cases on COCO-person, improving overall 
recall without any change to the segmentation stage.

**Better segmentation encoder:**
Swapping the ResNet-50 encoder for ResNet-101 or ResNet-152 would give the UNET 
decoder richer feature representations to work with, particularly at crop boundaries 
and on partially occluded subjects. This was not tested here due to CPU-only 
compute constraints but is a natural next step.

**Aspect-ratio-preserving resize:**
The current pipeline resizes all crops to a fixed 256×256, squashing tall, narrow 
person crops and distorting boundary details. Replacing this with a padding-based 
resize that preserves the original aspect ratio before feeding crops into UNET 
would reduce this distortion and likely improve precision on tall subjects.

**More training data:**
First two fine-tuning phases were constrained to under 1,000 and the last phase, a little over 4000 training images for the experiments due to compute limitations. 


