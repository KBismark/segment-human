
<h1 align="center">YOLO + UNET for human segmentation tasks</h1>
<p align="center"><a href="https://huggingface.co/spaces/Kbis/segment-human">Live Demo (HuggingFace)</a></p>    

<img width="1200" height="797" alt="YOLO26 + UNET Results Sample" src="https://github.com/user-attachments/assets/31afa36f-6d1d-4d83-a55f-70750cb8db4a" />

Human segmentation is a computer vision task that isolates human figures from complex backgrounds, ranging from easy, centered figures to 
occluded figures in unfavorable environments, including poor lighting. It is an active and challenging research field with diverse 
applications and approaches toward perfect, or near-perfect, human segmentation. Some of these approaches are generally applied across 
different domains, and as solutions for segmenting objects other than humans. Many of them achieve impressive results on their object of 
concern.

In this experiment, I apply instance segmentation to segment human figures. The main concepts:    
- YOLO model detects the bounding boxes of each instance
- Region of Interest Align crops and standardizes these instances from the feature vector.
- UNET takes each cropped instance one by one as input and gives the segmented output.
 

## Pipeline
- **Step 1**: Input image       
- **Step 2**: YOLO26 (small) for person bounding boxes (conf ≥ 0.4)       
- **Step 3**: For each detected box, crop and resize to 256×256       
- **Step 4**: UNET (ResNet-50 encoder) produces binary mask (256×256)     
- **Step 5**: Resize mask back to original crop dimensions        
- **Step 6**: Place mask into full-image canvas        
- **Step 7**: Final full-image binary mask

## Evaluation Metrics

Each model is evaluated using five metrics computed per image, then averaged:

| Metric | What it measures |
|---|---|
| **mIoU** | Overlap between predicted and ground truth mask |
| **Pixel Accuracy** | Proportion of correctly classified pixels |
| **Precision** | Of all predicted person pixels, how many were correct |
| **Recall** | Of all actual person pixels, how many were found |
| **FPS** | Full pipeline throughput per image |

## Fine-tuning and Testing
Ran on NVIDIA T4 GPU on Colab | 16 RAM | Batch size = 8 | 50 Epochs.   

The UNET decoder was fine-tuned using the validation set of the COCO dataset due to resource limits. Only images containing at least one person were used for training. In all, there were 2693 images of the person class in the validation set. The encoder uses ImageNet pretrained weights with a ResNet50 backbone. NVIDIA T4 Tensor Core GPU Testing was done entirely on a different dataset using the Penn Fudan Pedestrian dataset. Results can be found in the summary csv file in the results folder. A summary result is provided below:

| Dataset | mIoU | Accuracy | Precision | Recall | FPS |
|---|---|---|---|---|---|
| Penn-Fudan | 0.807 | 0.958 | 0.829 | 0.967 | 14.49 |

## Installation
- Clone this repo with `git clone https://github.com/KBismark/segment-human.git`.    
-  Install dependencies `pip install -r requirements.txt`.
-  Check the respective files for downloading the training dataset and the test dataset in the `dataset/` folder.
-  You may use the [colab.ipynb](./colab.ipynb) to save time   

## Inference
You can get the pretrained weights at [Google Drive](https://drive.google.com/file/d/1MK7L2T2A3VQ5Zgn_2YlYkxEWg_s5caAa/view?usp=sharing) and save in the `checkpoints/` folder. Run `python test.py` to test on the Penn Fudan Pedestrian dataset.     

Running `python test.py` without the saved checkpoints will automatically begin training and also run the tests after.     




