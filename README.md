<h1 align="center">YOLO + UNET for human segmentation tasks</h1>
<p align="center"><a href="https://huggingface.co/spaces/Kbis/segment-human">Live Demo (HuggingFace)</a></p>    

>   
> ![YOLO26 + UNET Sample](results/sample/yolo26_unet.png)    
>

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


## Datasets  

- **LIP** [2000 images from Human Parsing Dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset)     
- **COCO** [2000 images of person-class subset of COCO 2017 validation set](https://cocodataset.org)    
- **Penn-Fudan** [170 images from Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)    
- **MADS** [1192 images from Martial Arts, Dancing and Sports dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset)       

## Findings
Results of this experiment is documented here ([models/yolo26_unet](models/yolo26_unet))    
