# Implementation and evaluation of segmentation models from literature on human segmentation tasks

Human segmentation is a computer vision task that isolates human figures from complex backgrounds, ranging from easy, centered figures to 
occluded figures in unfavorable environments, including poor lighting. It is an active and challenging research field with diverse 
applications and approaches toward perfect, or near-perfect, human segmentation. Some of these approaches are generally applied across 
different domains, and as solutions for segmenting objects other than humans. Many of them achieve impressive results on their object of 
concern.

In my experiments, I take ideas from existing literature on how segmentation tasks were approached, whether for humans or other objects,
 and apply them to segmenting human figures in images. I also combine some of these ideas to form new approaches and test the resulting 
 hypotheses. I look into how these various methods perform will perform in real world.

The results are not state of the art. The algorithms and implementations may not exactly match what's suggested in the literature; they're 
inspired by it, not a strict reproduction.

## Datasets  

- **LIP** [2000 images from Human Parsing Dataset](https://huggingface.co/datasets/mattmdjaga/human_parsing_dataset)     
- **COCO** [2000 images of person-class subset of COCO 2017 validation set](https://cocodataset.org)    
- **Penn-Fudan** [170 images from Penn-Fudan Pedestrian Dataset](https://www.cis.upenn.edu/~jshi/ped_html/)    
- **MADS** [1192 images from Martial Arts, Dancing and Sports dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset)       

All datasets contain RGB images with pixel-level binary human segmentation masks. Images were split per dataset using a stratified 
75/10/15 train/validation/test split to ensure each dataset is represented across all three splits except for **Penn-Fudan**, which was a 60/40 split for train/test 

## Models Tested

- [YOLO26 + UNET](models/yolo26_unet/README.md)

## Results

See [results](results) for sample results on images and comparison tables across models and datasets. 

