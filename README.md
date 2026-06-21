# Implementation and evaluation of segmentation models from literature on human segmentation tasks

Human segmentation is a computer vision task that isolates human figures from complex backgrounds, ranging from easy, centered figures to 
occluded figures in unfavorable environments, including poor lighting. It is an active and challenging research field with diverse 
applications and approaches toward perfect, or near-perfect, human segmentation. Some of these approaches are generally applied across 
different domains, and as solutions for segmenting objects other than humans. Many of them achieve impressive results on their object of 
concern.

In my experiments, I take ideas from existing literature on how segmentation tasks were approached, whether for humans or other objects,
 and apply them to segmenting human figures in images. I also combine some of these ideas to form new approaches and test the resulting 
 hypotheses. What I'm looking for is how these various methods perform on my datasets when applied.

My results are not state of the art. My algorithms and implementations may not exactly match what's suggested in the literature; they're 
inspired by it, not a strict reproduction.

## Datasets

- **LIP** (Look Into Person)
- **COCO** (person subset)
- **Penn-Fudan** Pedestrian Dataset
<!-- - **MADS** (Martial Arts, Dancing and Sports) -->

## Models Tested

- [YOLO26 + UNET](models/yolo26_unet/README.md), inspired by Sheraz et al. (2025)

## Results

See [results/metrics.csv](results/metrics.csv) for the full comparison table across models and datasets. 

