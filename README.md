# Self-Supervised Pre-Training of YOLOv8 via LeJEPA for Agricultural Phenotyping

This repository contains the codebase, evaluation scripts, and experimental figures for the methodological study on **Self-Supervised Pre-Training of YOLOv8 via LeJEPA**. The project explores non-contrastive self-supervised learning for agricultural computer vision, specifically focusing on downstream tasks like cotton boll detection and defoliated plot status classification.

## Overview

Agricultural computer vision pipelines often face data scarcity due to the high labor costs associated with manual annotations in outdoor field environments. This project mitigates this limitation by pre-training a YOLOv8 convolutional backbone on unannotated multi-camera field images using the **LeJEPA** (Joint-Embedding Predictive Architecture with Sketched Isotropic Gaussian Regularization) framework. 

Unlike traditional contrastive approaches (e.g., SimCLR, MoCo) that rely on negative pairs, this approach grounds self-supervised prediction in optimal distribution theory using **SIGReg**. To adapt this for object detection, the project implements a **Dense Multi-Scale Prediction Loss** across the spatial feature grids ($P_3, P_4, P_5$) while strictly utilizing non-geometric augmentations (blur, noise, color jitter) to preserve pixel-to-pixel coordinate alignment.

## Downstream Tasks

The pre-trained YOLOv8 backbone is evaluated on two distinct downstream agricultural tasks:
1. **Object Detection (Cotton Boll Detection):** End-to-end fine-tuning of the pre-trained backbone, neck, and detection head for bounding box regression.
2. **Image Classification (Plot Status Classification):** Linear probing on a frozen backbone to classify defoliated cotton fields into three categories: `in_plot`, `between_plots`, and `headland`.

## Datasets

The datasets required to run the pre-training and downstream evaluation codes can be found and downloaded from the BSAIL data repository:
* **Dataset Link:** [https://uflbsail.net/data/](https://uflbsail.net/data/)

*Note: The pre-training phase utilizes a subset (50,000 images) of the MARS-X multi-camera dataset to optimize computational overhead, leveraging physical camera angles tailored for top-down and oblique canopy perspectives.*

## Acknowledgments and Citations

This project builds upon and draws inspiration from the foundational contrastive learning and multi-camera plant phenotyping research conducted by Daniel Petti and the UGA-BSAIL team. 

**Source Repository:** Please visit the original contrastive learning repository that inspired the multi-camera sampling approach used in this project:
[UGA-BSAIL / self-supervised-learning](https://github.com/UGA-BSAIL/self-supervised-learning)

**Relevant Citations:**

If you use this code or the associated datasets, please cite the following foundational work:

* **Petti, D., Li, C., & Liu, N. (2026).** *Contrastive multi-view representation learning for multi-camera plant phenotyping: A cotton field study.* Plant Phenomics, 8(2), 100193.
* **Balestriero, R., & LeCun, Y. (2025).** *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* arXiv preprint arXiv:2511.08544.
* **Wang, X., Zhang, R., Shen, C., Kong, T., & Li, L. (2021).** *Dense contrastive learning for self-supervised visual pre-training.* In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 8514-8523).
