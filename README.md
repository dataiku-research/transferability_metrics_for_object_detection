# Transferability Metrics for Object Detection

This repository is the official implementation of [Transferability Metrics for Object Detection](https://arxiv.org/abs/todo). 

![](images/agg_logme.png)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper,  you can use :

**[train_real_1.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/train_real_1.py)** : script to train model on *Real 1* datasets (CHESS, VOC, ...). For each training is save the model (.ptch), the two pickles containing the train and eval loggers and a summary plot of the training

**[train_real_2.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/train_real_2.py)** : script to train model on *Real 2* datasets (boostrapped datasets from open_images). For each training the script save the model (.ptch), the two pickles containing the train and eval loggers and a summary plot of the training. This script support multiprocessing on multiple gpus and multiple machines

**[train_synthetic.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/train_synthetic.py)** : script to train model on synthetic *MNIST OD Datasets. It can be used in for transfer learning or to simulate pretrained models.

--

**[references](https://github.com/dataiku-research/transferability_metrics_for_object_detection/tree/main/references)** folder contains reference scripts from pytorch for object detection. It contains many helper function that are used in other scripts and notebooks


## Extracting features

To extract the global and local level features, you can use : 

**[extract_features.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/extract_features.py)** to extract features for synthetic datasets and for *Real 1* task. You can specify the layers where to extract the features. The script contains one function
to extract features by small batches and a function to aggregate these batches.

**[extract_features_oi.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/extract_features_oi.py)** to extract features for *Real 2* task. You can specify the layers where to extract the features. The script contains one function to extract features by small batches and a function to aggregate these batches. This script works for both ResNet and ViT backbones.


## Computing metrics and results

Scripts for the different transferability metrics are in **[metric.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/metric.py)**.

To compute results, you can use,

**[results_synthetic_data.ipynb](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/results_synthetic_data.ipynb)** for synthetic datasets

**[results_real_data.ipynb](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/results_real_data.ipynb)** for real datasets

The file **[plot_utils.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/plot_utils.py)** contains helper function to draw plots and compute correlations.

## Datasets

Code for generating MNIST like object detection datasets can be found in **add Simona's code**

Code for generating bootstrapped datasets from Open Images can be found in **[create_oi_datasets.ipynb](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/create_oi_datasets.ipynb)**

Other datasets can be donwloaded here : [VOC](http://host.robots.ox.ac.uk/pascal/VOC/), [CHESS](https://public.roboflow.com/object-detection/chess-full), 
[BCCD](https://www.tensorflow.org/datasets/catalog/bccd), [Global Wheat](https://www.kaggle.com/c/global-wheat-detection) and [Open Images](https://storage.googleapis.com/openimages/web/index.html).

**[data_load.py](https://github.com/dataiku-research/transferability_metrics_for_object_detection/blob/main/data_load.py)** : scripts with custom classes for different OD Datasets. COCO-fashion with a json containing all annotations, VOC with an xml annotation file per image, Global Wheat-fashion with a csv containing all annotations and MNIST-like with one text file per image.

## Results

Our model achieves the following performance on :

### Correlation between mAP and transferability metrics for different tasks

![](images/summary_table.png)


## Contributing


Our implementation of TransRate and regularized H-Score have been merged in the [Transfer-Learning Library]((https://github.com/thuml/Transfer-Learning-Library)). For other transferability metrics we've used their implementation.
