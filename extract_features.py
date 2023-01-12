import torch
from torchvision.datasets.vision import data
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data_load import COCODataset, Open_Images_Dataset, VOC_Base, Global_Wheat_Dataset, MNIST_Base
from data_load import VOC_label_encoder, BCCD_label_encoder, Open_Images_label_encoder
from torch import nn
from torch.nn.functional import avg_pool2d
import numpy as np
from references import utils
from torchvision.ops import roi_align, MultiScaleRoIAlign
import gc
import os
import itertools


def build_model_eval(dataset_name='BCCD', dataset_source = 'COCO'):

    num_classes_dict = {'BCCD': 4, 'Global_Wheat': 2, 'Open_Images': 14,
                        'COCO': 81, 'VOC': 21, 'CHESS': 14, 'MNIST': 11, 'EMNIST' : 27, 'KMNIST' : 11, 'FASHION_MNIST': 11, 'USPS':11}

    num_classes = num_classes_dict[dataset_name]
    print('Building model for', dataset_name, 'with', num_classes, 'classes')

    faster_model = fasterrcnn_resnet50_fpn(pretrained= True)

    # get number of input features for the classifier
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features


# Replace weights if using a model pretrained not on COCO
    if dataset_source != 'COCO' : 

        faster_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes_dict[dataset_source]) #As we load the weights from the pretrained model, head needs to have the same shape

        file_path =  f"/path/to/model/ft_models/5_layers/{dataset_source}/{dataset_source}iter_0.ptch"
        param_dict_base = torch.load(file_path)
        print('Loading weights from ' , dataset_source)
        faster_model.load_state_dict(param_dict_base) #Load weights of the pretrained model


    # replace the pre-trained head with a new one
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)

    faster_model.eval()

    return faster_model


from typing import Dict, Iterable, Callable
from torch import nn, Tensor


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        _ = self.model(x)
        return self._features

# The feature extractor can be used to extract features from multiple layers in one pass. But it reduces the maximum batch size
# as it needs more memory.


def extract_features(dataset_name='BCCD',dataset_source = 'COCO', data_dir = None, backbone_layers = [0,1,2,3,4], 
                     fpn_layers = [0,1,2,3], fpn_ms = True, fc_7 = False,
                     batch_size=4, subsample = 1000, size=(800, 800), 
                     output_dir = None):
                
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = build_model_eval(dataset_name=dataset_name, dataset_source= dataset_source).to(device)

    if dataset_name == 'BCCD':
        dataset = VOC_Base(data_dir + 'BCCD/BCCD', image_set='trainval',
                           size=size, target_transform=BCCD_label_encoder, transforms=None)
    elif dataset_name == 'CHESS':

        path2data = data_dir + "CHESS/train"
        path2json = data_dir + "CHESS/train/_annotations.coco.json"
        dataset = COCODataset(path2data, path2json, size=size)

    elif dataset_name == 'VOC':
        dataset = VOC_Base(data_dir + "VOC2012/", image_set='train',
                           size=size, target_transform=VOC_label_encoder)

    elif dataset_name == 'Open_Images':
        path2data = data_dir + "OPEN_IM/open-images-v6/train/data"
        path2json = data_dir + "OPEN_IM/open-images-v6/train/labels.json"
        dataset = Open_Images_Dataset(
            path2data, path2json, size=size, transforms=Open_Images_label_encoder)

    elif dataset_name == 'Global_Wheat':
        dataset = Global_Wheat_Dataset(
            data_dir + 'GLOBAL_WHEAT/train', data_dir + 'GLOBAL_WHEAT/train.csv', size=size)

    else: 
        data_dir = '/data.nfs/AUTO_TL_OD/data/'

    if dataset_name == 'MNIST':
        dataset= MNIST_Base(root = data_dir + 'mnist_detection/train', size = size)

    elif dataset_name == 'KMNIST':
        dataset= MNIST_Base(root = data_dir + 'kmnist_detection/train', size = size)

    elif dataset_name == 'EMNIST':
        dataset= MNIST_Base(root = data_dir + 'emnist_detection/train', size = size)

    elif dataset_name == 'FASHION_MNIST':
        dataset= MNIST_Base(root = data_dir + 'fashionmnist_detection/train', size = size)

    elif dataset_name == 'USPS':
        dataset= MNIST_Base(root = data_dir + 'usps_detection/train', size = size)
    
    #Create list of features to extract
    extracted_layers = []
    all_backbone_layers = ["backbone.body.maxpool", "backbone.body.layer1", "backbone.body.layer2",
                                                      "backbone.body.layer3", "backbone.body.layer4"]
    if backbone_layers is not None :
        chosen_backbone_layers = [all_backbone_layers[i] for i in backbone_layers]
        extracted_layers += chosen_backbone_layers
    if (fpn_layers is not None) or (fpn_ms is True) or (fc_7 is True):
        extracted_layers += ['backbone.fpn']

    print('List of extracted layers : ', extracted_layers)

    # Instantiate the feature extractor
    resnet_features = FeatureExtractor(model, layers= extracted_layers)

    torch.manual_seed(0)  # For reproducibility

    indices = torch.randperm(len(dataset)).tolist()
    dataset_cut = torch.utils.data.Subset(dataset, indices[:subsample])
    data_loader = torch.utils.data.DataLoader(
        dataset_cut, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    for i, data in enumerate(data_loader):
        X, Y = data
        X = torch.stack(X)  # Stack tensors in one

        # Transform targets
        all_boxes = []  # List of tensors [B, 4] with B number of boxes per image
        all_labels = []
        n_boxes = []  # Number of boxes in each image
        for j in range(len(Y)):
            annot = Y[j]  # extract label of object j
            all_boxes.append(annot['boxes'].to(device))
            all_labels.append(annot['labels'])
            n_boxes.append(len(annot['labels']))

        n_boxes = torch.tensor(n_boxes).to(device)
        X = X.to(device)
        with torch.no_grad():
            outputs = resnet_features(X)
    
        map_sizes = [200, 200, 100, 50, 25]  #  sizes of the feature maps

        # For all layers from backbone, extract feature maps of bounding boxes and feature maps of all image with duplicates
        if backbone_layers is not None:

            if 0 in backbone_layers : 
            # Rename output for code factorization
                outputs['backbone.body.layer0'] = outputs.pop(
                    'backbone.body.maxpool') 
            
            for l in backbone_layers: 
                features = avg_pool2d(
                    input=outputs[f"backbone.body.layer{l}"], kernel_size= map_sizes[l])
                # Duplicate feature maps
                features = torch.repeat_interleave(features, n_boxes, dim=0)

                features_bbox = roi_align(input=outputs[f"backbone.body.layer{l}"], boxes = all_boxes, output_size=(7, 7),
                    spatial_scale= map_sizes[l]/size[0], sampling_ratio=2)
                features_bbox = avg_pool2d(input= features_bbox, kernel_size= 7)
                torch.save(
                    features,  output_dir + f"layer_{l+1}/feats_batch_{i}.pt")
                torch.save(
                    features_bbox,  output_dir + f"layer_{l+1}/feats_bbox_batch_{i}.pt")

        if fpn_layers is not None:
            for l in fpn_layers: 
                features = avg_pool2d(
                    input=outputs['backbone.fpn'][str(l)], kernel_size= map_sizes[l+1]) #l+1 as there are only 4 layers in fpn
                # Duplicate feature maps
                features = torch.repeat_interleave(features, n_boxes, dim=0)

                features_bbox = roi_align(input= outputs['backbone.fpn'][str(l)], boxes = all_boxes, output_size=(7, 7),
                    spatial_scale= map_sizes[l+1]/size[0], sampling_ratio=2)

                features_bbox = avg_pool2d(input= features_bbox, kernel_size= 7)
                torch.save(
                    features,  output_dir + f"fpn_{l}/feats_batch_{i}.pt")
                torch.save(
                    features_bbox,  output_dir + f"fpn_{l}/feats_bbox_batch_{i}.pt")
            
        
        if fpn_ms == True  or fc_7 == True:
            m = MultiScaleRoIAlign(['0', '1', '2', '3'], output_size=(7 ,7), sampling_ratio=2)
            # Duplicate feature maps of the last fpn layer
            features = outputs['backbone.fpn']['3']
            features = avg_pool2d(
                input= features, kernel_size= map_sizes[4])
            features = torch.repeat_interleave(features, n_boxes, dim=0)
            
            #Take the features
            features_bbox = m(outputs['backbone.fpn'] , all_boxes, len(Y) * [(size[0], size[1])]) #If images are fed in a size smaller than 800, this could be wrong
            features_bbox_ms = avg_pool2d(input= features_bbox, kernel_size= 7)

            if fpn_ms == True : 
                torch.save(
                    features,  output_dir + f"fpn_ms/feats_batch_{i}.pt")
                torch.save(
                    features_bbox_ms,  output_dir + f"fpn_ms/feats_bbox_batch_{i}.pt")
            
            if fc_7 == True :
                features_bbox = features_bbox.flatten(start_dim=1) # Convert (N, 256, 7, 7) to (N, 12544)
                with torch.no_grad():
                    features_bbox = model.roi_heads.box_head.fc6(features_bbox)
                    features_bbox = model.roi_heads.box_head.fc7(features_bbox)

                torch.save(
                    features,  output_dir + f"fc_7/feats_batch_{i}.pt")
                torch.save(
                    features_bbox,  output_dir + f"fc_7/feats_bbox_batch_{i}.pt")


        #Save Labels
        bboxes = torch.cat(all_boxes, dim=0)
        labels = torch.reshape(torch.cat(all_labels, dim=0), (-1, 1)).to(device)
        Y = torch.cat((bboxes, labels), dim=1)

        torch.save(
            Y,  output_dir + f"labels/labels_batch_{i}.pt")


        #print('Features saved')
        del outputs
        gc.collect()



def assemble_batches(dataset_dir= None, backbone_layers = [0,1,2,3,4], fpn_layers = [0,1,2,3], fpn_ms = True, fc_7 = False):

    #Create list of layer names to extract
    layer_names = []
    if backbone_layers is not None:
        backbone_layer_names = [f'layer_{i+1}' for i in backbone_layers]
        layer_names += backbone_layer_names
    if fpn_layers is not None:
        fpn_layer_names = [f'fpn_{i}' for i in fpn_layers]
        layer_names += fpn_layer_names
    if fpn_ms == True:
        layer_names.append('fpn_ms')
    if fc_7 == True:
        layer_names.append('fc_7')

    print("Assemble batches for layers :", layer_names)

    for layer in layer_names : 
        all_features = []
        all_features_bbox = []
        for file in os.listdir(dataset_dir + f"/{layer}/"):
            if file.startswith("feats_batch"):
                filename = os.path.join(
                    dataset_dir + f"/{layer}/", file)
                feats = torch.load(filename)
                all_features.append(feats)
            if file.startswith("feats_bbox_batch"):
                filename = os.path.join(
                    dataset_dir + f"/{layer}/", file)
                feats_bbox = torch.load(filename)
                all_features_bbox.append(feats_bbox)

        all_features = torch.cat(all_features, dim=0)
        all_features_bbox = torch.cat(all_features_bbox, dim=0)

        # Reshape from (n_boxes, n_dim, 1, 1) to (n_boxes, n_dim)
        n_boxes, n_dim, _, _ = all_features.shape
        all_features = torch.reshape(all_features, (n_boxes, n_dim))

        if layer != 'fc_7':
            all_features_bbox = torch.reshape(all_features_bbox, (n_boxes, n_dim)) # If fc_7 tensor is already in (n_boxes, n_dim)

        # Save concatenated features
        torch.save(
            all_features, dataset_dir + f"/{layer}/all_features.pt")
        torch.save(all_features_bbox,
                    dataset_dir + f"/{layer}/all_features_bbox.pt")
 
    #Assemble label batches
    all_labels = []
    for file in os.listdir(dataset_dir + f"/labels/"):
        if file.startswith("labels_batch"):
            filename = os.path.join(
                dataset_dir + f"/labels/", file)
            labels = torch.load(filename)
            all_labels.append(labels)

    all_labels = torch.cat(all_labels, dim=0)
    torch.save(
        all_labels, dataset_dir + f"/labels/all_labels.pt")

if __name__ == "__main__":

    synth_datasets = ['MNIST', 'EMNIST', 'KMNIST', 'FASHION_MNIST', 'USPS']
    for dataset_source, dataset_target in itertools.permutations(synth_datasets, 2):
        print('Extract features for ', dataset_target, 'from ', dataset_source)
        extract_features(dataset_target, dataset_source, output_dir = f"/path/to/features{dataset_target}/from_{dataset_source}/", 
                         size = (800,800), backbone_layers= [4], fpn_layers= [0,1,2,3], fpn_ms= True, fc_7= True)
        assemble_batches(dataset_dir=f"/path/to/features{dataset_target}/from_{dataset_source}/", 
                         backbone_layers= [4], fpn_layers=  [0,1,2,3], fpn_ms= True, fc_7= True)


    for dataset in ['BCCD', 'CHESS', 'VOC', 'Open_Images', 'Global_Wheat']:
        print('Extract features for ', dataset)
        extract_features(dataset, dataset_source = 'COCO', output_dir = f"/path/to/features{dataset}/", 
                         size = (800,800), backbone_layers= [4], fpn_layers= [0,1,2,3], fpn_ms= True, fc_7= True)
        assemble_batches(dataset_dir=f"/path/to/features{dataset}/", 
                         backbone_layers= [4], fpn_layers=  [0,1,2,3], fpn_ms= True, fc_7= True)

  




