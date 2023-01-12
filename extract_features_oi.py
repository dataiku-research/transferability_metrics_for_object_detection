import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data_load import VOC_OI
from typing import Dict, Iterable, Callable
from torch import nn, Tensor
from torch.nn.functional import avg_pool2d
import numpy as np
from references import utils
from torchvision.ops import roi_align, MultiScaleRoIAlign
import gc
import os
import itertools


def build_model_eval():

    faster_model = fasterrcnn_resnet50_fpn(pretrained= True)

    # get number of input features for the classifier
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, 6)

    faster_model.eval()

    return faster_model


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


def extract_features(dataset,transformer = True, backbone_layers = [0,1,2,3,4], 
                     fpn_layers = [0,1,2,3], fpn_ms = True, penultimate = False,
                     batch_size=4, subsample = 1000, size=(800, 800), 
                     output_dir = None):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = build_model_eval()
    model.to(device)

    #Create list of features to extract
    extracted_layers = []
    all_backbone_layers = ["backbone.body.maxpool", "backbone.body.layer1", "backbone.body.layer2",
                                                      "backbone.body.layer3", "backbone.body.layer4"]
    if backbone_layers is not None :
        chosen_backbone_layers = [all_backbone_layers[i] for i in backbone_layers]
        extracted_layers += chosen_backbone_layers
    if (fpn_layers is not None) or (fpn_ms is True) or (penultimate is True):
        extracted_layers += ['backbone.fpn']

    print('List of extracted layers : ', extracted_layers, 'if penultimate = True will use features extracted from fpn_ms to pass trough the head')
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
        #print('Features extracted')


    
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
            
        
        if fpn_ms == True  or penultimate == True:
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
            
            if penultimate == True :

                if transformer == True : 
                    with torch.no_grad():
                        features_bbox = model.roi_heads.box_head(features_bbox)
                    
                    torch.save(
                        features,  output_dir + f"penultimate/feats_batch_{i}.pt")
                    torch.save(
                        features_bbox,  output_dir + f"penultimate/feats_bbox_batch_{i}.pt")

                else : 
                    features_bbox = features_bbox.flatten(start_dim=1) # Convert (N, 256, 7, 7) to (N, 12544)
                    with torch.no_grad():
                        features_bbox = model.roi_heads.box_head.fc6(features_bbox)
                        features_bbox = model.roi_heads.box_head.fc7(features_bbox)

                    torch.save(
                        features,  output_dir + f"penultimate/feats_batch_{i}.pt")
                    torch.save(
                        features_bbox,  output_dir + f"penultimate/feats_bbox_batch_{i}.pt")


        #Save Labels
        bboxes = torch.cat(all_boxes, dim=0)
        labels = torch.reshape(torch.cat(all_labels, dim=0), (-1, 1)).to(device)
        Y = torch.cat((bboxes, labels), dim=1)

        torch.save(
            Y,  output_dir + f"labels/labels_batch_{i}.pt")


        #print('Features saved')
        del outputs
        gc.collect()


def assemble_batches(dataset_dir= None, backbone_layers = [0,1,2,3,4], fpn_layers = [0,1,2,3], fpn_ms = True, penultimate = False):

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
    if penultimate == True:
        layer_names.append('penultimate')

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

        if layer != 'penultimate':
            all_features_bbox = torch.reshape(all_features_bbox, (n_boxes, n_dim)) # If penultimate tensor is already in (n_boxes, n_dim)

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

    classes_list = [['Snack', 'Person', 'Salad', 'Vegetable', 'Fast food'],
                    ['Vehicle', 'Human hair', 'Mammal', 'Auto part', 'Boat'],
                    ['Wheel', 'Bicycle wheel', 'Clothing', 'Sports equipment', 'Bicycle'],
                    ['Human leg', 'Human hand', 'Clothing', 'Human hair', 'Human arm'],
                    ['Human mouth', 'Mammal', 'Human head', 'Girl', 'Human hair'],
                    ['Plant', 'Window', 'Wheel', 'Person', 'Houseplant'],
                    ['Human hair', 'Human arm', 'Person', 'Sports equipment', 'Footwear'],
                    ['Food', 'Fish', 'Salad', 'Snack', 'Seafood'],
                    ['Human head', 'Sports equipment', 'Human body', 'Human hand', 'Girl'],
                    ['Footwear', 'Boy', 'Mammal', 'Sports equipment', 'Sports uniform'],
                    ['Human leg', 'Mammal', 'Woman', 'Tree', 'Boat'],
                    ['Wheel', 'Person', 'Clothing', 'Man', 'Fixed-wing aircraft'],
                    ['Plant', 'Clothing', 'Tire', 'Car', 'Land vehicle'],
                    ['Dog', 'Animal', 'Human face', 'Clothing', 'Carnivore'],
                    ['Clothing', 'Dog', 'Human hair', 'Mammal', 'Animal'],
                    ['Person', 'Human hair', 'Drink', 'Woman', 'Tableware'],
                    ['Tree', 'Mammal', 'Plant', 'Human ear', 'Bird'],
                    ['Food', 'Baked goods', 'Fast food', 'Dairy Product', 'Snack'],
                    ['Footwear', 'Land vehicle', 'Plant', 'Man', 'Carnivore'],
                    ['Car', 'Tire', 'Land vehicle', 'Tree', 'Vehicle']]

    print("Extract features for bootstrapped datasets from COCO")
    for i in range(1,21):

        dataset = VOC_OI('/path/to/data/' + f'dataset_{i}/', classes_list[i -1], size = (800, 800))

        extract_features(dataset, transformer = True, output_dir = f"/path/to/features/vit/dataset_{i}/", 
                         size = (800,800), backbone_layers= [4], fpn_layers= [0,1,2,3], fpn_ms= True, penultimate= True)
        assemble_batches(dataset_dir=f"/path/to/features/vit/dataset_{i}/", 
                         backbone_layers= [4], fpn_layers=  [0,1,2,3], fpn_ms= True, penultimate= True)
