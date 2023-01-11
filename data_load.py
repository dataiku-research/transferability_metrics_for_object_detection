import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
import pandas as pd
import json
from torchvision import transforms
from torchvision.datasets import VOCDetection, CocoDetection, VisionDataset
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
import collections
from xml.etree.ElementTree import Element as ET_Element
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import parse as ET_parse
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, List


class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, size = (256,256), transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.target_size = size

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB') #Images both in RGB and greyscale
        # Resize Image 
        width, height = img.size
        rs = transforms.Resize(self.target_size)
        img = rs.forward(img)
        img = transforms.functional.to_tensor(img)
        #Compute Scaling factor
        width_scale = self.target_size[1] / width
        height_scale = self.target_size[0] / height

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0] * width_scale
            ymin = coco_annotation[i]['bbox'][1] * height_scale
            xmax = xmin + coco_annotation[i]['bbox'][2] * width_scale
            ymax = ymin + coco_annotation[i]['bbox'][3] * height_scale
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]['category_id'])
            areas.append((xmax - xmin) * (ymax - ymin))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Tensorise img_id
        img_id = torch.tensor([img_id], dtype = torch.int64) 
        
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
            #img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    

class Open_Images_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, size =(256, 256)):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.target_size = size

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB') #Images both in RGB and greyscale
        # Resize Image 
        width, height = img.size
        rs = transforms.Resize(self.target_size)
        img = rs.forward(img)
        img = transforms.functional.to_tensor(img)
        #Compute Scaling factor
        width_scale = self.target_size[1] / width
        height_scale = self.target_size[0] / height


        # number of objects in the image
        num_objs = len(coco_annotation)
        
        #Class tags for only vehicles 
        class_tags = ['/m/012n7d', '/m/0199g', '/m/01bjv', '/m/01bjv', '/m/04_sv', '/m/07r04', '/m/01lcw4', '/m/01prls', '/m/01x3jk', '/m/09ct_', '/m/0cmf2', '/m/0k5j', '/m/0pg52']
        class_ids = [2,3,6,42,73,90,247,302,312,342,473,522,558]

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        for i in range(num_objs):
            if coco_annotation[i]['category_id'] in class_ids:
                xmin = coco_annotation[i]['bbox'][0] * width_scale
                ymin = coco_annotation[i]['bbox'][1] * height_scale
                xmax = xmin + coco_annotation[i]['bbox'][2] * width_scale
                ymax = ymin + coco_annotation[i]['bbox'][3] * height_scale
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(coco_annotation[i]['category_id'])
                areas.append((xmax - xmin) * (ymax - ymin))
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        #labels = torch.ones((num_objs,), dtype=torch.int64) if you want to replace with ones
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
            #img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    


class VOC_Base(VisionDataset):

    _SPLITS_DIR = "Main"
    _TARGET_DIR = "Annotations"
    _TARGET_FILE_EXT = ".xml"

    def __init__(
        self,
        root: str,
        image_set: str = "train", 
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        size = (256, 256)
    ):
        super().__init__(root, transforms, transform, target_transform)

        if not os.path.isdir(self.root):
            raise RuntimeError("Dataset not found")

        splits_dir = os.path.join(self.root, "ImageSets", self._SPLITS_DIR)
        split_f = os.path.join(splits_dir, image_set.rstrip("\n") + ".txt")
        with open(os.path.join(split_f)) as f:
            file_names = [x.strip() for x in f.readlines()]

        image_dir = os.path.join(self.root, "JPEGImages")
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]

        target_dir = os.path.join(self.root, self._TARGET_DIR)
        self.annotations = [os.path.join(target_dir, x + self._TARGET_FILE_EXT) for x in file_names]
        self.target_size = size

        assert len(self.images) == len(self.annotations)

    def __len__(self) -> int:
        return len(self.images)
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = Image.open(self.images[index]).convert("RGB")
        width, height = img.size
        rs = transforms.Resize(self.target_size)
        img = rs.forward(img)
        img = transforms.functional.to_tensor(img)
        annotation = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())   
        #Compute Scaling factor
        width_scale = self.target_size[1] / width
        height_scale = self.target_size[0] / height
        
        #Create Annotation
        img_name = annotation['annotation']['filename']
        boxes = []
        labels = []
        areas = []
        objects = annotation['annotation']['object']
        for obj in objects:
                labels.append(obj['name'])
                box = obj['bndbox']
                boxes.append([float(box['xmin']) * width_scale, float(box['ymin']) * height_scale, float(box['xmax']) * width_scale, float(box['ymax']) * height_scale])
                areas.append((float(box['xmax'])-float(box['xmin'])) * width_scale * (float(box['ymax'])-float(box['ymin'])) * height_scale)

        areas = torch.as_tensor(areas, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
    
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = torch.tensor([index], dtype = torch.int64)
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
        

        return img, my_annotation



    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOC_Base.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    

class VOC_OI(VisionDataset):

    def __init__(
        self,
        root: str,
        classes : list,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        size = (256, 256)
    ):
        super().__init__(root, transforms, transform, target_transform)

        if not os.path.isdir(self.root):
            raise RuntimeError("Dataset not found")

        self.class_dict = dict(zip(classes, range(1, len(classes) +1))) # To convert image labels to integers
        self.images = list(sorted(os.listdir(os.path.join(root, "data"))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, "labels"))))

        self.target_size = size

        assert len(self.images) == len(self.annotations)

    def __len__(self) -> int:
        return len(self.images)
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img_path = os.path.join(self.root, "data", self.images[index])
        annotation_path = os.path.join(self.root, "labels", self.annotations[index])

        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        rs = transforms.Resize(self.target_size)
        img = rs.forward(img)
        img = transforms.functional.to_tensor(img)
        annotation = self.parse_voc_xml(ET_parse(annotation_path).getroot())   
        #Compute Scaling factor
        width_scale = self.target_size[1] / width
        height_scale = self.target_size[0] / height
        
        #Create Annotation
        img_name = annotation['annotation']['filename']
        boxes = []
        labels = []
        areas = []
        objects = annotation['annotation']['object']
        for obj in objects:
                labels.append(self.class_dict[obj['name']]) #Conversion of string labels to integers.
                box = obj['bndbox']
                boxes.append([float(box['xmin']) * width_scale, float(box['ymin']) * height_scale, float(box['xmax']) * width_scale, float(box['ymax']) * height_scale])
                areas.append((float(box['xmax'])-float(box['xmin'])) * width_scale * (float(box['ymax'])-float(box['ymin'])) * height_scale)

        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

    
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = torch.tensor([index], dtype = torch.int64)
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
        

        return img, my_annotation


    @staticmethod
    def parse_voc_xml(node: ET_Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(VOC_Base.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {node.tag: {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}}
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


class Global_Wheat_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None, size = (256, 256)):
        self.root = root
        self.transforms = transforms
        self.annotations = pd.read_csv(annotation)
        # Select unique values as IDs
        self.ids = list(sorted(self.annotations.image_id.unique()))
        self.target_size = size

    def __getitem__(self, index):
        # Image ID
        img_id = self.ids[index]
        # path for input image
        path = str(img_id) + '.jpg'
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        width, height = img.size
        rs = transforms.Resize(self.target_size)
        img = rs.forward(img)
        img = transforms.functional.to_tensor(img)
        #Compute Scaling factor
        width_scale = self.target_size[1] / width
        height_scale = self.target_size[0] / height
        # Annotations
        annotation = self.annotations[self.annotations.image_id == img_id].reset_index(drop = True)

        # number of objects in the image
        num_objs = len(annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        areas = []
        for i in range(num_objs):
            bbox = annotation.iloc[i].bbox
            bbox = json.loads(bbox)
            xmin = float(bbox[0]) * width_scale
            ymin = float(bbox[1]) * height_scale
            xmax = xmin + float(bbox[2]) * width_scale
            ymax = ymin + float(bbox[3]) * height_scale
            area = (xmax - xmin) * (ymax - ymin)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)
            areas.append(area)
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Size of bbox (Rectangular)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = torch.tensor([index], dtype = torch.int64)
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
            #img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


class MNIST_Base(VisionDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        size = (256, 256)
    ):
        super().__init__(root, transforms, transform, target_transform)

        if not os.path.isdir(self.root):
            raise RuntimeError("Dataset not found")

        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.imgs = [file for file in self.imgs if file.endswith(".png") ]
        self.annots = list(sorted(os.listdir(os.path.join(root, "labels"))))
        self.target_size = size

        assert len(self.imgs) == len(self.annots)

    def __len__(self) -> int:
        return len(self.imgs)
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target)
        """
        img_path = os.path.join(self.root, "images", self.imgs[index])
        annotation_path = os.path.join(self.root, "labels", self.annots[index])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        rs = transforms.Resize(self.target_size)
        img = rs.forward(img)
        img = transforms.functional.to_tensor(img)
        #Compute Scaling factor
        width_scale = self.target_size[1] / width
        height_scale = self.target_size[0] / height
        
        #Create Annotation

        boxes = []
        labels = []
        areas = []
        with open(annotation_path, "r") as fp:
            for line in list(fp.readlines())[1:]:
                label, xmin, ymin, xmax, ymax = [int(_) for _ in line.split(",")]
                labels.append(label)
                boxes.append([float(xmin) * width_scale, float(ymin) * height_scale, float(xmax) * width_scale, float(ymax) * height_scale])
                areas.append((float(xmax)-float(xmin)) * width_scale * (float(ymax)-float(ymin)) * height_scale)


        areas = torch.as_tensor(areas, dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
    
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = torch.tensor([index], dtype = torch.int64)
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, my_annotation = self.transforms(img, my_annotation)
        

        return img, my_annotation
    

def collate_fn(batch):
    return tuple(zip(*batch))


def BCCD_label_encoder(my_annotation):
    BCCD_dict = {'WBC' : 1, 'RBC' : 2, 'Platelets' : 3}
    annotation = my_annotation
    labels = annotation["labels"]
    labels = [BCCD_dict[label] for label in labels]
    labels = torch.as_tensor(labels, dtype=torch.int64)
    annotation["labels"] = labels
    
    return annotation



def Open_Images_label_encoder(img, my_annotation):
    open_im_classes_to_labels = {2 : 1 ,3 : 2, 6 : 3, 42 : 4, 
                        73 : 5, 90 : 6, 247 : 7, 302 : 8,
                        312 : 9, 342: 10, 473 : 11, 
                        522 : 12, 558 : 13
                       }
    annotation = my_annotation
    labels = annotation["labels"]
    labels = [open_im_classes_to_labels[label.item()] for label in labels]
    labels = torch.as_tensor(labels, dtype=torch.int64)
    annotation["labels"] = labels
    
    return img, annotation


def VOC_label_encoder( my_annotation):
    keys= ['background','Aeroplane',"Bicycle",'Bird',"Boat","Bottle","Bus","Car","Cat","Chair",'Cow',"Diningtable","Dog","Horse","Motorbike",'Person', "Pottedplant",'Sheep',"Sofa","Train","TVmonitor"]
    keys = [key.lower() for key in keys]
    values = [i for i in range(21)]
    VOC_dictionary = dict(zip(keys, values))
    annotation = my_annotation
    labels = annotation["labels"]
    labels = [VOC_dictionary[label] for label in labels]
    labels = torch.as_tensor(labels, dtype=torch.int64)
    annotation["labels"] = labels
    
    return annotation