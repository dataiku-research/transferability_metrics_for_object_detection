import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from data_load import COCODataset, Open_Images_Dataset, VOC_Base, Global_Wheat_Dataset, MNIST_Base
from references.engine import train_one_epoch, evaluate
from references import utils
from data_load import VOC_label_encoder, BCCD_label_encoder, Open_Images_label_encoder
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import time
import gc


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_model(dataset = 'BCCD', pretrained  = False, pretrained_backbone=False, trainable_backbone_layers = 5):

  num_classes_dict = {'BCCD' : 4, 'Global_Wheat' : 2, 'Open_Images' : 14, 'COCO' : 81, 'VOC' : 21, 'CHESS' : 14}
 
  num_classes  = num_classes_dict[dataset]
  print('Building model for', dataset, 'with', num_classes, 'classes')

  faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = pretrained, 
                                                                      pretrained_backbone=pretrained_backbone, 
                                                                      trainable_backbone_layers=trainable_backbone_layers)


  # get number of input features for the classifier
  in_features = faster_model.roi_heads.box_predictor.cls_score.in_features

  # replace the pre-trained head with a new one
  faster_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  return faster_model


def main(model, dataset = 'BCCD', size =(128,128), data_dir ='/path/to/dir', batch_size = 8, lr = 0.0001, test_set_size = 0.20, generator = None ):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if dataset == 'Open_Images' : 

        path2data= data_dir + "OPEN_IM/open-images-v6/train/data"
        path2json= data_dir + "OPEN_IM/open-images-v6/train/labels.json"

        dataset  = Open_Images_Dataset(path2data, path2json, size  = size, transforms= Open_Images_label_encoder)
        indices = torch.randperm(len(dataset)).tolist()
        split_index = int(len(indices) * (1-test_set_size))
        dataset_train = torch.utils.data.Subset(dataset, indices[:10000])
        dataset_test = torch.utils.data.Subset(dataset, indices[1000:12000])  

    elif dataset == 'Global_Wheat':

        dataset = Global_Wheat_Dataset(data_dir + 'GLOBAL_WHEAT/train', 
                                data_dir + 'GLOBAL_WHEAT/train.csv',
                                size = size)
        indices = torch.randperm(len(dataset)).tolist()
        split_index = int(len(indices) * (1-test_set_size))
        dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
        dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])  
    
    elif dataset == 'BCCD':
        dataset_train = VOC_Base(data_dir + 'BCCD/BCCD', image_set = 'trainval',
                        size = size, target_transform = BCCD_label_encoder, transforms = None)
        dataset_test = VOC_Base(data_dir + 'BCCD/BCCD', image_set = 'test',
                size = size, target_transform = BCCD_label_encoder, transforms = None)

    elif dataset == 'COCO': 
        path2data=data_dir + "COCO/train2017/train2017"
        path2json=data_dir + "COCO/annotations/instances_train2017.json"

        dataset = COCODataset(path2data, path2json, size = size)
        indices = torch.randperm(len(dataset)).tolist()
        split_index = int(len(indices) * (1-test_set_size))
        dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
        dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])
    
    elif dataset == 'VOC':
        dataset = VOC_Base(data_dir + "VOC2012/", image_set = 'train', size = size, target_transform=VOC_label_encoder)

        indices = torch.randperm(len(dataset)).tolist()
        split_index = int(len(indices) * (1-test_set_size))
        dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
        dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])
    
    elif dataset == 'CHESS':
        path2data=data_dir + "CHESS/train"
        path2json=data_dir + "CHESS/train/_annotations.coco.json"
        dataset = COCODataset(path2data, path2json, size = size)
        indices = torch.randperm(len(dataset)).tolist()
        split_index = int(len(indices) * (1-test_set_size))
        dataset_train = torch.utils.data.Subset(dataset, indices[:split_index])
        dataset_test = torch.utils.data.Subset(dataset, indices[split_index:])
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_Base(root = data_dir + 'MNIST_OD/train', size = size)
        dataset_test = MNIST_Base(root = data_dir + 'MNIST_OD/test', size = size)
  

    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size= batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn, worker_init_fn = seed_worker, generator = g)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size= 1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn, worker_init_fn = seed_worker, generator = g)


    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params, lr=0.005,
    #                             momentum=0.9, weight_decay=0.00005)

    optimizer = torch.optim.Adam(params, lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, 
                                                             patience=10, threshold=0.0001, threshold_mode='rel', 
                                                             cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    # Prepare the list containing the train and eval logs
    avg_loss, avg_loss_classifier, avg_loss_box_reg, avg_loss_objectness, avg_loss_rpn_reg, iter_loss  = ([] for i in range(6))
    mAP_05_95, mAP_50, AR_1, AR_10, stats_list = ([] for i in range(5))

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_log = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)
  

        with torch.no_grad():
            coco_evaluator = evaluate(model, data_loader_test, device=device)
            lr_scheduler.step(coco_evaluator.coco_eval['bbox'].stats[1]) #LR schedule on mAP_50

        #Convert MetricLogger and cocoEvaluator to lists
        avg_loss.append(train_log.meters['loss'].avg)
        avg_loss_classifier.append(train_log.meters['loss_classifier'].avg)
        avg_loss_box_reg.append(train_log.meters['loss_box_reg'].avg)
        avg_loss_objectness.append(train_log.meters['loss_objectness'].avg)
        avg_loss_rpn_reg.append(train_log.meters['loss_rpn_box_reg'].avg)
        iter_loss += train_log.meters['loss'].deque

        stats = coco_evaluator.coco_eval['bbox'].stats
        stats_list.append(stats)
        mAP_05_95.append(stats[0])
        mAP_50.append(stats[1])
        AR_1.append(stats[6])
        AR_10.append(stats[7])

        #To avoid memory leakage
        del train_log
        del coco_evaluator
        gc.collect()

    train_log = {'avg_loss' : avg_loss, 'avg_loss_classifier' : avg_loss_classifier, 'avg_loss_box_reg' : avg_loss_box_reg,
                    'avg_loss_objectness' : avg_loss_objectness, 'avg_loss_rpn_reg' : avg_loss_rpn_reg,
                    'iter_loss' :  iter_loss}
    eval_log = {"mAP_05_95" : mAP_05_95, 'mAP_50' : mAP_50, 'AR_1' : AR_1, 'AR_10' : AR_10, 'stats' : stats_list}

    print("That's it!")
    return train_log, eval_log


def summary_plot(train_log, eval_log, dataset = None, savefig = False, dir = None, vers = 0):
    fig,axes = plt.subplots(5,2, figsize = (12,17))
    axes[0][0].plot(train_log['avg_loss'])
    axes[0][0].set_title('Average loss per epoch')
    axes[0][1].plot(train_log['iter_loss'])
    axes[0][1].set_title('Average loss per 10 iterations')
    axes[1][0].plot(train_log['avg_loss_classifier'])
    axes[1][0].set_title('Average classifier loss per epoch')
    axes[1][1].plot(train_log['avg_loss_box_reg'])
    axes[1][1].set_title('Average bbox reg loss per epoch')
    axes[2][0].plot(train_log['avg_loss_objectness'])
    axes[2][0].set_title('Average objectness loss per epoch')
    axes[2][1].plot(train_log['avg_loss_rpn_reg'])
    axes[2][1].set_title('Average rpn box reg loss per epoch')
    axes[3][0].plot(eval_log['mAP_05_95'])
    axes[3][0].set_title('(AP) @[ IoU=0.50:0.95  per epoch')
    axes[3][1].plot(eval_log['mAP_50'])
    axes[3][1].set_title('(AP) @[ IoU=0.50  per epoch')
    axes[4][0].plot(eval_log['AR_1'])
    axes[4][0].set_title('(AR) maxDets = 1  per epoch')
    axes[4][1].plot(eval_log['AR_10'])
    axes[4][1].set_title('((AR) maxDets = 10  per epoch')
    if dataset:
        fig.suptitle('Metrics Summary for Resnet on ' + dataset)
    if savefig:
        plt.savefig(dir + dataset + 'iter_' + str(vers) +'.png')


def main():
    seed = 6728131 
    pretrained = False
    pretrained_backbone = True
    vers = 21

    for dataset in ['Open_Images', 'VOC', 'BCCD', 'CHESS', 'Global_Wheat']:


        seed_all(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        # Load first and train only head
        model = build_model(dataset = dataset, pretrained = pretrained, pretrained_backbone = pretrained_backbone, trainable_backbone_layers=0)

        train_logger, eval_logger = main(dataset = dataset, model = model, num_epochs=30, lr = 0.0001, generator=g)
    
        # Save weights and logs of the head
        param_dict_base = model.state_dict()


        output_dir = '/data.nfs/AUTO_TL_OD/'
        dir = output_dir + "ft_models/" + dataset + '/'


        torch.save(param_dict_base, dir + dataset + 'iter_' + str(vers) + '.ptch')
        summary_plot(train_logger, eval_logger, dataset = dataset, savefig = True, dir = dir, vers = vers )
        pickle.dump(train_logger, open(dir + dataset + '_train' + 'iter_' + str(vers) + '.pkl', "wb"))
        pickle.dump(eval_logger, open(dir + dataset + '_eval' + 'iter_' + str(vers) + '.pkl', "wb"))

if __name__ == '__main__':
    main()

