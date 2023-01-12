import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from references.engine import train_one_epoch, evaluate
from references import utils
from data_load import MNIST_Base
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import time
import gc
import itertools


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

def build_model(dataset = 'MNIST', trainable_backbone_layers = 5, source_dataset = 'None'):

    num_classes_dict = {'MNIST' : 11, 'FASHION_MNIST' : 11, 'KMNIST' : 11, 'USPS' : 11, 'EMNIST': 27}

    num_classes  = num_classes_dict[dataset]
    print('Building model for', dataset, 'with', num_classes, 'classes')


    # Pretrained = False as we don't want to load COCO weights in this experiment
    faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = False, 
                                                                        pretrained_backbone= True, 
                                                                        trainable_backbone_layers=trainable_backbone_layers)

    # get number of input features for the classifier
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features


    # Replace weights if using a pretrained model
    if source_dataset is not None : 

        faster_model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes_dict[source_dataset]) #As we load the weights from the pretrained model, head needs to have the same shape

        file_path =  f"/path/to/model/ft_models/5_layers/{source_dataset}/{source_dataset}iter_0.ptch"
        print('Loading weights from ' , source_dataset)
        param_dict_base = torch.load(file_path)
        faster_model.load_state_dict(param_dict_base) #Load weights of the pretrained model


    # replace the pre-trained head with a new one
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features, num_classes)


    return faster_model


def train(model, dataset = 'MNIST', size =(128,128), num_epochs = 5, batch_size = 8, lr = 0.0001, g = None):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if dataset == 'MNIST':
        dataset_train = MNIST_Base(root = '/path/to/data/mnist_detection/train', size = size)
        dataset_test = MNIST_Base(root ='/path/to/data/mnist_detection/test', size = size)

    if dataset == 'KMNIST':
        dataset_train = MNIST_Base(root = '/path/to/data/kmnist_detection/train', size = size)
        dataset_test = MNIST_Base(root ='/path/to/data/kmnist_detection/test', size = size)

    if dataset == 'EMNIST':
        dataset_train = MNIST_Base(root = '/path/to/data/emnist_detection/train', size = size)
        dataset_test = MNIST_Base(root = '/path/to/data/emnist_detection/test', size = size)

    if dataset == 'FASHION_MNIST':
        dataset_train = MNIST_Base(root = '/path/to/data/fashionmnist_detection/train', size = size)
        dataset_test = MNIST_Base(root ='/path/to/data/fashionmnist_detection/test', size = size)

    if dataset == 'USPS':
        dataset_train = MNIST_Base(root = '/path/to/data/usps_detection/train', size = size)
        dataset_test = MNIST_Base(root ='/path/to/data/usps_detection/test', size = size)

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


def train(dataset = 'MNIST', seed = 6728131, vers = 0, n_epochs = [5],  transfer = False, source_dataset =  None):

    seed_all(seed)
    g = torch.Generator()
    g = g.manual_seed(seed)

    times = []

    output_dir = 'path/to/dir'
    dir = output_dir + "ft_models/base/" + dataset + '/'

    # Load first and train only head (first step)
    model = build_model(dataset = dataset, trainable_backbone_layers=0, source_dataset = source_dataset)


    #To comment for one step training / or imageNet
    start_time = time.time()
    train_logger, eval_logger = train(dataset = dataset, model = model, num_epochs= n_epochs[0], lr = 0.0001, g= g)
    total_time = time.time() - start_time

    # Save weights and logs of the head
    param_dict_base = model.state_dict()

    torch.save(param_dict_base, dir + dataset + 'iter_' + str(vers) + '.ptch')

    #To comment for one step training
    #train_log, eval_log = convert_loggers(train_logger, eval_logger)
    summary_plot(train_logger, eval_logger, dataset = dataset, savefig = True, dir = dir, vers = vers )
    pickle.dump(train_logger, open(dir + dataset + '_train' + 'iter_' + str(vers) + '.pkl', "wb"))
    pickle.dump(eval_logger, open(dir + dataset + '_eval' + 'iter_' + str(vers) + '.pkl', "wb"))
    times.append(total_time)

    # Delete model to free memory
    del model
    del param_dict_base
    gc.collect()


    if not transfer: 
        #Reload the weights of the pretrained head from first_step
        param_dict_base = torch.load(output_dir + "ft_models/base/" + dataset + '/' + dataset + 'iter_' + str(vers) + '.ptch')

        dir = output_dir + f"ft_models/5_layers/" + dataset + '/'

        model = build_model(dataset = dataset, trainable_backbone_layers= 5)
        model.load_state_dict(param_dict_base) # Load the parameters of pretrained head

        start_time = time.time()
        train_logger, eval_logger = train(dataset = dataset, model = model, num_epochs= n_epochs[1], lr = 0.00001,  g=g)
        total_time = time.time() - start_time
        print("--- %s seconds ---" % (total_time))
        times.append(total_time)

        #Save Mode
        torch.save(model.state_dict(), dir + dataset + 'iter_' + str(vers) + '.ptch')
        #train_log, eval_log = convert_loggers(train_logger, eval_logger)
        summary_plot(train_logger, eval_logger, dataset = dataset, savefig = True, dir = dir, vers = vers)
        pickle.dump(train_logger, open(dir + f"from_{source_dataset}/"+ dataset + '_train' + 'iter_' + str(vers) + '.pkl', "wb"))
        pickle.dump(eval_logger, open(dir + f"from_{source_dataset}/"+ dataset + '_eval' + 'iter_' + str(vers) + '.pkl', "wb"))

        del model
        gc.collect()


def main():
    seed = 6728131 
    datasets = ['MNIST', 'KMNIST', 'EMNIST', 'FASHION_MNIST', 'USPS']

    #Train "pretrained models"
    for dataset in datasets : 
        train(dataset = dataset,n_epochs= [10,20], transfer =  False, source_dataset= None)
    
    #Transfer between each pair of datasets
    for dataset_source, dataset_target in itertools.permutations(datasets, 2):
        train(dataset = dataset_target, n_epochs= [5],  transfer =  True, source_dataset= dataset_source)

