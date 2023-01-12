# %% [markdown]
# # Complexity Computation

# %%
import os
import torch
import torchvision
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from references.utils import setup_for_distributed
from data_load import VOC_OI
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import time
import gc

#%% Seeding all for reproducibility

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True Reduces performance
    #torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# %% [markdown]
# ## 1. Models

# %%

def build_model(transformer = True, pretrained  = True, pretrained_backbone=False, trainable_backbone_layers = 0):

    if transformer:
        faster_model  =  torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained = pretrained, 
                                                                      pretrained_backbone=pretrained_backbone, 
                                                                      trainable_backbone_layers=trainable_backbone_layers)
    else :
        
        faster_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = pretrained, 
                                                                      pretrained_backbone=pretrained_backbone, 
                                                                      trainable_backbone_layers=trainable_backbone_layers)


  # get number of input features for the classifier
    in_features = faster_model.roi_heads.box_predictor.cls_score.in_features

  # replace the pre-trained head with a new one
    faster_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 6)

    return faster_model


# %% [markdown]
# ## 4. Training the model

from references.engine import train_one_epoch, evaluate
from references import utils


def train(gpu, model, dataset, mp_args, batch_size = 2, num_epochs = 30, lr = 0.0001, output_dir = None, vers = 0):

    rank = mp_args.nr * mp_args.gpus + gpu	#Rank of the process over all processes   

    setup_for_distributed(rank == 0)    # Disable print if rank not 0        

    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=mp_args.world_size,                              
    	rank=rank                                               
    )      

    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}') 

    indices = torch.randperm(len(dataset)).tolist()    
    dataset_train = torch.utils.data.Subset(dataset, indices[:1000])
    dataset_test = torch.utils.data.Subset(dataset, indices[1000:1200])

    #Initialize distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset_train,
    	num_replicas=mp_args.world_size,
    	rank=rank,
        shuffle = True
    )

    test_sampler = torch.utils.data.distributed.DistributedSampler(
    	dataset_test,
    	num_replicas=mp_args.world_size,
    	rank=rank,
        shuffle = False
    )
    
    # define training and validation data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size= batch_size, shuffle= False, num_workers=4, sampler = train_sampler,
        collate_fn=utils.collate_fn, worker_init_fn = seed_worker)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=  batch_size, shuffle=False, num_workers=4, sampler = test_sampler,
        collate_fn=utils.collate_fn, worker_init_fn = seed_worker)

    # move model to the right device
    model.to(device)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    model_without_ddp = model.module

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.Adam(params, lr=lr)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, 
                                                             patience=10, threshold=0.0001, threshold_mode='rel', 
                                                             cooldown=0, min_lr=0, eps=1e-08, verbose=False)

    # Prepare the list containing the train and eval logs
    avg_loss, avg_loss_classifier, avg_loss_box_reg, avg_loss_objectness, avg_loss_rpn_reg, iter_loss  = ([] for i in range(6))
    mAP_05_95, mAP_50, AR_1, AR_10, stats_list = ([] for i in range(5))

    if rank == 0 : #Create output dir for version i
        output_dir = output_dir + f'checkpoints_vers{vers}/'
        os.mkdir(output_dir)

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_log = train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=100)


        checkpoint = {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "epoch": epoch,
        }
        utils.save_on_master(checkpoint, os.path.join(output_dir, f"model_{epoch}.pth"))
  
    
        # evaluate on the test dataset
        with torch.no_grad():
            coco_evaluator = evaluate(model, data_loader_test, device=device)
            lr_scheduler.step(coco_evaluator.coco_eval['bbox'].stats[0]) #LR schedule on mAP all IoUs

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

    #Save logs only for master process
    if rank == 0 :
        train_log = {'avg_loss' : avg_loss, 'avg_loss_classifier' : avg_loss_classifier, 'avg_loss_box_reg' : avg_loss_box_reg,
                        'avg_loss_objectness' : avg_loss_objectness, 'avg_loss_rpn_reg' : avg_loss_rpn_reg,
                        'iter_loss' :  iter_loss}
        eval_log = {"mAP_05_95" : mAP_05_95, 'mAP_50' : mAP_50, 'AR_1' : AR_1, 'AR_10' : AR_10, 'stats' : stats_list}

        print("That's it!")
        pickle.dump(train_log, open(output_dir + 'dataset_1' + '_train' +  '.pkl', "wb"))
        pickle.dump(eval_log, open(output_dir + 'dataset_1' + '_eval' + '.pkl', "wb"))

        written_params = {'batch_size' : batch_size, 'num_epochs' : num_epochs, 'lr' : lr}
        summary_plot(train_log, eval_log, dataset = 'dataset_1', savefig = True, dir = output_dir, written_params = written_params)



# %%
def summary_plot(train_log, eval_log, dataset = None, savefig = False, dir = None, written_params = None):
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
        fig.suptitle('Metrics Summary for Resnet on ' + dataset + 'with params :' + str(written_params))
    if savefig:
        plt.savefig(dir + dataset +'.png')

#%%

class multi_processing_args():
    def __init__(self, n, g, nr) -> None:
        self.nodes = n
        self.gpus = g
        self.nr = nr



def main():
    seed = 6728131 
    seed_all(seed)

    mp_args = multi_processing_args(1, 2, 0) #1 node and 2 GPUs

    data_dir = '/data.nfs/AUTO_TL_OD/Export/'
 
    
    #########################################################
    mp_args.world_size = mp_args.gpus * mp_args.nodes       #
    os.environ['MASTER_ADDR'] = 'localhost'                 # Can add real IP adress of process 0
    os.environ['MASTER_PORT'] = '21212'                     #
    #########################################################

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

    for i in range(0,21):
    # Load model and dataset
        print('Build model for dataset : ', i+1 )
        model = build_model(transformer= True, pretrained= True, trainable_backbone_layers= 0)
        dataset = VOC_OI(data_dir + f'dataset_{i +  1}/', classes_list[i], size = (800, 800))
        output_dir = f'/path/to/model/ft_model_from_oi/dataset_{i+1}/'

        train_args = (model, dataset, mp_args, 3, 30, 0.0001, output_dir, 20)

        start_time = time.time()
        mp.spawn(train, nprocs=mp_args.gpus, args= train_args)   
        total_time = time.time() - start_time
        print(total_time)


if __name__ == '__main__':
    main()


