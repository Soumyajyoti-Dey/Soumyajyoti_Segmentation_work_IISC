#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:29:51 2024

@author: soumyajyoti
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:27:42 2024

@author: soumyajyoti
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms as t
from torchvision.utils import save_image as save
#from pixelLevelAccuracy import pix_lev_acc
import torch.optim as optim
import matplotlib
#import torch
import torch.nn as nn
import torch.nn.functional as F
CUDA_LAUNCH_BLOCKING=1
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
torch.cuda.set_device(2)
#CUDA_VISIBLE_DEVICES=0
from sklearn.metrics import jaccard_score
from sklearn.metrics import confusion_matrix

#####GLOBALS
from utils import DiceLoss
from networks.vision_transformer import SwinUnet as ViT_seg
from config import get_config
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

num_classes = 8
weight_load = None
# weight_save = 'PAP_segnet_trainval_best_loss'
# weight_save2 = 'PAP_segnet_trainval_best_acuu'
# weight_save3 = 'PAP_segnet_trainval_best_loss'                    
img_folder = "./"
loss_file = "./val.txt"

input_val = "/run/media/cmatergpu/new_hdd/iisc_work/data/wsi/img_dir/val/"
labels_val ="/run/media/cmatergpu/new_hdd/iisc_work/data/wsi/ann_dir/val/"
input_train = "/run/media/cmatergpu/new_hdd/iisc_work/data/wsi/img_dir/train/"
labels_train = "/run/media/cmatergpu/new_hdd/iisc_work/data/wsi/ann_dir/train/"

batch = 32
worker = 1
shuffle_train = True
shuffle_val = False
h,w = 224,224
epochs = 30000
ignore = -100
start = 0
no_channel = 3
lr = 0.001 #default lr = 0.001

img_size = 224

######MODEL
class getDataset(Dataset):
    def __init__(self, image_path, label_path, image_transform=None, label_transform=None, size=None, num_of_classes=5):
        self.images = sorted(os.listdir(image_path))
        self.labels = sorted(os.listdir(label_path))
        self.noc = num_of_classes-1
        assert(len(self.images)==len(self.labels)), 'The two folders do not have same number of images'
        for i,filename in enumerate(self.images):
            self.images[i] = image_path+'/'+self.images[i]
            self.labels[i] = label_path+'/'+self.labels[i]      
       
        if size==None:
            self.size=(h,w)
        else:
            self.size=size
       
        if image_transform==None:
            self.image_transform = t.Compose([t.Resize(self.size),t.ToTensor()])
        else:
            self.image_transform = image_transform
       
        if label_transform==None:
            self.label_transform = t.Compose([t.Resize(self.size,interpolation=Image.NEAREST)])
        else:
            self.label_transform = label_transform

    def __getitem__(self, index):
        image = self.image_transform(Image.open(self.images[index]))
        if image.size()[0]==1:
            print("1D image")
            image = torch.cat([image]*3)
        t=Image.open(self.labels[index])
        label = self.label_transform(t)
        label=np.array(label)
        s = label.shape
        if(len(s) == 3):
            if(s[2]==3):
                label = label[:,:,0]
        label=torch.from_numpy(label).long()
        return (image,label.squeeze())
   
    def __len__(self):
        return (len(self.images))

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


   
   
   
dataset=getDataset(input_train,labels_train,num_of_classes=num_classes)
dataloader = DataLoader(dataset, batch_size=batch,
                        shuffle=shuffle_train, num_workers=worker)


dataset2=getDataset(input_val,labels_val,num_of_classes=num_classes)
dataloader2 = DataLoader(dataset2, batch_size=batch,
                        shuffle=shuffle_val, num_workers=worker)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('#### Test Model ###')
#x = torch.rand(4, 3, 224, 224).to(device)
net = ViT_seg(config, img_size=img_size, num_classes=num_classes).cuda()
#net = torch.load("/run/media/cmatergpu/new_hdd/iisc_work/code/code/Swin-Unet-main/Swin_Unet.pt")

criterion = nn.CrossEntropyLoss()
dice_loss = DiceLoss(num_classes)
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)

if weight_load!= None:
    state = torch.load(weight_load ,map_location=lambda storage, loc: storage.cuda())      
    net.load_state_dict(state['state_dict'])
    if lr == 0.001:
        optimizer.load_state_dict(state['optimizer'])

train_loss_vs_epoch=[]
validation_loss_vs_epoch=[]


min_validation = 9999
for epoch in range(1,epochs):
    running_loss = 0.0
    net.train()
    print("Training for epoch", epoch + start, "Started :")
    for i,data in enumerate(dataloader,1):
       
        inputs1, labels1 = data
        inputs2=(inputs1.to(device))
        labels2=(labels1.to(device))
        optimizer.zero_grad()
        #print("hahaaa")
       
        outputs = net(inputs2)
        #print(outputs.shape)
        _,predicted = torch.max(outputs,1)
        loss = criterion(outputs, labels2)
        loss_dice = dice_loss(outputs, labels2, softmax=True)
        loss = 0.4 * loss + 0.6 * loss_dice
        
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
         
        img=torch.stack([predicted]*3,dim=1)
        lb =torch.stack([labels2]*3,dim=1)

        save((lb.float().data.cpu())/(num_classes-1),img_folder+"/lbt.jpg")
        save((img.float().data.cpu())/(num_classes -1),img_folder+"/opt.jpg")
       
    # save((lb.float().data.cpu())/(no_class -1),img_folder+"/result_segnet/lbt %d.jpg"%( epoch + start ))
    # save((img.float().data.cpu())/(no_class -1),img_folder+"/result_segnet/opt %d.jpg"%( epoch + start ))
    train_loss_vs_epoch.append(running_loss/len(dataloader))
   
   
   
    net.eval()
    val_loss=0
    P1 = 0.0
    # P2 = 0.0
    for i,data in enumerate(dataloader2,1):
       
        inputs1, labels1 = data
        inputs2 = (inputs1.to(device))
        labels2 = (labels1.to(device))
        
        outputs = net(inputs2)
        _,predicted = torch.max(outputs,1)
        loss = criterion(outputs, labels2)
        loss_dice = dice_loss(outputs, labels2, softmax=True)
        loss = 0.4 * loss + 0.6 * loss_dice
        val_loss += loss.item()

        #pix_acc1 = pix_lev_acc(predicted,labels2 )
        #P1+= pix_acc1
        #print(predicted.shape)
        #print(labels2.shape)


        predicted_=(predicted.cpu().numpy().flatten())
        labels2_=(labels2.cpu().numpy().flatten())
        c=confusion_matrix(labels2_, predicted_)
        print(c)
        acc_val = jaccard_score(labels2_, predicted_,average='micro')
        P1+= acc_val





        

        



       
        img = torch.stack([predicted]*3,dim=1)
        lb = torch.stack([labels2]*3,dim=1)
        save((lb.float().data.cpu())/(num_classes -1 ),img_folder+"/lbv.jpg")
        save((img.float().data.cpu())/(num_classes -1 ),img_folder+"/opv.jpg")

    validation_loss_vs_epoch.append(val_loss/len(dataloader2))  
    plt.plot(train_loss_vs_epoch,'r',validation_loss_vs_epoch,'b')
    plt.savefig('plot_swinUnet.png')

    if val_loss < min_validation :
        min_validation = val_loss
        torch.save(net,'./Swin_Unet.pt');
        print('model saved')
    print('[epoch: %d] train loss: %.6f, val loss= %0.6f' %(epoch+1, loss/len(dataloader), val_loss/len(dataloader2)))

    print("Val set IOU = ", P1/len(dataloader2))


print('Finished Training')
