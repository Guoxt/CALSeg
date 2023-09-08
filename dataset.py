#!/usr/bin/env python
#coding:utf8
import nibabel as nib
import os
#import matplotlib.pyplot as plt
import numpy as np
import random
#from torchvision import  transforms as T
from PIL import Image
from torch.utils import data
import torch
#from sklearn.preprocessing import LabelEncoder
#from sklearn.utils import shuffle

###
class Brats17(data.Dataset):
    def __init__(self, root, transforms = None, train = True, test = False, val = False):
        self.test = test
        self.train = train
        self.val = val

        if self.train:
            self.root = 'userhome/GUOXUTAO/DATA_ALL/WMH_liangli/WMH/userhome/ye/DATA_ALL/WMH/2D_01/train/data/'
            self.folderlist = os.listdir(self.root)
            #self.folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/jidayi_naobaizi/filename/data/1.npy')))
            #self.folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/jidayi_naobaizi/filename/data/2.npy')))
            #self.folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/jidayi_naobaizi/filename/data/3.npy')))
        elif self.val:
            self.root = '/userhome/ye/DATA_ALL/sag_cuiti_80/npy/data/'
            #self.folderlist = list(np.load('/userhome/ye/DATA_ALL/sag_cuiti_80/filename/data/1.npy'))
            #self.folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/sag_cuiti_80/filename/data/2.npy')))
            #self.folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/sag_cuiti_80/filename/data/3.npy')))
            #self.folderlist.extend(list(np.load('/userhome/ye/DATA_ALL/sag_cuiti_80/filename/data/4.npy')))
        elif self.test:
            self.root = ''
            self.folderlist = os.listdir(os.path.join(self.root))

    def __getitem__(self,index):
          
        if self.train:                            
            if 1 > 0 :
                ss = 64
                sss = 64
                #print(self.folderlist[index])
                path = self.root
                img = np.load(os.path.join(path,self.folderlist[index]))
                img = np.asarray(img)
                #print(img.shape)
                index_x = np.random.randint(ss,img.shape[1]-ss,size=1)
                index_y = np.random.randint(sss,img.shape[2]-sss,size=1)
                #index_z = np.random.randint(sss,img.shape[3]-sss,size=1)
                #print(index_x,index_y,index_z)
                
                #if random.random() < 0.8:
                #    while np.sum(img[2,index_x[0]-ss:index_x[0]+ss,index_y[0]-sss:index_y[0]+sss,index_z[0]-sss:index_z[0]+sss]==1.0) < 1:
                #        index_x = np.random.randint(ss,img.shape[1]-ss,size=1)
                #        index_y = np.random.randint(sss,img.shape[2]-sss,size=1)
                #        index_z = np.random.randint(sss,img.shape[3]-sss,size=1)
                
                img_in = img[:,index_x[0]-ss:index_x[0]+ss,index_y[0]-sss:index_y[0]+sss]          
                img_out = img_in[0:2,:,:].astype(float)
                label_out = img_in[2:3,:,:].astype(float)
                #print(img_in.shape)
                img = torch.from_numpy(img_out).float()        
                label = torch.from_numpy(label_out).long() 
                #print(label_out.max(),label_out.min())
                
        elif self.val:
            path = self.root
            img = np.load(os.path.join(path,self.folderlist[index]))
            img = np.asarray(img)
            img_out = img[0,:,:,:].astype(float)
            label_out = img[1,:,:,:].astype(float)
            #print(img.shape)
            img = torch.from_numpy(img_out).unsqueeze(0).float()     
            label = torch.from_numpy(label_out).long()
        else:
            print('###$$$$$$$$$$$$$$$$$$$^^^^^^^^^^^^^')     

        return img, label

    def __len__(self):
        return len(self.folderlist)







