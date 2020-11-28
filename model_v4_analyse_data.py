#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 01:15:15 2020

@author: ppd
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import torch.utils.data
import torch.nn as nn

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torchvision.utils

import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2

from tracker import Tracker
import time
import imageio
import functools 
import pickle as pkl

import model_v4_common as m4c

model_name = "model_cross_mean_5ch_output_6all"

main_path = '/Home/ppd/works/'
# main_path = '<path>/<to>/<notebook>'
data_path = os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices')
out_path = os.path.join(data_path, 'output_'+model_name)

###
print(data_path)
#images_files_all  = [os.path.join(data_path, 'input_x', f) for f in os.scandir(os.path.join(data_path, 'input_x')) if f.path.endswith('.jpg') ]
#print ("Total Number of Images: {} (should be 1800)".format(len(images_files_all)))

#load dataset
ims = np.load(data_path+"/dataset_ims_for_"+model_name+".npy", allow_pickle=True)[()]
out = np.load(data_path+"/dataset_labels_for_"+model_name+".npy", allow_pickle=True)[()]
out_coor = np.load(data_path+"/dataset_label_coors_for_"+model_name+".npy", allow_pickle=True)[()]


###
#plt.imshow(np.reshape(out[100], (-1, int(img_tmp.shape[1]*scale_percent/100))))
out[638].shape
plt.imshow(ims[939])
plt.imshow(out[939][0])
plt.imshow(out[939][1])
plt.imshow(out[939][2])
plt.imshow(out[939][3])
plt.imshow(out[939][4])
plt.show()
#plt.plot(out[100])
#out[0, 79, 185]


m4c.show_coordinates(ims[939], out[939], out_path, rad=5, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0)


###
#sample = {'image': ims, 'labels': out}
#print(len(out))
#print(out[:4])

data_set =  m4c.pig_dataset(ims, out, out_coor)


train_samples = int(len(data_set)*80/100)
test_samples = len(data_set) - train_samples
print(train_samples, test_samples)

class ChunkSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

batch_size = 16
loader_train = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=ChunkSampler(train_samples, 0), num_workers=1)
loader_test = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=ChunkSampler(test_samples, train_samples), num_workers=1)
#tt = torch.utils.data.DataLoader(sample,batch_size=16,shuffle=True, num_workers=2)


###
print(len(data_set))
print('input shape: ', data_set[0][0].shape)
print('output shape: ', data_set[0][1].shape)

print(data_set[0][0].dtype)
print(data_set[0][1].dtype)
#for i_batch, (x, y, z) in enumerate(loader_train, 0):
#    print(i_batch, x, y, z)

###
def avg_shoulder_tail_length(loader, comment):
    j=0
    avg_S_T_length_vec = []
    with torch.no_grad():
        for i, (x, y, y_ind) in enumerate(loader):
            for k in range(len(y_ind)):
                given_pts = out_coor[int(y_ind[k].numpy())]
                lenght_S_T = 0 
                for pts in given_pts:
                    lenght_S_T = lenght_S_T + np.linalg.norm(pts[2]-pts[3])
                avg_lenght_S_T = int(round(lenght_S_T / len(given_pts)))
                avg_S_T_length_vec.append(avg_lenght_S_T)
                j = j+1
    print(j)
    n, bins, patches = plt.hist(x=avg_S_T_length_vec, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Avg. length')
    plt.ylabel('No. of images')
    plt.title('Average shoulder-tail length per image ('+comment+' set)')
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plt.savefig(out_path+"/"+comment+"_hist.png", dpi=150) 

avg_shoulder_tail_length(loader_train, "train")
avg_shoulder_tail_length(loader_test, "test")




