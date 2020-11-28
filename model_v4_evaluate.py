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
out_path = os.path.join(data_path, 'output_model_cross_mean_5ch_output_4')

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
#plt.imshow(out[10])
#plt.plot(out[100])
#out[0, 79, 185]

m4c.show_coordinates(ims[513], out[513], out_path, rad=5, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0)


###
#sample = {'image': ims, 'labels': out}
#print(len(out))
#print(out[:4])


data_set =  m4c.pig_dataset(ims, out, out_coor)


train_samples = int(len(data_set)*99/100)
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
device = m4c.useCuda(True)

dtype = torch.FloatTensor
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available(): #check for GPU
  dtype = torch.cuda.FloatTensor
  print("GPU available")
  torch.device("cuda")
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  



###
latent_space_dim = 128
capacity = 32
learning_rate = 1e-3
modelAE = m4c.AE(latent_dims=latent_space_dim, capacity=capacity).to(device)
optimizer = optim.Adam(modelAE.parameters(), lr=learning_rate)


###
modelAE.load_state_dict(torch.load(data_path+"/"+model_name+".pyc"), strict=False)


###
modelAE.eval()
cnt = 0
total_count_pred = 0
total_count_given = 0
with torch.no_grad():
    for i, (x, y, z) in enumerate(loader_test):
        y_hat, mu, logvar = modelAE(x.to(device))
        for bt in range(len(x)):
            #print(out_coor[int(z[bt].numpy())])
            given_pts = out_coor[int(z[bt].numpy())]
            #print(x[bt].squeeze(dim=0).numpy().shape, y[bt].numpy().shape)
            out_image = y_hat[bt].cpu().numpy()
            for ch in range(out_image.shape[0]):
                out_image[ch,out_image[ch]<0.001] = 0
                out_image[ch,out_image[ch]>0.001] = 1
            label, min_max_coor = m4c.analyseLines(out_image)
            
            #print(min_max_coor)
            count_pred, count_given = m4c.countMatch(min_max_coor, given_pts)
            total_count_pred = total_count_pred + count_pred
            total_count_given = total_count_given + count_given
            
            #show_coordinates(x[bt].squeeze(dim=0).numpy(), out_image, out_path, rad=5, 
            #                  threshold=0.001, save_fig=True, load_ind=cnt, bt_ind=bt)
            #show_coordinates2(x[bt].squeeze(dim=0).numpy(), out_image, out_path, label, min_max_coor, rad=5, 
                              #threshold=0.001, save_fig=True, load_ind=cnt, bt_ind=bt)
            #plt.plot(y_hat[bt].cpu().numpy()[0])
            cnt = cnt + 1
acc = (total_count_pred/total_count_given)*100
print(acc)