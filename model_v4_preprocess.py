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


main_path = '/Home/ppd/works/'
# main_path = '<path>/<to>/<notebook>'
data_path = os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices')
out_path = os.path.join(data_path, 'output_model_cross_mean_5ch_output_4')

###
print(data_path)
#images_files_all  = [os.path.join(data_path, 'input_x', f) for f in os.scandir(os.path.join(data_path, 'input_x')) if f.path.endswith('.jpg') ]
#print ("Total Number of Images: {} (should be 1800)".format(len(images_files_all)))

###
scale_percent = 50 # percent of original size
total_sample = 500 #not in use
t_height = 400
t_width = 640

#images_file
###s_all
metadata1 = pd.read_csv(data_path+"/cam 5 ATW20181113-190716-1542132436_highAnnotation.csv")
#metadata2 = pd.read_csv(data_path+"/input_xAnnotation.csv")
metadata2 = pd.read_csv(data_path+"/cam 5 ATW20181113-190716-1542132436_lessAnnotation.csv")

metadata = [metadata1, metadata2]
metadata = pd.concat(metadata).reset_index(drop=True)

metadata = metadata[(metadata.p1 != "-1,-1") & (metadata.p2 != "-1,-1" ) & (metadata.p3 != "-1,-1" ) & (metadata.p4 != "-1,-1" )]
metadata['p1SX'], metadata['p1SY'] = metadata.p1.str.split(',', 1).str
metadata['p2SX'], metadata['p2SY'] = metadata.p2.str.split(',', 1).str
metadata['p3SX'], metadata['p3SY'] = metadata.p3.str.split(',', 1).str
metadata['p4SX'], metadata['p4SY'] = metadata.p4.str.split(',', 1).str

metadata['p1SX'] = pd.to_numeric(metadata['p1SX']) #* scale_percent / 100
metadata['p1SY'] = pd.to_numeric(metadata['p1SY']) #* scale_percent / 100
metadata['p2SX'] = pd.to_numeric(metadata['p2SX']) #* scale_percent / 100
metadata['p2SY'] = pd.to_numeric(metadata['p2SY']) #* scale_percent / 100
metadata['p3SX'] = pd.to_numeric(metadata['p3SX']) #* scale_percent / 100
metadata['p3SY'] = pd.to_numeric(metadata['p3SY']) #* scale_percent / 100
metadata['p4SX'] = pd.to_numeric(metadata['p4SX']) #* scale_percent / 100
metadata['p4SY'] = pd.to_numeric(metadata['p4SY']) #* scale_percent / 100
metadata

metadata.fileName = metadata.fileName.apply(str)
unique_file_names_all = metadata.fileName.unique()

metadata_sample = metadata.groupby("fileName")
#metadata_sample.head()
#metadata_sample.get_group(unique_file_names[0])

#metadata_sample = metadata.sample(500).reset_index()
#print(metadata_sample)
unique_file_names_prune = []
for ui in unique_file_names_all:
    if len(metadata_sample.get_group(ui))>=1:
        unique_file_names_prune.append(ui)

print("unique names prune: " + str(len(unique_file_names_prune)) )
unique_file_names = np.array(unique_file_names_prune)
total_sample = len(unique_file_names_prune) #sample rectification due to small dataset
# Random sampling
rand_ind = np.random.randint(0, len(unique_file_names), total_sample)
unique_file_names = unique_file_names[rand_ind]
print("unique names sample: " + str(len(unique_file_names)) )

###
metadata_sample.get_group(unique_file_names[10])


###
#image_path = data_path+"/"+metadata_sample.get_group(unique_file_names[1]).iloc[0].baseFolderName+"/"+metadata_sample.get_group(unique_file_names[1]).iloc[0].fileName
#img_tmp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#print("Original shape: ",img_tmp.shape)
#print("Reduced shape: ",(int(img_tmp.shape[0]*scale_percent/100), int(img_tmp.shape[1]*scale_percent/100)))


ims = np.empty((total_sample, t_height, t_width), dtype=np.float32)
orig_ims_shapes = []
for i in range(total_sample):
  print(i, end=" ")
  if i%30==0: print() 
  image_path = data_path+"/"+metadata_sample.get_group(unique_file_names[i]).iloc[0].baseFolderName+"/"+metadata_sample.get_group(unique_file_names[i]).iloc[0].fileName
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  #print('Original Dimensions : ',img.shape)
  orig_ims_shapes.append(img.shape)
  
  width = t_width
  height = t_height
  dim = (width, height)
  # resize image
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
  ims[i] = resized/255.0
  #print('Resized Dimensions : ',resized.shape) 
  #plt.imshow(ims[i])
  #plt.show()

plt.imshow(ims[513])
#print(ims[2])

    
out = np.empty((total_sample, 5, t_height, t_width), dtype=np.float32)
out_coor = []

for i in range(total_sample):
  out_img = np.zeros((5, t_height, t_width), dtype=np.float32)
  gr = metadata_sample.get_group(unique_file_names[i])
  out_coor_gr = []
  for j in range(len(gr)):
    #print(int(gr.iloc[j].p3SY), int(gr.iloc[j].p3SX))
    out_coor_tmp = []
    adjustment_height = t_height/orig_ims_shapes[i][0]
    adjustment_width = t_width/orig_ims_shapes[i][1]
    p1SY = int(gr.iloc[j].p1SY * adjustment_height)
    p1SX = int(gr.iloc[j].p1SX * adjustment_width)
    p2SY = int(gr.iloc[j].p2SY * adjustment_height)
    p2SX = int(gr.iloc[j].p2SX * adjustment_width)
    p3SY = int(gr.iloc[j].p3SY * adjustment_height)
    p3SX = int(gr.iloc[j].p3SX * adjustment_width)
    p4SY = int(gr.iloc[j].p4SY * adjustment_height)
    p4SX = int(gr.iloc[j].p4SX * adjustment_width)
    
    out_img[0, p1SY, p1SX] = 255
    out_img[1, p2SY, p2SX] = 255
    out_img[2, p3SY, p3SX] = 255
    out_img[3, p4SY, p4SX] = 255

    """
    x_inds = np.arange(min(int(gr.iloc[j].p3SY),  int(gr.iloc[j].p4SY)), 
                       max(int(gr.iloc[j].p3SY),  int(gr.iloc[j].p4SY)), 1).tolist()
    y_inds = np.arange(min(int(gr.iloc[j].p3SX),  int(gr.iloc[j].p4SX)), 
                       max(int(gr.iloc[j].p3SX),  int(gr.iloc[j].p4SX)), 1).tolist()
    """
    #print(len(x_inds), len(y_inds))

    out_img[4]=m4c.drawLine([p3SY, p3SX], [p4SY, p4SX], out_img[4])
    
    out_coor_tmp.append((p1SY, p1SX))
    out_coor_tmp.append((p2SY, p2SX))
    out_coor_tmp.append((p3SY, p3SX))
    out_coor_tmp.append((p4SY, p4SX))
    out_coor_gr.append(np.array(out_coor_tmp))
    
    #out_img[54, 481] = 255
    #print(out_img[int(gr.iloc[j].p3SY), int(gr.iloc[j].p3SX)])
  out_coor.append(np.array(out_coor_gr))  
  out[i] = out_img/255
  out[i][0] = cv2.GaussianBlur(out[i][0], (5,5), 1.5)
  out[i][1] = cv2.GaussianBlur(out[i][1], (5,5), 1.5)
  out[i][2] = cv2.GaussianBlur(out[i][2], (5,5), 1.5)
  out[i][3] = cv2.GaussianBlur(out[i][3], (5,5), 1.5)
  out[i][4] = cv2.GaussianBlur(out[i][4], (5,5), 1.5)
  
  

out[513].shape
plt.imshow(out[513][2])
plt.imshow(out[513][4])
plt.show()

#print(out_img)
#print(int(gr.iloc[0].p3SY))
#print(int(gr.iloc[0].p3SX))


###
#Save dataset
#np.save(data_path+"/dataset_ims_for_model_cross_mean_5ch_output_6all.npy", ims)
#np.save(data_path+"/dataset_labels_for_model_cross_mean_5ch_output_6all.npy", out)
#np.save(data_path+"/dataset_label_coors_for_model_cross_mean_5ch_output_6all.npy", out_coor)