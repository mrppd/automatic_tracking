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

#from tracker import Tracker
import time
import imageio
import functools 
import pickle as pkl
import threading as th
import sys
# model_v4_common contains all the common functions for preprocessing data 
# and creating model. It might not be needed if you use your own back-end script.
import model_v4_common as m4c


terminationFlag = 0
def modelCall(sendStr):
    global terminationFlag

    # This parser function should be used in your script. This is designed to take parameters from TrainWindow GUI. 
    def argParser(argStr):
        tmpStr = argStr.split("|")
        tmpMetaList = tmpStr[0].strip().split("<")
        modelName = tmpStr[1].strip()
        epochStr = tmpStr[2].strip()
        trainValSplit = tmpStr[3].strip()
        batchSize = tmpStr[4].strip()
        isLoadModel = tmpStr[5].strip()
    
        metaList = []
        for x in tmpMetaList:
            tmpMetaSingle = x.split(">")
            if(len(tmpMetaSingle)==2):
                metaList.append((tmpMetaSingle[0].strip(), tmpMetaSingle[1].strip()))
        
        #print(metaList)
        #print(epochStr)
        return metaList, modelName, int(epochStr), int(trainValSplit), int(batchSize), int(isLoadModel)
    #strArg = "/media/sf_D_DRIVE/works/Educational info/Gottingen/Thesis/dataset/cam 5 ATW20181113-190716-1542132436_highAnnotation.csv > /media/sf_D_DRIVE/works/Educational info/Gottingen/Thesis/dataset/cam 5 ATW20181113-190716-1542132436_high < /media/sf_D_DRIVE/works/Educational info/Gottingen/Thesis/dataset/cam 5 ATW20181113-190716-1542132436_lessAnnotation.csv > /media/sf_D_DRIVE/works/Educational info/Gottingen/Thesis/dataset/cam 5 ATW20181113-190716-1542132436_less < /media/sf_D_DRIVE/works/Educational info/Gottingen/Thesis/dataset/cam 1820181201-204202-1543693322_highAnnotation.csv > /media/sf_D_DRIVE/works/Educational info/Gottingen/Thesis/dataset/cam 1820181201-204202-1543693322_high <  | /media/sf_D_DRIVE/test.pyc | 3 | 80 | 16 | 0"
    #strArg = "/Home/ppd/works/Pig_Videos/Kamera-120170504-120223-1493892143_1slices/cam 5 ATW20181113-190716-1542132436_highAnnotation.csv
    #strArg = "/Home/ppd/works/Pig_Videos/Kamera-120170504-120223-1493892143_1slices/cam 5 ATW20181113-190716-1542132436_highAnnotation.csv > /Home/ppd/works/Pig_Videos/Kamera-120170504-120223-1493892143_1slices/cam 5 ATW20181113-190716-1542132436_high <  | /Home/ppd/works/Pig_Videos/Kamera-120170504-120223-1493892143_1slices/testModel.pyc | 5 | 80 | 16 | 0"
    metaList, modelName, epoch, trainValSplit, batchSize, isLoadModel = argParser(sendStr)

    # if(len(sys.argv)==2):    
    #     metaList, modelName, epoch, trainValSplit, batchSize, isLoadModel = argParser(sys.argv[1])
    # else:
    #     exit()

    print(metaList)
    print(modelName)
    print('Epoch: ', epoch)
    print('Train/Validation Split: ', trainValSplit)
    print('Batch Size: ', batchSize)
    print('Is model loaded: ', isLoadModel)    

    #QApplication.processEvents()
    def secToStr(t):
        return ("%dh, %02dm, %02d.%03ds" % functools.reduce(
            lambda ll,b : divmod(ll[0],b) + ll[1:], [(t*1000,),1000,60,60]))

    useCudaI = True
    write = False
    read = False
    ##model_name = "model_cross_mean_5ch_output_6all"
    ##out_path = os.path.join(data_path, 'output_'+model_name)

    ###
    #images_files_all  = [os.path.join(data_path, 'input_x', f) for f in os.scandir(os.path.join(data_path, 'input_x')) if f.path.endswith('.jpg') ]
    #print ("Total Number of Images: {} (should be 1800)".format(len(images_files_all)))

    ###
    scale_percent = 50 # percent of original size
    total_sample = 500 #not in use
    t_height = 400
    t_width = 640


    #images_file
    ###s_all

    metaDataList = []
    for metaDataPath, dataPath in metaList:
        metaDataTmp = pd.read_csv(metaDataPath)
        metaDataTmp['providedFolderName'] = dataPath
        metaDataList.append(metaDataTmp)

    metadata = metaDataList
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
    metadata = metadata.drop_duplicates()
    metadata['baseFolderName_fileName'] = metadata['baseFolderName'] + "_" + metadata['fileName']

    metadata.baseFolderName_fileName = metadata.baseFolderName_fileName.apply(str)
    unique_file_names_all = metadata.baseFolderName_fileName.unique()

    metadata_sample = metadata.groupby("baseFolderName_fileName")
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
        image_path = metadata_sample.get_group(unique_file_names[i]).iloc[0].providedFolderName+"/"+metadata_sample.get_group(unique_file_names[i]).iloc[0].fileName
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
        if(terminationFlag==1):
            return

    #plt.imshow(ims[451])
    #print(ims[2])

    ###
    def onLine(P, P1, P2):
        x, y = P[0], P[1]
        x1, y1 = P1[0], P1[1]
        x2, y2 = P2[0], P2[1]
        res = ((x-x1)*(y1-y2))-((y-y1)*(x1-x2))
        #print(res)
        if(abs(res) <= 0.0001):
           return True
        return False

    def drawLine(origin, endpoint,surface, varb):
        deltaX = endpoint[0] - origin[0]
        deltaY = endpoint[1] - origin[1]
        try:
            deltaMax = max(max(abs(deltaX), abs(deltaY)), 1)
            deltaX = deltaX/deltaMax
            deltaY = deltaY/deltaMax
            for n in range(deltaMax):
                surface[int(origin[0]), int(origin[1])]=255
                #print(int(origin[0]), int(origin[1])
                origin[0] = origin[0] + deltaX;
                origin[1] = origin[1] + deltaY;
        except:
            print(origin, endpoint, varb)
        return surface
   
    
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

            out_img[4]=drawLine([p3SY, p3SX], [p4SY, p4SX], out_img[4], i)
    
            out_coor_tmp.append((p1SY, p1SX))
            out_coor_tmp.append((p2SY, p2SX))
            out_coor_tmp.append((p3SY, p3SX))
            out_coor_tmp.append((p4SY, p4SX))
            out_coor_gr.append(np.array(out_coor_tmp))
    
            #out_img[54, 481] = 255
            #print(out_img[int(gr.iloc[j].p3SY), int(gr.iloc[j].p3SX)])
    
        out_coor.append(np.array(out_coor_gr))  
  
        out[i] = out_img/255
        out[i][0] = cv2.GaussianBlur(out[i][0], (5,5), 1.6)
        out[i][1] = cv2.GaussianBlur(out[i][1], (5,5), 1.1)
        out[i][2] = cv2.GaussianBlur(out[i][2], (5,5), 2.1)
        out[i][3] = cv2.GaussianBlur(out[i][3], (5,5), 2.1)
        out[i][4] = cv2.GaussianBlur(out[i][4], (5,5), 3.1)
        if(terminationFlag==1):
            return
  
  

    # out[65].shape
    # plt.imshow(ims[638])
    # plt.imshow(out[638][0])
    # plt.imshow(out[638][1])
    # plt.imshow(out[638][2])
    # plt.imshow(out[638][3])
    # plt.imshow(out[638][4])
    # plt.show()

    #print(out_img)
    #print(int(gr.iloc[0].p3SY))
    #print(int(gr.iloc[0].p3SX))


    ###
    #Save dataset
    # =============================================================================
    # if(write==True):
    #     np.save(data_path+"/dataset_ims_for_"+model_name+".npy", ims)
    #     np.save(data_path+"/dataset_labels_for_"+model_name+".npy", out)
    #     np.save(data_path+"/dataset_label_coors_for_"+model_name+".npy", out_coor)
    # #load dataset
    # if(read==True):
    #     ims = np.load(data_path+"/dataset_ims_for_"+model_name+".npy", allow_pickle=True)[()]
    #     out = np.load(data_path+"/dataset_labels_for_"+model_name+".npy", allow_pickle=True)[()]
    #     out_coor = np.load(data_path+"/dataset_label_coors_for_"+model_name+".npy", allow_pickle=True)[()]
    # =============================================================================
    data_set =  m4c.pig_dataset(ims, out, out_coor)

    train_samples = int(len(data_set)*trainValSplit/100)
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

    batch_size = batchSize
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
    device = m4c.useCuda(useCudaI)

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

    if(terminationFlag==1):
        return
    if(isLoadModel==1):
        modelAE.load_state_dict(torch.load(modelName), strict=False)

    ###
    def train_and_evaluate_model(device, model, opt, cap, loader_train, train_samples, loader_test, test_samples, num_epochs=10):
        global terminationFlag
        #criterion = nn.MSELoss()
        #optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.99))

        trainL, testL, epochs = [], [], []
        for e in range(1,num_epochs+1):
          
            train_loss = m4c.train(model, opt, loader_train, device)
            train_loss /= train_samples
            test_loss = m4c.test(model, opt, loader_test, device)
            test_loss /= test_samples
            trainL.append(train_loss)
            testL.append(test_loss)
            epochs.append(e)
            #print("epoch: "+str(e))

            print(f'Epoch {e}, Train Loss: {train_loss:.8f}, Validation Loss: {test_loss:.8f}')
            if(terminationFlag==1):
                break

        # sns.lineplot(x=epochs,y=trainL, label='Train')
        # sns.lineplot(x=epochs,y=testL, label='Validation')
        # plt.legend()
        # plt.show()
        return model


    start_time = time.time()
    modelAE = train_and_evaluate_model(device, model=modelAE, opt=optimizer, cap=capacity, 
                                           loader_train=loader_train, train_samples=train_samples, 
                                           loader_test=loader_test, test_samples=test_samples, 
                                           num_epochs=epoch)
    modelAE
    if(terminationFlag==1):
        return
    print("Total time:", m4c.secToStr(time.time()-start_time), "\n")   
    print('Saving the model. Please wait!...')
    torch.save(modelAE.state_dict(), modelName)
    print('Model Saved!!!')
