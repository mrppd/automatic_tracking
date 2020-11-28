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



def isVanished(track, blength=6):
    current = track.trace[-1][0]
    if((len(track.trace)-1)<blength):
        blength = len(track.trace)-1
    for i in range(2, 2+blength):
        dist =  np.linalg.norm(track.trace[-i][0]-current)
        if(dist>0.001):
            return False
    return True

#print(isVanished(tracker.tracks[j], 6))

def socialInteraction2(frame, track, track_ind, shoulders, tails, frame_no):
    current_pred_point = track[track_ind].trace[-1][0]
    if(isVanished(track[track_ind])):
        return frame
    
    print(current_pred_point)
    print(min_max_coor)
    print(shoulders)
    print(tails)
    
    min_dist = 999999999
    ind = 0
    for i in range(len(shoulders)):
        dist =  np.linalg.norm(shoulders[i]-current_pred_point)
        if(dist<min_dist):
            min_dist = dist
            ind = i
    print(shoulders[ind], tails[ind])
    radius = (np.linalg.norm(shoulders[ind]-tails[ind])/2)
    
    print("Distance threshold: ", radius)
    
    for j in range(len(track)):
        if(j==track_ind or len(track[j].trace)<=1):
            continue
        if(isVanished(track[j])):
            continue
        other_pred_point = track[j].trace[-1][0]
        min_dist2 = 999999999
        ind2 = 0
        for i in range(len(shoulders)):
            dist2 =  np.linalg.norm(shoulders[i]-other_pred_point)
            if(ind==i):
                continue
            if(dist2<min_dist2):
                min_dist2 = dist2
                ind2 = i
        print(shoulders[ind2], tails[ind2])
        radius2 = (np.linalg.norm(shoulders[ind2]-tails[ind2])/2)
        print("Distance threshold2: ", radius2)
        
        if(m4c.circleIntersect(x1=shoulders[ind][0], y1=shoulders[ind][1], 
                           x2=shoulders[ind2][0], y2=shoulders[ind2][1], 
                           r1=radius, r2=radius2)>0):
            dict_interaction = {"frame_no":frame_no, "pig1_id":track[track_ind].trackId, 
                                "pig2_id":track[j].trackId, "interaction":2}
            rows_list_interaction.append(dict_interaction)
            cv2.circle(frame, (shoulders[ind][0], shoulders[ind][1]), int(round(radius,0)), (255, 0, 255), 2)
            cv2.circle(frame, (shoulders[ind2][0], shoulders[ind2][1]), int(round(radius2,0)), (255, 0, 255), 2)
            
        if(m4c.circleIntersect(x1=shoulders[ind][0], y1=shoulders[ind][1], 
                           x2=tails[ind2][0], y2=tails[ind2][1], 
                           r1=radius, r2=radius2)>0):
            dict_interaction = {"frame_no":frame_no, "pig1_id":track[track_ind].trackId, 
                                "pig2_id":track[j].trackId, "interaction":3}
            rows_list_interaction.append(dict_interaction)
            cv2.circle(frame, (shoulders[ind][0], shoulders[ind][1]), int(round(radius,0)), (0, 0, 255), 2)
            cv2.circle(frame, (tails[ind2][0], tails[ind2][1]), int(round(radius2,0)), (255, 255, 0), 2)
    return frame
            
            
                

#analysis from video
RUN_TILL = 1200
rows_list_interaction = []  
def createimage(w,h):
    size = (w, h, 1)
    img = np.ones((w,h,3),np.uint8)*255
    return img
CODE_NAME = "piglet1_slow11_3fps"
VIDEO_IN_PATH = os.path.join(main_path, 'Pig_Videos/Video_SocialInteraction/cam 5 ATW20181113-190716-1542132436.mp4')
VIDEO_OUT_PATH = os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices/track_'+CODE_NAME+'.mp4')

cap = cv2.VideoCapture(VIDEO_IN_PATH)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH/2);
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT/2);
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames: ", length)
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps)

frame_per_sec = 3
gap = int(round(fps/frame_per_sec))
current_frame_no = 0

video_out = cv2.VideoWriter(VIDEO_OUT_PATH, cv2.VideoWriter_fourcc(*"MJPG"), frame_per_sec, 
                            (1280, 400))
images = []
tracker = Tracker(30, 10, 10)
skip_frame_count = 0
track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127),
                
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (127, 127, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]

tmp_data = []
while(True):
    #current_frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    if(current_frame_no>RUN_TILL):
        break
    
    cap.set(1, current_frame_no-1)  
    ret, frame = cap.read()
    if not ret:
        break            
    
    #if(current_frame_no%5!=0):
    #    continue
    print("Current frame ", current_frame_no)
    current_frame_no = current_frame_no + gap
    
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    resizedGrayFrame = cv2.resize(grayFrame, (640, 400), 
                                  interpolation = cv2.INTER_AREA)
    
    input_image = torch.from_numpy((resizedGrayFrame/255.0).astype('f')).unsqueeze_(0).unsqueeze_(0)

    with torch.no_grad():
        y_hat, mu, logvar = modelAE(input_image.to(device))
        
        out_image = y_hat[0].cpu().numpy()
        for ch in range(out_image.shape[0]):
            out_image[ch,out_image[ch]<0.001] = 0
            out_image[ch,out_image[ch]>0.001] = 1
        label, min_max_coor = m4c.analyseLines(out_image)
        
    
        #centers = np.array([(min_max_coor[lind][(label[lind]-2)]).tolist() for lind in range(len(min_max_coor))])
        centers = np.array([[min_max_coor[lind][(label[lind]-2)][1], min_max_coor[lind][(label[lind]-2)][0]] for lind in range(len(min_max_coor))])
        tails = np.array([[min_max_coor[lind][(label[lind]-3)][1], min_max_coor[lind][(label[lind]-3)][0]] for lind in range(len(min_max_coor))])
        
        
        frame = createimage(resizedGrayFrame.shape[0], resizedGrayFrame.shape[1])
        if (len(centers) > 0):
            tracker.update(centers)
            for j in range(len(tracker.tracks)):
                if (len(tracker.tracks[j].trace) > 1):
                    x = int(tracker.tracks[j].trace[-1][0,0])
                    y = int(tracker.tracks[j].trace[-1][0,1])
                    tl = (x-10,y-10)
                    br = (x+10,y+10)
                    if(not isVanished(tracker.tracks[j])):
                        cv2.rectangle(frame,tl,br,track_colors[j],1)
                        cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
                        #print("trace length: ", len(tracker.tracks[j].trace))
                        for k in range(len(tracker.tracks[j].trace)):
                            x = int(tracker.tracks[j].trace[k][0,0])
                            y = int(tracker.tracks[j].trace[k][0,1])
                            #draw traces of current prediction
                            cv2.circle(frame,(x,y), 3, track_colors[j],-1)
                        #draw current predictions
                        cv2.circle(frame,(x,y), 6, track_colors[j],-1)
                    frame = socialInteraction2(frame, tracker.tracks, j, centers, tails, current_frame_no)
                #draw original points
                #cv2.circle(frame,(int(data[j,i,0]),int(data[j,i,1])), 6, (0,0,0),-1)
            #cv2.imshow('image',frame)
            # cv2.imwrite("image"+str(i)+".jpg", frame)
            # images.append(imageio.imread("image"+str(i)+".jpg"))
            #time.sleep(0.1)
            
        #print(y_hat.shape)
        #show_coordinates(imm[0].squeeze(dim=0).numpy(), y_hat[0].cpu().numpy(), rad=7, threshold=0.001)
    
        outImage = m4c.show_coordinates2(input_image.squeeze(dim=0).squeeze(dim=0).numpy(), out_image, 
                                     out_path, label, min_max_coor, rad=5, threshold=0.001, 
                                     save_fig=False, ret_cv=True)
        
        resFrame = cv2.hconcat([outImage, frame])
        #resFrame = (resFrame*255).astype(np.uint8)
        
        video_out.write(resFrame)
        # Display the resulting frame 
        #outImage2 = cv2.cvtColor(outImage.get().astype('f'), cv2.COLOR_BGR2RGB)
        ##cv2.imshow('Frame',  resFrame) 
        
    if cv2.waitKey(30) & 0xFF == ord('q'):
        cv2.destroyAllWindows() 
        break
cap.release()
video_out.release()
df_interaction = pd.DataFrame(rows_list_interaction)
df_interaction['time_sec'] = round(df_interaction['frame_no']/fps, 5)
df_interaction.to_csv(os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices/df_interaction_'+CODE_NAME+'.csv'))

