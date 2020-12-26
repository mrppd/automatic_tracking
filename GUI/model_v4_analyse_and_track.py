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

import threading as th
from tracker import Tracker
import time
import imageio
import functools 
import pickle as pkl
import threading as th

import model_v4_common as m4c

strArg = "/Home/ppd/works/Pig_Videos/Kamera-120170504-120223-1493892143_1slices/model_cross_mean_5ch_output_6all.pyc | /Home/ppd/works/Pig_Videos/Video_SocialInteraction/cam 5 ATW20181113-190716-1542132436.mp4 | /Home/ppd/works | 0 | 10 | 6.22756203 | 0"
terminationFlag = 0
rows_list_interaction = []

# This parser function should be used in your script. This is designed to take parameters from Analyse and Track GUI. 
def argParser(argStr):
        tmpStr = argStr.split("|")
        modelName = tmpStr[0].strip()
        videoPath = tmpStr[1].strip()
        outputPath = tmpStr[2].strip()
        startFrame = tmpStr[3].strip()
        endFrame = tmpStr[4].strip()
        frameRate = tmpStr[5].strip()
        generateVideo = tmpStr[6].strip()
        return modelName, videoPath, outputPath, int(startFrame), int(endFrame), float(frameRate), int(generateVideo)


def loadModel(strArg):
    modelName, _, _, _, _, _, _ = argParser(strArg)
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
    print("Please wait. Loading model. This may take several minutes...")
    time.sleep(1.5)
    def loadM(modelName, modelAE):
        modelAE.load_state_dict(torch.load(modelName), strict=False)
        print("Model loaded!!!")
        
    x = th.Thread(target=loadM, args=(modelName, modelAE,))
    x.start()
    x.join()
    return modelAE, device


def analyseVideo(strArg, modelAE, device):
    global terminationFlag, rows_list_interaction

    modelName, videoPath, outputPath, startFrame, endFrame, frameRate, generateVideo = argParser(strArg)
    
    
    print(modelName)
    print(videoPath)
    print(outputPath)
    print('Start Frame: ', startFrame)
    print('End Frame: ', endFrame)
    print('Frame rate: ', frameRate) 
    if generateVideo: 
        print('Generate Video: Yes')
    else:
        print('Generate Video: No')    
    
    ###

    if(terminationFlag==1):
        exit()
    
    def circleIntersect(x1, y1, x2, y2, r1, r2): 
        distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  
        radSumSq = (r1 + r2) * (r1 + r2);  
        if (distSq == radSumSq): 
            return 1 
        elif (distSq > radSumSq): 
            return -1 
        else: 
            return 2
    
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
        global rows_list_interaction
        current_pred_point = track[track_ind].trace[-1][0]
        if(isVanished(track[track_ind])):
            return frame
        
        print(current_pred_point)
        #print(min_max_coor)
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
            
            if(circleIntersect(x1=shoulders[ind][0], y1=shoulders[ind][1], 
                               x2=shoulders[ind2][0], y2=shoulders[ind2][1], 
                               r1=radius, r2=radius2)>0):
                dict_interaction = {"frame_no":frame_no, "pig1_id":track[track_ind].trackId, 
                                    "pig2_id":track[j].trackId, "interaction":2}
                rows_list_interaction.append(dict_interaction)
                cv2.circle(frame, (shoulders[ind][0], shoulders[ind][1]), int(round(radius,0)), (255, 0, 255), 2)
                cv2.circle(frame, (shoulders[ind2][0], shoulders[ind2][1]), int(round(radius2,0)), (255, 0, 255), 2)
                
            if(circleIntersect(x1=shoulders[ind][0], y1=shoulders[ind][1], 
                               x2=tails[ind2][0], y2=tails[ind2][1], 
                               r1=radius, r2=radius2)>0):
                dict_interaction = {"frame_no":frame_no, "pig1_id":track[track_ind].trackId, 
                                    "pig2_id":track[j].trackId, "interaction":3}
                rows_list_interaction.append(dict_interaction)
                cv2.circle(frame, (shoulders[ind][0], shoulders[ind][1]), int(round(radius,0)), (0, 0, 255), 2)
                cv2.circle(frame, (tails[ind2][0], tails[ind2][1]), int(round(radius2,0)), (255, 255, 0), 2)

        return frame
                
                
                    
    #analysis from video
    def createimage(w,h):
        size = (w, h, 1)
        img = np.ones((w,h,3),np.uint8)*255
        return img
    
    def startAnalyse(videoPath, frameRate, startFrame, endFrame, generateVideo, modelAE, device):
        global terminationFlag, rows_list_interaction
        RUN_TILL = endFrame
        
        CODE_NAME = str(time.time())
        VIDEO_IN_PATH = videoPath
        VIDEO_OUT_PATH = os.path.join(outputPath, 'output_'+CODE_NAME+'.mp4')
        
        cap = cv2.VideoCapture(VIDEO_IN_PATH)
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH/2);
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT/2);
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames: ", length)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("FPS: ", fps)
        
        frame_per_sec = frameRate
        gap = int(round(fps/frame_per_sec))
        current_frame_no = startFrame
        
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
            print("pass 1")
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
                
                print("pass 2")
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
                if generateVideo:
                    outImage = m4c.show_coordinates2(input_image.squeeze(dim=0).squeeze(dim=0).numpy(), out_image, 
                                                     label, min_max_coor, rad=5, threshold=0.001, 
                                                     save_fig=False, ret_cv=True)
                    print("pass 3")
                    resFrame = cv2.hconcat([outImage, frame])
                    #resFrame = (resFrame*255).astype(np.uint8)
                
                    video_out.write(resFrame)
                    # Display the resulting frame 
                    #outImage2 = cv2.cvtColor(outImage.get().astype('f'), cv2.COLOR_BGR2RGB)
                    ##cv2.imshow('Frame',  resFrame) 
            if(terminationFlag==1):
                break    
            #if cv2.waitKey(30) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows() 
            #    break
        cap.release()
        video_out.release()
        print("pass 4")
        if(terminationFlag==1):
            return
        if (len(rows_list_interaction)):
            df_interaction = pd.DataFrame(rows_list_interaction)
            df_interaction['time_sec'] = round(df_interaction['frame_no']/fps, 5)
            df_interaction.to_csv(os.path.join(outputPath, 'output'+CODE_NAME+'.csv'))
            print("CSV file generated with name: ", 'output'+CODE_NAME+'.csv')
        
        print("Analysis Completed!!!")
    
    
    #startAnalyse(videoPath, frameRate, startFrame, endFrame)
    
    x = th.Thread(target=startAnalyse, args=(videoPath, frameRate, startFrame, endFrame, generateVideo, modelAE, device,))
    x.start()
    x.join()
    del modelAE




def main_func(strArg):
    modelAE, device = loadModel(strArg)
    analyseVideo(strArg, modelAE, device)

#main_func(strArg)