#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 23:03:44 2020

@author: pronaya
"""

######## Load the Packages
import numpy as np
import math
import cv2
import argparse
import os
from os import environ
import sys
import re

main_path = '/Home/ppd/works/Pig_Videos/Video_SocialInteraction/'
file_name = 'Kamera120170630-022812-1498782492'
video_path = os.path.join(main_path, file_name+'.mp4')
image_path = os.path.join(main_path, file_name)


def frameExtract(KPS, VIDEO_PATH, IMAGE_PATH, n_rows = 2, n_images_per_row = 2):
    if not os.path.exists(IMAGE_PATH):
        os.makedirs(IMAGE_PATH)  
    cap = cv2.VideoCapture(VIDEO_PATH)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", length)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print("FPS: ", fps)
    hop = round(fps / KPS)
    curr_frame = 0
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, ch = frame.shape
        roi_height = int(height / n_rows)
        roi_width = int(width / n_images_per_row)
        #print(roi_height, roi_width)

        if curr_frame % hop == 0:
            images = []
            for x in range(0, n_rows):
                for y in range(0,n_images_per_row):
                    tmp_image=frame[x*roi_height:(x+1)*roi_height, y*roi_width:(y+1)*roi_width]
                    images.append(np.array(tmp_image))
            
            for x in range(0, n_rows):
                for y in range(0, n_images_per_row):
                    #cv2.imshow(str(x*n_images_per_row+y+1), images[x*n_images_per_row+y])
            
                    cv2.imwrite(os.path.join(IMAGE_PATH, 
                            "Cam0_Train5_{0:s}_{1:0000006d}.jpg"
                            .format(str(x*n_images_per_row+y+1), curr_frame)), 
                                images[x*n_images_per_row+y]) # set a new number in front of _frame_ for each video if you want to store several videos in one folder 
        
        #if cv2.waitKey(25) & 0xFF == ord('q'):
            #break
        if curr_frame%1000==0:
            print("Current frame: ", curr_frame)
        curr_frame += 1
        
        
    cap.release()
    print(IMAGE_PATH[0:(len(IMAGE_PATH)-8)])


frameExtract(4, video_path, image_path+'_1slices', 1, 1)
#frameExtract(1, video_path, image_path+'_4slices', 2, 2)