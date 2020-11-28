"""
Created on Sat Apr 13 00:18:22 2019

@author: Pronaya
"""

import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
import os
import pickle

main_path = '/Home/ppd/works/'
url = os.path.join(main_path, 'Pig_Videos/Video_SocialInteraction/')
fileName = "cam 1820181201-204202-1543693322"
ext = ".mp4"
way = 2
threshold = 10   # for the way 1, different threshold values are needed. An intuitive guess can be made by observing the difference plot.
takeMaxFrame = 500 

writeVideos = False
writeImages = True
showFrames = False

"""
sub = "MOG2"
if sub == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

"""    

# Open the video file using opencv     
capture = cv2.VideoCapture(url+fileName+ext)
if not capture.isOpened:
    print('Unable to open: ' + url)
    exit(0)


frame_width = int(capture.get( cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get( cv2.CAP_PROP_FRAME_HEIGHT))
FPS = capture.get(cv2.CAP_PROP_FPS)
# Initializing video writers 
if(writeVideos==True):
    outLessMove = cv2.VideoWriter(url+'less_movement_'+fileName+ext, cv2.VideoWriter_fourcc('A','V','C','1'), 
                                  FPS, (frame_width,frame_height), True)
    outHighMove = cv2.VideoWriter(url+'high_movement_'+fileName+ext, cv2.VideoWriter_fourcc('A','V','C','1'), 
                                  FPS, (frame_width,frame_height), True)
    outHighMoveSelected = cv2.VideoWriter(url+'high_movement_selected_'+fileName+ext, cv2.VideoWriter_fourcc('A','V','C','1'), 
                                  FPS, (frame_width,frame_height), True)


#capture.set(cv2.CAP_PROP_POS_AVI_RATIO,1)
#length = capture.get(cv2.CAP_PROP_POS_MSEC)
totalFrames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
print("Tootal Frames: "+str(totalFrames) + "\n")

# Find the Frobenius Norms
normList = []
while True:
    #capture.set(1, 3-1)    
    ret, frame = capture.read()
    if frame is None:
        break    
    
    # Converting the color frame to gray frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
    #frame = cv2.GaussianBlur(frame, (21, 21), 0)
    
    # Find the Frobenius Norm for single frame
    fnorm = np.linalg.norm(frame)
    normList.append(fnorm)
    #print(fnorm)
    """
    cv2.imshow('Frame', frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
    """
    # Show the percentage for Frobenius Norm calculation
    #percent = (capture.get(cv2.CAP_PROP_POS_FRAMES)/totalFrames)*100
    if(int(capture.get(cv2.CAP_PROP_POS_FRAMES))%1000==0):
        percent = (capture.get(cv2.CAP_PROP_POS_FRAMES)/totalFrames)*100
        print(str(round(percent)) + "%")



plt.plot(normList[0:2500])
plt.savefig(url+'NormList.png', format='png', dpi=500)

# Smoothing the values using Savitzky–Golay filter 
#https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter
# Any kinds of filters can be used
newNormList = savgol_filter(normList, 101, 3)


def smooth(x, box_pts, wh):
    if wh == "gaussian" :
        box = gaussian_filter(np.ones(box_pts), 1.5)
    elif wh == "box":
        box = np.ones(box_pts)/box_pts        
    x_smooth = np.convolve(x, box, mode='same')
    return x_smooth

#newNormList = smooth(normList, 50, "gaussian") # gaussian or box
plt.plot(newNormList[0:2500])
plt.savefig(url+'SmoothNormList.png', format='png', dpi=500)


"""
newSubList = []
movingAvgList = []
movingavg = 0
for i in range(0, len(normList)):
    movingavg =movingavg + normList[i]
    if(i>0):
        movingavg = movingavg/2
    movingAvgList.append(movingavg)
    
    if((normList[i]-movingavg)>=0):
        newSubList.append(normList[i]-movingavg)


movingAvgList = movingAvgList - min(movingAvgList)
newSubList = newSubList -  min(newSubList)
plt.plot(newSubList)
plt.plot(movingAvgList[0:300])
"""


# There are different ways to find the differences
interval = int(FPS+1)
if(way==1):
    # Way 1: find the difference between the values of position i and i+interval
    # here interval is 2 * frames per seconds (FPS)
    # Moreover, taking the average of some of these consecutive differences will make the 
    # curve smoother and reduce small fluctuations, but it wokrs without averaging. 
    # averaging is done with a size of interval.
    diffList = []
    for i in range(0, len(newNormList)-(interval*2-1)):
        diff = 0
        for j in range(i, i+interval):
            diff = diff + abs(newNormList[j]-newNormList[j+interval])

        diff = diff/ interval
        diffList.append(diff)
else:
    # Way 2: find the difference between the values of position i and i+1
    # Then, Take the average of these consecutive differences with a window size of interval. 
    # here interval is 2 * frames per seconds (FPS)
    diffList = []
    for i in range(interval, len(newNormList)):
        diff = 0
        for j in range(i-(interval-1), i):
            diff = diff + abs(newNormList[j]-newNormList[j-1])

        diff = diff/ len(range(i-(interval-1), i))
        diffList.append(diff)     

plt.plot(diffList[0:2500]) 
plt.savefig(url+'diffList.png', format='png', dpi=500)

diffListSortedIndex = np.argsort(-np.array(diffList))

capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0)   #start from the beginning
#capture.set(1, interval-1)  
i = 0
selectedFramesSet = []
while True:
    ret, frame = capture.read()
    
    if(i>=len(diffList)):
        break
    
    # Skip all the frames with the corresponding diffList values which are greater than threshold.
    if(diffList[i]>threshold):  
        i = i+interval
        if(i>totalFrames):
            break
        capture.set(1, i-1)  
        continue
    
    if frame is None:
        break
    
    #print(ret)
    
    # Select the less movement frames no
    selectedFramesSet.append(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))
    
    # Make video using those selected less movement frames
    if(writeVideos==True):
        outLessMove.write(frame)
    
    # Save less movement frames as jpeg
    if(writeImages==True):    
        if not os.path.exists(url+fileName+"_less/"):
            os.makedirs(url+fileName+"_less/")
        cv2.imwrite(url+fileName+"_less/"+"frame"+str(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))+".jpg", frame)
    
    # Show less movement frames on screen
    if(showFrames==True):
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
        cv2.imshow('Frame', frame)
        #cv2.imshow('FG Mask', fgMask)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    
    i = i + 1 
    # Show percentage
    if(i%1000==0):
        percent = (i/totalFrames)*100
        print(str(round(percent)) + "%")


if(writeVideos==True):
    outLessMove.release()    

totaFramesSet = list(range(0, int(totalFrames)))    
movingFramesSet = list(set(totaFramesSet) - set(selectedFramesSet)) #Select the high movement frames
movingFramesSet.sort()
selectedFramesSet.sort()



#start from the beginning
capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0) 
j = 0
jMax = len(movingFramesSet)
for i in movingFramesSet:
    capture.set(1, i-1)  
    ret, frame = capture.read() 
    
    # Make video using those selected high movement frames
    if(writeVideos==True):
        outHighMove.write(frame)
        
    # Save high movement frames as jpeg
    if(writeImages==True):   
        if not os.path.exists(url+fileName+"_high/"):
            os.makedirs(url+fileName+"_high/")
        cv2.imwrite(url+fileName+"_high/"+"frame"+str(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))+".jpg", frame)
    
    # Show less movement frames on screen
    if(showFrames==True):
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.imshow('Frame', frame)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    
    j = j + 1  
    # Show percentage
    if(j%100==0):
        percent = (j/jMax)*100
        print(str(round(percent)) + "%")


if(writeVideos==True):
    outHighMove.release()



#Selected frames
#start from the beginning
capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0) 
j = 0
jMax = takeMaxFrame
for i in diffListSortedIndex:
    capture.set(1, i-1)  
    ret, frame = capture.read() 
    
    if((j>jMax) or (not ret)):
        break
    
    # Make video using those selected high movement frames
    if(writeVideos==True):
        outHighMoveSelected.write(frame)
        
    # Save high movement frames as jpeg
    if(writeImages==True):   
        if not os.path.exists(url+fileName+"_high_selected/"):
            os.makedirs(url+fileName+"_high_selected/")
        cv2.imwrite(url+fileName+"_high_selected/"+"frame"+str(int(capture.get(cv2.CAP_PROP_POS_FRAMES)))+".jpg", frame)
    
    # Show less movement frames on screen
    if(showFrames==True):
        cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
        cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        cv2.imshow('Frame', frame)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break
    
    j = j + 1  
    # Show percentage
    if(j%100==0):
        percent = (j/jMax)*100
        print(str(round(percent)) + "%")


if(writeVideos==True):
    outHighMoveSelected.release()
    
    
"""
# ovservation of less 
capture.set(cv2.CAP_PROP_POS_AVI_RATIO,0) 
for i in selectedFramesSet:
    capture.set(1, i-1)  
    ret, frame = capture.read()
    
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
        
    cv2.imshow('Frame', frame)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

"""

# Closes all the frames
cv2.destroyAllWindows()    