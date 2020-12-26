#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 01:21:36 2020

@author: ppd
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 01:27:58 2020

@author: ppd
"""

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

class pig_dataset(torch.utils.data.Dataset):

    def __init__(self, ims, out, out_coor, transform=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.ims = ims
        self.out = out
        self.out_coor_ind = range(len(out_coor))
        
        self.transform = transform

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):

        image =  torch.from_numpy(self.ims[idx]).unsqueeze(dim=0)
        labels =  torch.from_numpy(self.out[idx])
        label_coors_ind = torch.from_numpy(np.array(self.out_coor_ind[idx]))
        #labels = labels.reshape(-1, 2)
        sample = (image, labels, label_coors_ind)

        return sample
    
### model
class AE(nn.Module):

    def __init__(self, latent_dims=10, capacity=32):
        super(AE, self).__init__()

        self.latent_dims  = latent_dims
        #self.image_size   = image_size
        self.capacity     = capacity

        # Encoder 
        self.encoder = nn.Sequential(
            nn.Conv2d(1, self.capacity, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.capacity), 
            nn.Conv2d(self.capacity,2*self.capacity, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(2*self.capacity), 
            nn.Flatten() # because we'll use Linear layers for trafo into latent space
        )

        # Reparametrization (fc = fully connected)
        self.fc_mu      = nn.Linear(2*self.capacity*100*160, self.latent_dims)
        self.fc_logvar  = nn.Linear(2*self.capacity*100*160, self.latent_dims)
        self.fc_z       = nn.Linear(self.latent_dims, 2*capacity*100*160)

        # Decoder
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(2*self.capacity),
            nn.ConvTranspose2d(2*self.capacity, self.capacity, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.capacity), 
            nn.ConvTranspose2d(self.capacity, 5, kernel_size=(4,4), stride=2, padding=1),
            nn.Sigmoid()
        )
  
    def encode(self, x):
        ''' Encoder: output is (mean, log(variance))'''
        x = self.encoder(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def sample(self, mu, logvar):
        ''' Sample from Gaussian with mean `mu` and SD `sqrt(exp(logvarz))`'''
        if self.training:
            #print('––––––SAMPLING–––––––')
            # Reparameterization + sampling
            #sample_latent = torch.normal(mu, torch.sqrt(torch.exp(logvar))) 
            # => bad idea, because: sampling is not differentiable

            # reparametrization
            eps = torch.randn_like(mu)
            #print(f'eps:{torch.mean(eps):.2f}')

            # sampling
            sample_latent = eps.mul(torch.sqrt(torch.exp(logvar))).add_(mu)
            #print(f'mu:{torch.mean(mu):.2f}')
            #print(f'var:{torch.mean(torch.sqrt(torch.exp(logvar))):.2f}')
            #print('––––––\nlatent shapes:')
            #print(torch.mean(sample_latent))
            
            #print(torch.mean(sample_latent))
            #print(sample_latent.shape)
    
            return sample_latent
        else:
            # During testing we don't sample but take the mean
            return mu

    def decode(self, z):
        '''Decoder: produces reconstruction from sample of latent z'''
        z = self.fc_z(z)
        #z = nn.BatchNorm1d(2*self.capacity*100*160)(z)
        z = z.view(z.shape[0], 64, 100, 160)
        x_hat = self.decoder(z)
      
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        #z = self.sample(mu, logvar)
        #print(torch.mean(z))
        x_hat = self.decode(mu)
        #print(torch.mean(x_hat))
        return x_hat, mu, logvar



def secToStr(t):
    return ("%dh, %02dm, %02d.%03ds" % functools.reduce(
        lambda ll,b : divmod(ll[0],b) + ll[1:], [(t*1000,),1000,60,60]))


def onLine(P, P1, P2):
    x, y = P[0], P[1]
    x1, y1 = P1[0], P1[1]
    x2, y2 = P2[0], P2[1]
    res = ((x-x1)*(y1-y2))-((y-y1)*(x1-x2))
    #print(res)
    if(abs(res) <= 0.0001):
        return True
    return False


def drawLine(origin, endpoint,surface):
    deltaX = endpoint[0] - origin[0]
    deltaY = endpoint[1] - origin[1]

    deltaMax = max(abs(deltaX), abs(deltaY))
    deltaX = deltaX/deltaMax
    deltaY = deltaY/deltaMax
    for n in range(deltaMax):
        surface[int(origin[0]), int(origin[1])]=255
        #print(int(origin[0]), int(origin[1])
        origin[0] = origin[0] + deltaX;
        origin[1] = origin[1] + deltaY;
        
    return surface


def show_coordinates(in_im, out_im, out_path, rad=10, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0):
  cidx = np.where(out_im > threshold)
  print(cidx)

  #print(in_im)
  in_im2 = np.empty((in_im.shape[0], in_im.shape[1], 3), dtype=np.float64)
  in_im2[:, :, 0] = in_im
  in_im2[:, :, 1] = in_im
  in_im2[:, :, 2] = in_im


  in_im2 = cv2.UMat(in_im2)
  for i in range(len(cidx[0])):
    #print(cidx[2][i], cidx[1][i])
    if(cidx[0][i]==0):
      in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(255, 0, 0), thickness=cv2.FILLED)
    if(cidx[0][i]==1):
      in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(0, 0, 255), thickness=cv2.FILLED)
    if(cidx[0][i]==2):
      in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(255, 0, 255), thickness=cv2.FILLED)
    if(cidx[0][i]==3):
      in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(255, 255, 0), thickness=cv2.FILLED) 
    if(cidx[0][i]==4):
      in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=int(rad/5), color=(0, 255, 255), thickness=cv2.FILLED)
 
  in_im2 =  in_im2.get().astype('f')
  #print(in_im)
  plt.imshow(in_im2)
  if(save_fig):
      if not os.path.exists(out_path):
        os.makedirs(out_path)
      plt.savefig(out_path+"/"+str(load_ind)+"_"+str(bt_ind)+".png", dpi=150) 
  plt.show()
  

def useCuda(use_cuda=True):
    if use_cuda and not torch.cuda.is_available():
        print("Error: cuda requested but not available, will use cpu instead!")
        device = torch.device('cpu')
    elif not use_cuda:
        print("Info: will use cpu!")
        device = torch.device('cpu')
    else:
        print("Info: cuda requested and available, will use gpu!")
        device = torch.device('cuda:0')
    return device


###
def vae_loss(recon_y, y, mu, logvar):
    # Reconstruction losses are calculated using binary cross entropy (BCE) and 
    # summed over all elements and batch
    bce_loss = F.binary_cross_entropy(recon_y, y, reduction='mean')
    #bce_loss = F.mse_loss(recon_y, y)
    # See Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    kld_loss = -0.5*torch.sum(1.0+logvar - mu**2 - torch.exp(logvar))

    total_loss = bce_loss #+ kld_loss

    return total_loss, bce_loss, kld_loss


###
def train(model, optimizer, loader, device):
    model.train()
    epoch_loss = 0

    for i, (x, y, _) in enumerate(loader):
  
        x = x.to(device)
        y = y.to(device)
        current_batch_size = x.shape[0]
        #print(f'===================================\nIndex: {i}')

        
        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        y_hat, mu, logvar = model(x)
        total_loss, bce_loss, kld_loss = vae_loss(y_hat, y, mu, logvar)
        total_loss, bce_loss, kld_loss = total_loss/current_batch_size, bce_loss/current_batch_size, kld_loss/current_batch_size

        # backward pass
        total_loss.backward()
        epoch_loss += total_loss.item()
        
        # update the weights
        optimizer.step()

    return epoch_loss


###
def test(model, optimizer, loader, device):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y, _) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            current_batch_size = x.shape[0]
            # forward pass
            y_hat, mu, logvar = model(x)

            total_loss, bce_loss, kld_loss = vae_loss(y_hat, y, mu, logvar)
            total_loss, bce_loss, kld_loss = total_loss/current_batch_size, bce_loss/current_batch_size, kld_loss/current_batch_size
            
            test_loss += total_loss.item()

    return test_loss


###
terminationFlag = 0
def train_and_evaluate_model(device, model, opt, cap, loader_train, train_samples, loader_test, test_samples, num_epochs=10):
    global terminationFlag
    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.99))

    trainL, testL, epochs = [], [], []
    for e in range(1,num_epochs):
      
        train_loss = train(model, opt, loader_train, device)
        train_loss /= train_samples
        test_loss = test(model, opt, loader_test, device)
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

  
def analyseLines(test_image):
    add_next = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1], 
                [2, 0], [-2, 0], [0, 2], [0, -2], [2, -2], [2, 2], [-2, 2], [-2, -2],
                [-1, 2], [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1]]
    
    def dfs(graph, start, component):
        #call = call + 1
        if(marker[start[0]][start[1]]==1):
            return 0
        if(graph[start[0]][start[1]]==1.0):
            component = component + 1
            coor_list.append(start)
        else:
            return 0
        marker[start[0]][start[1]] = 1
        pc =0 
        for coor_add in add_next:
            #print(coor_add)
            next_coor = [start[0]+coor_add[0], start[1]+coor_add[1]]
            #print(next_coor)
            if(next_coor[0]<0 or next_coor[1]<0 or next_coor[0]>=graph.shape[0] or next_coor[1]>=graph.shape[1]):
                continue
            if(graph[next_coor[0]][next_coor[1]]==1.0 and marker[next_coor[0]][next_coor[1]]==0):
                 pc = pc + dfs(graph, next_coor, component)
        return component+pc
    
    #call = 0
    cnt = 0 
    coor_list = []
    tmp = []
    min_max_coor = []
    local_threshold_sum = 0
    local_threshold_cnt = 0
    local_threshold = 0
    marker = np.zeros(test_image[4].shape, dtype=int)  
    for i in range(test_image[4].shape[0]):
        for j in range(test_image[4].shape[1]):
            if(marker[i][j]==1): continue
            coor_list = []
            ret = dfs(test_image[4], [i, j], 0)
            if(ret>1000):
                local_threshold_cnt = local_threshold_cnt + 1
                local_threshold_sum = local_threshold_sum + len(coor_list)        
                local_threshold = (local_threshold_sum / local_threshold_cnt)*0.50
                ##print(local_threshold)       
                if(len(coor_list)>local_threshold):
                    ##print("No of coord: ", len(coor_list))
                    tmp = np.array(coor_list)
                    tmp = tmp[np.lexsort(tmp.T[::-1])]
                    min_max_coor.append([tmp[0], tmp[-1]])
                    cnt = cnt + 1
                #print(ret)
                #print(i, j)
    
    ##print(cnt)   
    
    shoulder_coors = np.column_stack(np.where(test_image[2]==1))
    tail_coors = np.column_stack(np.where(test_image[3]==1))
    
    label = []
    for line_ind in range(len(min_max_coor)):
        min_dist_sc = 999999999
        #sc_ind = 0
        for sc in range(len(shoulder_coors)):
            dist = np.linalg.norm(shoulder_coors[sc]-min_max_coor[line_ind][0])
            if(min_dist_sc>=dist):
                min_dist_sc = dist
                #sc_ind = sc
        
        #print(sc_ind, min_dist_sc)
        
        min_dist_tc = 999999999
        #tc_ind = 0
    
        for tc in range(len(tail_coors)):
            dist = np.linalg.norm(tail_coors[tc]-min_max_coor[line_ind][0])
            if(min_dist_tc>=dist):
                min_dist_tc = dist
                #tc_ind = tc
        
        #print(tc_ind, min_dist_tc)
        if(min_dist_sc>min_dist_tc):
            label.append(3)
        else:
            label.append(2)
        
        ##print(label[line_ind], min_max_coor[line_ind])
        
    return (label, min_max_coor)

#label, min_max_coor = analyseLines(out_image)


def show_coordinates2(in_im, out_im, label, min_max_coor, rad=10, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0, ret_cv = False):
    cidx = np.where(out_im > threshold)
    #print(cidx)
    #print(in_im)
    """
    in_im2 = np.empty((in_im.shape[0], in_im.shape[1], 3), dtype=np.float64)
    in_im2[:, :, 0] = in_im
    in_im2[:, :, 1] = in_im
    in_im2[:, :, 2] = in_im
    """
    in_im = (in_im*255).astype(np.uint8)
    in_im2 = cv2.cvtColor(in_im, cv2.COLOR_GRAY2RGB)
    
    
    #in_im2 = cv2.UMat(in_im2)
    for i in range(len(cidx[0])):
        #print(cidx[2][i], cidx[1][i])
        if(cidx[0][i]==0):
            in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(255, 0, 0), thickness=cv2.FILLED)
        if(cidx[0][i]==1):
            in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(0, 0, 255), thickness=cv2.FILLED)
        if(cidx[0][i]==2):
            in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(255, 0, 255), thickness=cv2.FILLED)
        if(cidx[0][i]==3):
            in_im = cv2.circle(in_im2, (cidx[2][i], cidx[1][i]), radius=rad, color=(255, 255, 0), thickness=cv2.FILLED) 
        #if(cidx[0][i]==4):
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    
    for ind, line in enumerate(min_max_coor):  
        #print(ind, line)
        in_im = cv2.line(in_im2, (line[0][1], line[0][0]), (line[1][1], line[1][0]), (0, 255, 255), 3)
        if(label[ind]==2):
            in_im = cv2.putText(in_im2,'S', (line[0][1], line[0][0]), font, 0.8, (123, 204, 102), 2, cv2.LINE_AA)
            in_im = cv2.putText(in_im2,'T', (line[1][1], line[1][0]), font, 0.8, (123, 204, 102), 2, cv2.LINE_AA)
        elif(label[ind]==3):
            in_im = cv2.putText(in_im2,'T', (line[0][1], line[0][0]), font, 0.8, (123, 204, 102), 2, cv2.LINE_AA)
            in_im = cv2.putText(in_im2,'S', (line[1][1], line[1][0]), font, 0.8, (123, 204, 102), 2, cv2.LINE_AA)
    
    if(ret_cv==False):
        #in_im2 =  in_im2.get().astype('f')
        #print(in_im)
        plt.imshow(in_im2)
        
    if(save_fig):
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        plt.savefig(out_path+"/"+str(load_ind)+"_"+str(bt_ind)+"_text.png", dpi=150) 
    #plt.show()
    if(ret_cv==True):
        return in_im2
    

def countMatch(analysed_points, given_points, verbose=False):
    if(verbose):
        print(analysed_points)
        print(given_points)
    analysed_center_points = []
    for line in analysed_points:
        analysed_center_points.append((line[0]+line[1])/2)
    #print(analysed_center_points)
    
    map_arr_acp = np.zeros((len(analysed_center_points)), dtype=int)
    count_pred = 0
    for i in range(given_points.shape[0]):
        sh_pt = given_points[i][2]
        tl_pt = given_points[i][3]
        center_pt = (sh_pt[0]+tl_pt[0])/2, (sh_pt[1]+tl_pt[1])/2 
        radius = np.linalg.norm(tl_pt-sh_pt)/7
        if(verbose):
            print(i)
            print("rad: ", radius)
        for ind_acp, acp in enumerate(analysed_center_points):
            if(map_arr_acp[ind_acp]==1):
                continue   
            dist =  np.linalg.norm(center_pt-acp)
            if(verbose):
                print("dist: ", dist)
            if(dist<radius):
                count_pred = count_pred + 1
                map_arr_acp[ind_acp]=1
                break;
    return(count_pred, given_points.shape[0])
            

def circleIntersect(x1, y1, x2, y2, r1, r2): 
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  
    radSumSq = (r1 + r2) * (r1 + r2);  
    if (distSq == radSumSq): 
        return 1 
    elif (distSq > radSumSq): 
        return -1 
    else: 
        return 2    