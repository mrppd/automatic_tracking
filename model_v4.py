# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

def secToStr(t):
    return ("%dh, %02dm, %02d.%03ds" % functools.reduce(
        lambda ll,b : divmod(ll[0],b) + ll[1:], [(t*1000,),1000,60,60]))

main_path = '/Home/ppd/works/'
# main_path = '<path>/<to>/<notebook>'
data_path = os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices')

write = False
read = True
model_name = "model_cross_mean_5ch_output_6all"
out_path = os.path.join(data_path, 'output_'+model_name)

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
metadata3 = pd.read_csv(data_path+"/cam 1820181201-204202-1543693322_highAnnotation.csv")

metadata = [metadata1, metadata2, metadata3]
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
  
  

out[65].shape
plt.imshow(ims[638])
plt.imshow(out[638][0])
plt.imshow(out[638][1])
plt.imshow(out[638][2])
plt.imshow(out[638][3])
plt.imshow(out[638][4])
plt.show()

#print(out_img)
#print(int(gr.iloc[0].p3SY))
#print(int(gr.iloc[0].p3SX))


###
#Save dataset
if(write==True):
    np.save(data_path+"/dataset_ims_for_"+model_name+".npy", ims)
    np.save(data_path+"/dataset_labels_for_"+model_name+".npy", out)
    np.save(data_path+"/dataset_label_coors_for_"+model_name+".npy", out_coor)
#load dataset
if(read==True):
    ims = np.load(data_path+"/dataset_ims_for_"+model_name+".npy", allow_pickle=True)[()]
    out = np.load(data_path+"/dataset_labels_for_"+model_name+".npy", allow_pickle=True)[()]
    out_coor = np.load(data_path+"/dataset_label_coors_for_"+model_name+".npy", allow_pickle=True)[()]


###
#plt.imshow(np.reshape(out[100], (-1, int(img_tmp.shape[1]*scale_percent/100))))
plt.imshow(out[828][3])
#plt.plot(out[100])
#out[0, 79, 185]

def show_coordinates(in_im, out_im, rad=10, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0):
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

show_coordinates(ims[550], out[550], rad=5, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0)


###
#sample = {'image': ims, 'labels': out}
#print(len(out))
#print(out[:4])
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

data_set =  pig_dataset(ims, out, out_coor)


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
use_cuda = True
if use_cuda and not torch.cuda.is_available():
    print("Error: cuda requested but not available, will use cpu instead!")
    device = torch.device('cpu')
elif not use_cuda:
    print("Info: will use cpu!")
    device = torch.device('cpu')
else:
    print("Info: cuda requested and available, will use gpu!")
    device = torch.device('cuda:0')


### 
class VAE(nn.Module):

    def __init__(self, latent_dims=10, capacity=32):
        super(VAE, self).__init__()

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


dtype = torch.FloatTensor
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available(): #check for GPU
  dtype = torch.cuda.FloatTensor
  print("GPU available")
  torch.device("cuda")
  torch.set_default_tensor_type('torch.cuda.FloatTensor')
  


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
def train(model, optimizer, loader):
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
def test(model, optimizer, loader):
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
latent_space_dim = 128
capacity = 32
learning_rate = 1e-3
modelVAE = VAE(latent_dims=latent_space_dim, capacity=capacity).to(device)
optimizer = optim.Adam(modelVAE.parameters(), lr=learning_rate)

from torchsummary import summary


###   model_cross_mean_5ch_output_6all
modelVAE.load_state_dict(torch.load(data_path+"/"+model_name+".pyc"), strict=False)


###
def train_and_evaluate_model(model=modelVAE, opt=optimizer, capacity=capacity, num_epochs=10):

    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.99))

    trainL, testL, epochs = [], [], []
    for e in range(1,num_epochs):
      
        train_loss = train(model, optimizer, loader_train)
        train_loss /= train_samples
        test_loss = test(model, optimizer, loader_test)
        test_loss /= test_samples
        trainL.append(train_loss)
        testL.append(test_loss)
        epochs.append(e)
        #print("epoch: "+str(e))

        print(f'Epoch {e}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}')

    sns.lineplot(x=epochs,y=trainL, label='Train')
    sns.lineplot(x=epochs,y=testL, label='Test')
    plt.legend()
    plt.show()
    return model


###
start_time = time.time()
model_128 = train_and_evaluate_model(num_epochs=3000)
model_128
print("Total time:", secToStr(time.time()-start_time), "\n")   
torch.save(model_128.state_dict(), data_path+"/"+model_name+".pyc")


###    test
modelVAE.eval()
data_iterator = iter(loader_test)
images, labels, label_coor_ind = data_iterator.next()

"""
imm = images[3].unsqueeze(0)
with torch.no_grad():
    y_hat, mu, logvar = modelVAE(imm.to(device))
    print(y_hat.shape)
    show_coordinates(imm[0].squeeze(dim=0).numpy(), y_hat[0].cpu().numpy(), rad=7, threshold=0.001)
"""
test_data=3
test_image = []
with torch.no_grad():
    y_hat, mu, logvar = modelVAE(images.to(device))
    given_pts = out_coor[int(label_coor_ind[test_data].numpy())]
    print(y_hat[test_data].shape)
    print(images[test_data].squeeze(dim=0).numpy().shape, labels[test_data].numpy().shape)
    #print(ims[350].shape, out[350].shape)
    show_coordinates(images[test_data].squeeze(dim=0).numpy(), y_hat[test_data].cpu().numpy(), rad=7, threshold=0.001)

    plt.plot(y_hat[test_data].cpu().numpy()[0])
    inp_image = images[test_data].squeeze(dim=0).numpy()
    out_image =  y_hat[test_data].cpu().numpy()


    for ch in range(out_image.shape[0]):
        #out_image[ch,out_image[ch]<0.001] = 0
        out_image[ch,out_image[ch]>0.001] = out_image[ch,out_image[ch]>0.001]*10000

    plt.imshow(out_image[2])





def get_max(image,sigma,alpha=3,size=10):
    from copy import deepcopy
    import numpy as np
    # preallocate a lot of peak storage
    k_arr = np.zeros((10000,2))
    image_temp = deepcopy(image)
    peak_ct=0
    while True:
        k = np.argmax(image_temp)
        j,i = np.unravel_index(k, image_temp.shape)
        if(image_temp[j,i] >= alpha*sigma):
            k_arr[peak_ct]=[j,i]
            # this is the part that masks already-found peaks.
            x = np.arange(i-size, i+size)
            y = np.arange(j-size, j+size)
            xv,yv = np.meshgrid(x,y)
            # the clip here handles edge cases where the peak is near the 
            #    image edge
            image_temp[yv.clip(0,image_temp.shape[0]-1),
                               xv.clip(0,image_temp.shape[1]-1) ] = 0
            peak_ct+=1
        else:
            break
    # trim the output for only what we've actually found
    return k_arr[:peak_ct]


maxCoor = get_max(out_image[2], 1.5)




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

label, min_max_coor = analyseLines(out_image)


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


show_coordinates2(inp_image, out_image, label, min_max_coor, rad=5, threshold=0.001, save_fig=False)


def countMatch(analysed_points, given_points, verbose=False):
    if(verbose):
        print(analysed_points)
        print(given_points)
    FP = 0
    FN = 0
    analysed_center_points = []
    for line in analysed_points:
        analysed_center_points.append((line[0]+line[1])/2)
    #print(analysed_center_points)
    
    map_arr_acp = np.zeros((len(analysed_center_points)), dtype=int)
    map_arr_gp = np.zeros((given_points.shape[0]), dtype=int)
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
                map_arr_gp[i]=1
                break;
    for ind_acp, acp in enumerate(analysed_center_points):
        if(map_arr_acp[ind_acp]==0):
            FP = FP + 1
            
    for i in range(given_points.shape[0]):
        if(map_arr_gp[i]==0):
            FN = FN + 1
    return(count_pred, given_pts.shape[0], FP, FN)
            
    

count_pred, count_given, FP, FN = countMatch(min_max_coor, given_pts)
print(count_pred, count_given, FP, FN)

###
modelVAE.eval()
cnt = 0
total_count_pred = 0
total_count_given = 0
total_FP = 0
total_FN = 0
with torch.no_grad():
    for i, (x, y, z) in enumerate(loader_train):
        y_hat, mu, logvar = modelVAE(x.to(device))
        for bt in range(len(x)):
            #print(out_coor[int(z[bt].numpy())])
            given_pts = out_coor[int(z[bt].numpy())]
            #print(x[bt].squeeze(dim=0).numpy().shape, y[bt].numpy().shape)
            out_image = y_hat[bt].cpu().numpy()
            for ch in range(out_image.shape[0]):
                out_image[ch,out_image[ch]<0.001] = 0
                out_image[ch,out_image[ch]>0.001] = 1
            label, min_max_coor = analyseLines(out_image)
            
            #print(min_max_coor)
            count_pred, count_given, FP, FN = countMatch(min_max_coor, given_pts)
            total_count_pred = total_count_pred + count_pred
            total_count_given = total_count_given + count_given
            total_FP = total_FP + FP
            total_FN = total_FN + FN
            
            show_coordinates(x[bt].squeeze(dim=0).numpy(), out_image, rad=5, 
                              threshold=0.001, save_fig=True, load_ind=cnt, bt_ind=bt)
            show_coordinates2(x[bt].squeeze(dim=0).numpy(), out_image, label, min_max_coor, rad=5, 
                              threshold=0.001, save_fig=True, load_ind=cnt, bt_ind=bt)
            #plt.plot(y_hat[bt].cpu().numpy()[0])
            cnt = cnt + 1
acc = (total_count_pred/total_count_given)*100
precision = total_count_pred/(total_count_pred+total_FP)
print(acc, total_FP, total_FN, precision)
      
##################################################################################
def circleIntersect(x1, y1, x2, y2, r1, r2): 
    distSq = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);  
    radSumSq = (r1 + r2) * (r1 + r2);  
    if (distSq == radSumSq): 
        return 1 
    elif (distSq > radSumSq): 
        return -1 
    else: 
        return 2

"""
interactionDict = {}    
def socialInteraction(track, shoulders, tails, frame_no):
    
    current_pred_point = track.trace[-1][0]
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
    radius = (np.linalg.norm(shoulders[ind]-tails[ind])/3)*2
    
    print("Distance threshold: ", radius)
    
    arrFrame = []
    
    for i in range(len(shoulders)):
        if(i==ind):
            continue
        
        dist_s = np.linalg.norm(shoulders[ind]-shoulders[i]) 
        print("head to head: ", dist_s)
        
        if(dist_s<radius):
            arrFrame.append((frame_no, 2))
            if track.trackId not in interactionDict.keys():
                interactionDict[track.trackId] = arrFrame
            else:
                interactionDict[track.trackId].append((frame_no, 2))
                
                
        dist_t = np.linalg.norm(shoulders[ind]-tails[i]) 
        print("head to tail: ", dist_t)
        
        if(dist_t<radius):
            arrFrame.append((frame_no, 3))
            if track.trackId not in interactionDict.keys():
                interactionDict[track.trackId] = arrFrame
            else:
                interactionDict[track.trackId].append((frame_no, 3))
                
"""

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
RUN_TILL = 1200
rows_list_interaction = []  
def createimage(w,h):
    size = (w, h, 1)
    img = np.ones((w,h,3),np.uint8)*255
    return img
CODE_NAME = "piglet1_slow12_6fps"
VIDEO_IN_PATH = os.path.join(main_path, 'Pig_Videos/Video_SocialInteraction/cam 5 ATW20181113-190716-1542132436.mp4')
VIDEO_OUT_PATH = os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices/track_'+CODE_NAME+'.mp4')

cap = cv2.VideoCapture(VIDEO_IN_PATH)
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_WIDTH/2);
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FRAME_HEIGHT/2);
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total frames: ", length)
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS: ", fps)

frame_per_sec = 6
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
        y_hat, mu, logvar = modelVAE(input_image.to(device))
        
        out_image = y_hat[0].cpu().numpy()
        for ch in range(out_image.shape[0]):
            out_image[ch,out_image[ch]<0.001] = 0
            out_image[ch,out_image[ch]>0.001] = 1
        label, min_max_coor = analyseLines(out_image)
        
    
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
    
        outImage = show_coordinates2(input_image.squeeze(dim=0).squeeze(dim=0).numpy(), out_image, 
                                     label, min_max_coor, rad=5, threshold=0.001, 
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


##### others
with open(os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices/df_interaction.pickle'), 'wb') as handle:
    pkl.dump(df_interaction, handle, protocol=pkl.HIGHEST_PROTOCOL)
with open(os.path.join(main_path, 'Pig_Videos/Kamera-120170504-120223-1493892143_1slices/interactionDict.pickle'), 'wb') as handle:
    pkl.dump(interactionDict, handle, protocol=pkl.HIGHEST_PROTOCOL)

