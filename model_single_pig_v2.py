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


main_path = '/Home/ppd/works/'
# main_path = '<path>/<to>/<notebook>'
data_path = os.path.join(main_path, 'Pig_Videos')
out_path = os.path.join(data_path, 'output')

###
print(data_path)
directories_name = ['Kamera120170728-052920-1501212560_2slices', 
                    'Kamera120170728-175923-1501257563_2slices',
                    'Kamera120170728-212923-1501270163_2slices',
                    'Kamera 120170504-120223-1493892143_2slices',
                    'Kamera 120170730-122934-1501410574_2slices',
                    'Kamera 120170730-155934-1501423174_2slices',
                    'Kamera 120180222-174653-1519318013_2slices']
images_files_all = []

for dir_name in directories_name:
    images_files  = [os.path.join(data_path, dir_name, f) for f in os.scandir(os.path.join(data_path, dir_name)) if f.path.endswith('.jpg') ]
    images_files_all = images_files_all + images_files

print ("Total Number of Images: {} (should be 25200)".format(len(images_files_all)))

###
scale_percent = 50 # percent of original size
total_sample = 500

###
csv_list = []
for dir_name in directories_name:
    csv_file_name = dir_name + 'Annotation.csv'
    metadata = pd.read_csv(data_path+'/'+csv_file_name)
    csv_list.append(metadata)

metadata = pd.concat(csv_list)
"""
metadata_bg = metadata[(metadata.p1 == "-1,-1") & (metadata.p2 == "-1,-1" ) & 
                       (metadata.p3 == "-1,-1" ) & (metadata.p4 == "-1,-1" )]
metadata = metadata[(metadata.p1 != "-1,-1") & (metadata.p2 != "-1,-1" ) & 
                    (metadata.p3 != "-1,-1" ) & (metadata.p4 != "-1,-1" )]
"""
metadata['p1SX'], metadata['p1SY'] = metadata.p1.str.split(',', 1).str
metadata['p2SX'], metadata['p2SY'] = metadata.p2.str.split(',', 1).str
metadata['p3SX'], metadata['p3SY'] = metadata.p3.str.split(',', 1).str
metadata['p4SX'], metadata['p4SY'] = metadata.p4.str.split(',', 1).str

metadata['p1SX'] = pd.to_numeric(metadata['p1SX']) * scale_percent / 100
metadata['p1SY'] = pd.to_numeric(metadata['p1SY']) * scale_percent / 100
metadata['p2SX'] = pd.to_numeric(metadata['p2SX']) * scale_percent / 100
metadata['p2SY'] = pd.to_numeric(metadata['p2SY']) * scale_percent / 100
metadata['p3SX'] = pd.to_numeric(metadata['p3SX']) * scale_percent / 100
metadata['p3SY'] = pd.to_numeric(metadata['p3SY']) * scale_percent / 100
metadata['p4SX'] = pd.to_numeric(metadata['p4SX']) * scale_percent / 100
metadata['p4SY'] = pd.to_numeric(metadata['p4SY']) * scale_percent / 100

metadata['baseFolderName'+'FileName'] = metadata['baseFolderName'] + '/' + metadata['fileName']
metadata



unique_file_names_all = metadata.baseFolderNameFileName.unique()

metadata_sample = metadata.groupby('baseFolderNameFileName')
#metadata_sample.head()
#metadata_sample.get_group(unique_file_names[0])

#metadata_sample = metadata.sample(500).reset_index()
#print(metadata_sample)
unique_file_names_prune = []
for ui in unique_file_names_all:
    if len(metadata_sample.get_group(ui))==1:
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
image_path = data_path+"/"+metadata_sample.get_group(unique_file_names[1]).iloc[0].baseFolderName+"/"+metadata_sample.get_group(unique_file_names[1]).iloc[0].fileName
img_tmp = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
print("Original shape: ",img_tmp.shape)
print("Reduced shape: ",(int(img_tmp.shape[0]*scale_percent/100), int(img_tmp.shape[1]*scale_percent/100)))
ims = np.empty((total_sample, int(img_tmp.shape[0]*scale_percent/100), int(img_tmp.shape[1]*scale_percent/100)), dtype=np.float32)

for i in range(total_sample):
  print(i, end=" ")
  if i%30==0: print() 
  image_path = data_path+"/"+metadata_sample.get_group(unique_file_names[i]).iloc[0].baseFolderName+"/"+metadata_sample.get_group(unique_file_names[i]).iloc[0].fileName
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
  #print('Original Dimensions : ',img.shape)

  width = int(img.shape[1] * scale_percent / 100)
  height = int(img.shape[0] * scale_percent / 100)
  dim = (width, height)
  # resize image
  resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA) 
  ims[i] = resized/255.0
  #print('Resized Dimensions : ',resized.shape) 
  #plt.imshow(ims[i])
  #plt.show()

plt.imshow(ims[2])
#print(ims[2])

###
out = np.empty((total_sample, 5, int(img_tmp.shape[0]*scale_percent/100), int(img_tmp.shape[1]*scale_percent/100)), dtype=np.float32)

for i in range(total_sample):
  out_img = np.zeros((5, int(img_tmp.shape[0]*scale_percent/100), int(img_tmp.shape[1]*scale_percent/100)), dtype=np.float32)
  gr = metadata_sample.get_group(unique_file_names[i])
  for j in range(len(gr)):
    #print(int(gr.iloc[j].p3SY), int(gr.iloc[j].p3SX))
    if(gr.iloc[j].p1SY<0):
        #print(gr.iloc[j].p1SY)
        continue
    out_img[0, int(gr.iloc[j].p1SY), int(gr.iloc[j].p1SX)] = 255
    out_img[1, int(gr.iloc[j].p2SY), int(gr.iloc[j].p2SX)] = 255
    out_img[2, int(gr.iloc[j].p3SY), int(gr.iloc[j].p3SX)] = 255
    out_img[3, int(gr.iloc[j].p4SY), int(gr.iloc[j].p4SX)] = 255

    #out_img[54, 481] = 255
    #print(out_img[int(gr.iloc[j].p3SY), int(gr.iloc[j].p3SX)])
    
  out[i] = out_img/255
  out[i][0] = cv2.GaussianBlur(out[i][0], (5,5), 1.5)
  out[i][1] = cv2.GaussianBlur(out[i][1], (5,5), 1.5)
  out[i][2] = cv2.GaussianBlur(out[i][2], (5,5), 1.5)
  out[i][3] = cv2.GaussianBlur(out[i][3], (5,5), 1.5)

out[i].shape
plt.imshow(out[10][3])
plt.show()

#print(out_img)
#print(int(gr.iloc[0].p3SY))
#print(int(gr.iloc[0].p3SX))


###
#Save dataset
#np.save(data_path+"/dataset_ims_single.npy", ims)
#np.save(data_path+"/dataset_labels_single.npy", out)
#load dataset
ims = np.load(data_path+"/dataset_ims_single.npy", allow_pickle=True)[()]
out = np.load(data_path+"/dataset_labels_single.npy", allow_pickle=True)[()]


###
#plt.imshow(np.reshape(out[100], (-1, int(img_tmp.shape[1]*scale_percent/100))))
#plt.imshow(out[10])
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
 
  in_im2 =  in_im2.get().astype('f')
  #print(in_im)
  plt.imshow(in_im2)
  if(save_fig):
      if not os.path.exists(out_path):
        os.makedirs(out_path)
      plt.savefig(out_path+"/"+str(load_ind)+"_"+str(bt)+".png", dpi=150) 
  plt.show()

show_coordinates(ims[10], out[10], rad=7, threshold=0.02, save_fig=False, load_ind=0, bt_ind=0)


###
#sample = {'image': ims, 'labels': out}
#print(len(out))
#print(out[:4])
class pig_dataset(torch.utils.data.Dataset):

    def __init__(self, ims, out, transform=None):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.ims = ims
        self.out = out
        
        self.transform = transform

    def __len__(self):
        return len(self.ims)

    def __getitem__(self, idx):

        image =  torch.from_numpy(self.ims[idx]).unsqueeze(dim=0)
        labels =  torch.from_numpy(self.out[idx])
        #labels = labels.reshape(-1, 2)
        sample = (image, labels)

        return sample

data_set =  pig_dataset(ims, out)



train_samples = int(len(data_set)*70/100)
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
loader_train = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=ChunkSampler(train_samples, 0), num_workers=2)
loader_test = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=ChunkSampler(test_samples, train_samples), num_workers=2)
#tt = torch.utils.data.DataLoader(sample,batch_size=16,shuffle=True, num_workers=2)


###
print(len(data_set))
print('input shape: ', data_set[0][0].shape)
print('output shape: ', data_set[0][1].shape)

print(data_set[0][0].dtype)
print(data_set[0][1].dtype)
#for i_batch, (x, y) in enumerate(loader_test, 0):
#    print(i_batch, x, y)


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
        self.fc_mu      = nn.Linear(2*self.capacity*100*80, self.latent_dims)
        self.fc_logvar  = nn.Linear(2*self.capacity*100*80, self.latent_dims)
        self.fc_z       = nn.Linear(self.latent_dims, 2*capacity*100*80)

        # Decoder
        self.decoder = nn.Sequential(
            nn.BatchNorm2d(2*self.capacity),
            nn.ConvTranspose2d(2*self.capacity, self.capacity, kernel_size=(4,4), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.capacity), 
            nn.ConvTranspose2d(self.capacity, 4, kernel_size=(4,4), stride=2, padding=1),
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
        z = z.view(z.shape[0], 64, 100, 80)
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
def train(model,optimizer):
    model.train()
    epoch_loss = 0

    for i, (x, y) in enumerate(loader_train):
  
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
def test(model,optimizer):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(loader_test):
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


###
#modelVAE.load_state_dict(torch.load(data_path+"/model_cross_mean_single.pyc"), strict=False)


###
def train_and_evaluate_model(model=modelVAE, opt=optimizer, capacity=capacity, num_epochs=10):

    #criterion = nn.MSELoss()
    #optimizer = optim.Adam(net.parameters(), lr=0.1, betas=(0.9, 0.99))

    trainL, testL, epochs = [], [], []
    for e in range(1,num_epochs):
      
        train_loss = train(model,optimizer)
        train_loss /= train_samples
        test_loss = test(model,optimizer)
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
model_128 = train_and_evaluate_model(num_epochs=5000)
model_128
torch.save(model_128.state_dict(), data_path+"/model_cross_mean_single.pyc")


###    
modelVAE.eval()
data_iterator = iter(loader_test)
images, labels = data_iterator.next()

test_data=10

with torch.no_grad():
    y_hat, mu, logvar = modelVAE(images.to(device))

    print(y_hat[test_data].shape)
    print(images[test_data].squeeze(dim=0).numpy().shape, labels[test_data].numpy().shape)
    #print(ims[350].shape, out[350].shape)
    show_coordinates(images[test_data].squeeze(dim=0).numpy(), y_hat[test_data].cpu().numpy(), rad=7, threshold=0.0001)

    plt.plot(y_hat[test_data].cpu().numpy()[0])


###
modelVAE.eval()
cnt = 0
with torch.no_grad():
    for i, (x, y) in enumerate(loader_test):
        y_hat, mu, logvar = modelVAE(x.to(device))
        
        for bt in range(len(x)):
            #print(x[bt].squeeze(dim=0).numpy().shape, y[bt].numpy().shape)
            show_coordinates(x[bt].squeeze(dim=0).numpy(), y_hat[bt].cpu().numpy(), rad=7, threshold=0.0005, 
                             save_fig=True, load_ind=cnt, bt_ind=bt)
            #plt.plot(y_hat[bt].cpu().numpy()[0])
            cnt = cnt + 1

            







