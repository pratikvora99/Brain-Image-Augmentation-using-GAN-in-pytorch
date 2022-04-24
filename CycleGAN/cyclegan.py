import torchsummary
from torch import nn
from torch.nn.utils import spectral_norm
from cyclegan_model import GeneratorG,GeneratorF,Discriminator

import time
import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
from tqdm import tqdm

import pytorch_ssim
from multiprocessing import Process
import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook
from time import time
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.autograd import Variable
import random
from torch.nn.utils import spectral_norm
from scipy.stats import truncnorm
import torch as th
from torchvision import transforms

class BRATSDataSet(Dataset):
  
  def __init__(self,path,transform=transforms.ToTensor(),types=['T1/','T2/','T1CE/','FLAIR/']):
    self.path = path
    self.type = types
    self.names = os.listdir(path+'T1/')
    self.transforms = transform
    if len(self.names)==0:
      raise RuntimeError("Found 0 files in {}".format(path))
    
  def __getitem__(self,idx):
    
    if torch.is_tensor(idx):
            idx = idx.tolist()
       
    t1 = self.transforms(Image.open(self.path+self.type[0]+self.names[idx]))
    t2 =  self.transforms(Image.open(self.path+self.type[1]+self.names[idx]))
    t1ce =  self.transforms(Image.open(self.path+self.type[2]+self.names[idx]))
    flair =  self.transforms(Image.open(self.path+self.type[3]+self.names[idx]))
    
    return {'T1':t1,'T2':t2,'T1CE':t1ce,'FLAIR':flair}
  
  def __len__(self):
    return len(self.names)

train_path = 'Preprocessed/Train/'
valid_path = 'Preprocessed/Valid/'

bs = 20
g_lr = 1e-3
f_lr = 5e-4
dx_lr = 1e-6
dy_lr = 1e-6
lambda1 = 0.3
lambda2 = 0.3

train_dataset = BRATSDataSet(train_path)
valid_dataset = BRATSDataSet(valid_path)

data_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,batch_size=bs)

a = next(iter(data_loader))
for i in a:
   a[i] = a[i][0].numpy().transpose(1,2,0).reshape(240,240)

fig,axs = plt.subplots(1,4,figsize=(10,10))
for i,j in enumerate(a):
  axs[i].imshow(a[j],cmap='Greys_r')
  axs[i].set_title(j)

device="cuda"

G = GeneratorG(1).to(device)
F = GeneratorF(1).to(device)
DX = Discriminator(3).to(device)
DY = Discriminator(1).to(device)

epochs = 10
d_update = 1
g_update = 1


def LSLoss(y,yhat):
  return torch.mean((y-yhat)**2)



l2 = nn.MSELoss().to(device)
l1 = nn.L1Loss().to(device)
ssim_loss = pytorch_ssim.SSIM()
ce = nn.CrossEntropyLoss()

def SSIM_NEW(y,yhat):
  x = ssim_loss(y,yhat)
  return -torch.log((1.0+x)/2.0)

optimizerG = optim.Adam(G.parameters(), lr=g_lr, betas=(0.9, 0.999))
optimizerF = optim.Adam(F.parameters(), lr=f_lr, betas=(0.9, 0.999))
optimizerDX = optim.Adam(DX.parameters(), lr=dx_lr, betas=(0.9, 0.999))
optimizerDY = optim.Adam(DY.parameters(), lr=dy_lr, betas=(0.9, 0.999))

for epoch in range(epochs):
  step = 0  
  for ii,(imgs) in tqdm(enumerate(data_loader),total=len(data_loader)):
    
    a = imgs['T1'].to(device)
    b = imgs['T2'].to(device)
    c = imgs['T1CE'].to(device)
    d = imgs['FLAIR'].to(device)

    X = torch.cat([a,b,c],dim=1).to(device)
    y = torch.tensor(d,device=device)
    
    labelsX1 = torch.full((X.size(0), X.size(1), 1), 1, device=device) + np.random.uniform(-0.1, 0.1)
    labelsX0 = torch.full((X.size(0), X.size(1), 1), 0, device=device) + np.random.uniform(0.0, 0.2)
    labelsy1 = torch.full((y.size(0), y.size(1), 1), 1, device=device) + np.random.uniform(-0.1, 0.1)
    labelsy0 = torch.full((y.size(0), y.size(1), 1), 0, device=device) + np.random.uniform(0.0, 0.2)
    
    
    DY.zero_grad()   
    DX.zero_grad()
    
    yhat = G(X)
    Xhat = F(y)
    
    
    D_X = DX(X)
    D_Xhat = DX(Xhat)
    
    DX_err = LSLoss(D_X,labelsX1) + LSLoss(D_Xhat,labelsX0)
    DX_err.to(device)
    DX_err.backward(retain_graph=True)
    optimizerDX.step()
    
    D_X_mean = D_Xhat.mean().item()
    
    D_Y = DY(y)
    D_Yhat = DY(yhat)
    
    DY_err = LSLoss(D_Y,labelsy1) + LSLoss(D_Yhat,labelsy0)
    DY_err.to(device)
    DY_err.backward(retain_graph=True)
    optimizerDY.step()
    
    G.zero_grad()
    F.zero_grad()

    D_Y_mean = D_Yhat.mean().item()
    
    G_err = LSLoss(D_Xhat,labelsX1)
    
    F_err = LSLoss(D_Yhat,labelsy1)
    
    Xhatt = F(yhat)
    yhatt = G(Xhat)
    
    cycle_loss = 2*l1(X,Xhatt) + 2*l1(y,yhatt) + SSIM_NEW(y,yhatt) + SSIM_NEW(X,Xhatt) + 2*(2*l1(X,Xhat) + 2*l1(y,yhat) + SSIM_NEW(y,yhat) + SSIM_NEW(X,Xhat))
    G_err2 = lambda1 * cycle_loss + G_err 
    G_err2.backward(retain_graph=True)
    
    optimizerG.step()

    F.zero_grad()

    F_err2 = lambda2 * cycle_loss + F_err
    F_err2.backward(retain_graph = True)

    optimizerF.step()
    
    if(step%10 == 0):
      print('\nEpoch: '+str(epoch)+' Step: ' + str(step))
      print('DX_error: ' + str(DX_err.mean().item()))
      print('DY_error: ' + str(DY_err.mean().item()))
      print('G_error: ' + str(G_err.mean().item()))
      print('F_error: ' + str(F_err.mean().item()))
      print('DX_mean: ' + str(D_X_mean))
      print('DY_mean: ' + str(D_Y_mean))
      
      plt.figure(figsize=(10,10))
      fake = yhat.to("cpu").clone().detach()
      fake2 = Xhat.to("cpu").clone().detach()
      fake3 = yhatt.to("cpu").clone().detach()
      fake4 = Xhatt.to("cpu").clone().detach()
      
      plt.subplot(3,4,5)
      plt.imshow(fake.numpy().transpose(0,2,3,1)[0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,6)
      plt.imshow(fake2.numpy().transpose(0,2,3,1)[0,:,:,0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,7)
      plt.imshow(fake2.numpy().transpose(0,2,3,1)[0,:,:,1].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,8)
      plt.imshow(fake2.numpy().transpose(0,2,3,1)[0,:,:,2].reshape(240,240),cmap='Greys_r')
      
      plt.subplot(3,4,9)
      plt.imshow(fake3.numpy().transpose(0,2,3,1)[0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,10)
      plt.imshow(fake4.numpy().transpose(0,2,3,1)[0,:,:,0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,11)
      plt.imshow(fake4.numpy().transpose(0,2,3,1)[0,:,:,1].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,12)
      plt.imshow(fake4.numpy().transpose(0,2,3,1)[0,:,:,2].reshape(240,240),cmap='Greys_r')
      
      plt.subplot(3,4,2)
      plt.imshow(a.to("cpu").clone().detach().numpy().transpose(0,2,3,1)[0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,3)
      plt.imshow(b.to("cpu").clone().detach().numpy().transpose(0,2,3,1)[0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,4)
      plt.imshow(c.to("cpu").clone().detach().numpy().transpose(0,2,3,1)[0].reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,1)
      plt.imshow(d.to("cpu").clone().detach().numpy().transpose(0,2,3,1)[0].reshape(240,240),cmap='Greys_r')
      
      plt.savefig('cyclegan_figures.png')
    
    torch.save(G,'./G_final.pt')
    torch.save(F,'./F_final.pt')
    torch.save(DX,'./DX_final.pt')    
    torch.save(DY,'./DY_final.pt')
    
    step += 1

