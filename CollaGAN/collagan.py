import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
from tqdm import tqdm_notebook

import os
import numpy as np
import pytorch_ssim
from collagan_model import Generator,Discriminator

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm_notebook as tqdm
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

def LSLoss(y,yhat):
  return torch.mean((y-yhat)**2).to(device)

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

bs = 8
gen_lr = 1e-5
dis_lr = 1e-3
lambda_gen_ls = 0.5
lambda_l1_cyc = 10
lambda_l1 = 1
lambda_l2_cyc = 10
lambda_l2 = 0
lambda_ce_gen = 15
lambda_ssim = 1

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

device = "cuda"

retrain = True

if retrain:
  Gen = torch.load('./generator.pth').to(device)  
  Dis = torch.load('./discriminator.pth').to(device)
else:
  Gen = Generator(5,True).to(device)
  Dis = Discriminator(True).to(device)

optimizer_g = optim.Adam(Gen.parameters(), lr=gen_lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(Dis.parameters(), lr=dis_lr, betas=(0.5, 0.999))

l2 = nn.MSELoss()
l1 = nn.L1Loss()
ssim_loss = pytorch_ssim.SSIM()
ce = nn.CrossEntropyLoss()

ssim_lambda = lambda x,y,r: -torch.log((1.0+x)/2.0) if y!=r else torch.zeros_like(x,device=device)
l1_lambda = lambda x,y,r: x if y!=r else torch.zeros_like(x,device=device)
l2_lambda = lambda x,y,r: x if y!=r else torch.zeros_like(x,device=device)

real = torch.ones((32,1),device=device)
fake = torch.zeros((32,1),device=device)

epochs = 10

for epoch in tqdm_notebook(range(epochs)):
  print('aaa')
  for i,imgs in tqdm_notebook(enumerate(data_loader),total=len(data_loader)):

    r = np.random.randint(0,4)
    mask = torch.zeros((bs,4,240,240),device=device)
    mask[:,r:r+1,:,:]=1
    mask_bool = [False for i in range(4)]
    mask_bool[r] = True
    a = imgs['T1'].to(device)
    b = imgs['T2'].to(device)
    c = imgs['T1CE'].to(device)
    d = imgs['FLAIR'].to(device)

    in_imgs = torch.cat([a,b,c,d,mask],dim=1)
    target = torch.tensor(in_imgs[:,r:r+1,:,:])
    in_imgs[:,r:r+1,:,:]=0
    recon = Gen(in_imgs)
    rf_recon,class_recon = Dis(recon)
    rf_target,class_target = Dis(target)   
    
    
    mask_0 = torch.zeros((bs,4,240,240),device=device)
    mask_0[:,0:1,:,:]=1
    mask_1 = torch.zeros((bs,4,240,240),device=device)
    mask_1[:,1:2,:,:]=1
    mask_2 = torch.zeros((bs,4,240,240),device=device)
    mask_2[:,2:3,:,:]=1
    mask_3 = torch.zeros((bs,4,240,240),device=device)
    mask_3[:,3:4,:,:]=1
    
    cyc_in_imgs_0 = torch.cat([a,b,c,d,mask_0],dim=1)
    cyc_in_imgs_0[:,r:r+1,:,:]=recon
    cyc_in_imgs_0[:,0:1,:,:]=torch.zeros((bs,1,240,240),device=device)
    
    
    cyc_in_imgs_1 = torch.cat([a,b,c,d,mask_1],dim=1)
    cyc_in_imgs_1[:,r:r+1,:,:]=recon
    cyc_in_imgs_1[:,1:2,:,:]=torch.zeros((bs,1,240,240),device=device)
    
    
    cyc_in_imgs_2 = torch.cat([a,b,c,d,mask_2],dim=1)
    cyc_in_imgs_2[:,r:r+1,:,:]=recon
    cyc_in_imgs_2[:,2:3,:,:]=torch.zeros((bs,1,240,240),device=device)
    
    
    cyc_in_imgs_3 = torch.cat([a,b,c,d,mask_3],dim=1)
    cyc_in_imgs_3[:,r:r+1,:,:]=recon
    cyc_in_imgs_3[:,3:4,:,:]=torch.zeros((bs,1,240,240),device=device)
    
    cyc_0 = Gen(cyc_in_imgs_0)
    cyc_1 = Gen(cyc_in_imgs_1)
    cyc_2 = Gen(cyc_in_imgs_2)
    cyc_3 = Gen(cyc_in_imgs_3)
    
    rf_cyc_0,class_cyc_0 = Dis(cyc_0)
    rf_cyc_1,class_cyc_1 = Dis(cyc_1)
    rf_cyc_2,class_cyc_2 = Dis(cyc_2)
    rf_cyc_3,class_cyc_3 = Dis(cyc_3)
    
    rf_tar_0,class_tar_0 = Dis(a)
    rf_tar_1,class_tar_1 = Dis(b)
    rf_tar_2,class_tar_2 = Dis(c)
    rf_tar_3,class_tar_3 = Dis(d)
    
    
    l2_recon = l2(recon,target)
    l2_cyc_0 = l2(cyc_0,a)
    l2_cyc_1 = l2(cyc_1,b)
    l2_cyc_2 = l2(cyc_2,c)
    l2_cyc_3 = l2(cyc_3,d)
    l2_cyc_0 = l2_lambda(l2_cyc_0,0,r)
    l2_cyc_1 = l2_lambda(l2_cyc_1,1,r)
    l2_cyc_2 = l2_lambda(l2_cyc_2,2,r)
    l2_cyc_3 = l2_lambda(l2_cyc_3,3,r)
    l2_cyc = l2_cyc_0+l2_cyc_1+l2_cyc_2+l2_cyc_3

    l1_recon = l1(recon,target)
    l1_cyc_0 = l1(cyc_0,a)
    l1_cyc_1 = l1(cyc_1,b)
    l1_cyc_2 = l1(cyc_2,c)
    l1_cyc_3 = l1(cyc_3,d)
    
    l1_cyc_0 = l1_lambda(l1_cyc_0,0,r)
    l1_cyc_1 = l1_lambda(l1_cyc_1,1,r)
    l1_cyc_2 = l1_lambda(l1_cyc_2,2,r)
    l1_cyc_3 = l1_lambda(l1_cyc_3,3,r)
    l1_cyc = l1_cyc_0+l1_cyc_1+l1_cyc_2+l1_cyc_3
    
    ssim_cyc_0 = ssim_loss(cyc_0,a)
    ssim_cyc_1 = ssim_loss(cyc_1,b)
    ssim_cyc_2 = ssim_loss(cyc_2,c)
    ssim_cyc_3 = ssim_loss(cyc_3,d)
    
    ssim_cyc_0 = ssim_lambda(ssim_cyc_0,0,r)
    ssim_cyc_1 = ssim_lambda(ssim_cyc_1,1,r)
    ssim_cyc_2 = ssim_lambda(ssim_cyc_2,2,r)
    ssim_cyc_3 = ssim_lambda(ssim_cyc_3,3,r)
    ssim_cyc = ssim_cyc_0+ssim_cyc_1+ssim_cyc_2+ssim_cyc_3
    
    ls_recon =  LSLoss(rf_recon,real)
    ls_cyc_0 = LSLoss(rf_cyc_0,real)
    ls_cyc_1 = LSLoss(rf_cyc_1,real)
    ls_cyc_2 = LSLoss(rf_cyc_2,real)
    ls_cyc_3 = LSLoss(rf_cyc_3,real)
    ls_gen = ls_recon+ls_cyc_0+ls_cyc_1+ls_cyc_2+ls_cyc_3
    
    ce_recon = ce(class_recon.view(bs,-1),torch.full((bs,),r,device=device).to(dtype=torch.long))
    ce_cyc_0 = ce(class_cyc_0.view(bs,-1),torch.full((bs,),0,device=device).to(dtype=torch.long))
    ce_cyc_1 = ce(class_cyc_1.view(bs,-1),torch.full((bs,),1,device=device).to(dtype=torch.long))
    ce_cyc_2 = ce(class_cyc_2.view(bs,-1),torch.full((bs,),2,device=device).to(dtype=torch.long))
    ce_cyc_3 = ce(class_cyc_3.view(bs,-1),torch.full((bs,),3,device=device).to(dtype=torch.long))
    ce_gen = ce_recon+ce_cyc_0+ce_cyc_1+ce_cyc_2+ce_cyc_3
    
    gen_loss = (lambda_gen_ls*ls_gen + lambda_l1_cyc*l1_cyc + lambda_l1*l1_recon 
                + lambda_l2_cyc*l2_cyc + lambda_l2*l2_recon
                + lambda_ce_gen*ce_gen + lambda_ssim*ssim_cyc)
    
    ce_tar_0 = ce(class_tar_0.view(bs,-1),torch.full((bs,),0,device=device).to(dtype=torch.long))
    ce_tar_1 = ce(class_tar_1.view(bs,-1),torch.full((bs,),1,device=device).to(dtype=torch.long))
    ce_tar_2 = ce(class_tar_2.view(bs,-1),torch.full((bs,),2,device=device).to(dtype=torch.long))
    ce_tar_3 = ce(class_tar_3.view(bs,-1),torch.full((bs,),3,device=device).to(dtype=torch.long))
    
    ls_target = LSLoss(rf_target,real)
    ls_recon = LSLoss(rf_recon,fake)
    
    ls_tar_0 = LSLoss(rf_tar_0,real)
    ls_cyc_0 = LSLoss(rf_cyc_0,fake)
    ls_tar_1 = LSLoss(rf_tar_1,real)
    ls_cyc_1 = LSLoss(rf_cyc_1,fake)
    ls_tar_2 = LSLoss(rf_tar_2,real)
    ls_cyc_2 = LSLoss(rf_cyc_2,fake)
    ls_tar_3 = LSLoss(rf_tar_3,real)
    ls_cyc_3 = LSLoss(rf_cyc_3,fake)
    
    ls_dis =  (ls_target+ ls_recon + ls_tar_0 + ls_tar_1 + ls_tar_2 + ls_tar_3
               + ls_cyc_0 + ls_cyc_1 + ls_cyc_2 + ls_cyc_3)
    dis_loss = (ls_dis)/5 + (ce_tar_0 + ce_tar_1 + ce_tar_2+ ce_tar_3)
    
    Gen.zero_grad()
    Dis.zero_grad()
    
    gen_loss.backward(retain_graph=True)
    optimizer_g.step()
    dis_loss.backward()
    optimizer_d.step()
    
    if i%500==499 or i==0:
      print(f'Step:{i},R={r}')
      print(f'DLoss:{ls_dis.item()/2}')
      print(f'GLoss:{ls_gen.item()}')
      plt_recon = recon[0].to("cpu").clone().detach()
      plt_tar = target[0].to("cpu").clone().detach()
      plt.figure(figsize=(10, 10))
      plt.subplot(3,4,1)
      plt.title('Recon')
      plt.imshow(plt_recon.numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,2)
      plt.title('Target')
      plt.imshow(plt_tar.numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,5)
      plt.title('a')
      plt.imshow(a[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,6)
      plt.title('b')
      plt.imshow(b[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,7)
      plt.title('c')
      plt.imshow(c[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,8)
      plt.title('d')
      plt.imshow(d[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,9)
      plt.title('cyc_a')
      plt.imshow(cyc_0[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,10)
      plt.title('cyc_b')
      plt.imshow(cyc_1[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,11)
      plt.title('cyc_c')
      plt.imshow(cyc_2[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.subplot(3,4,12)
      plt.title('cyc_d')
      plt.imshow(cyc_3[0].to("cpu").clone().detach().numpy().reshape(240,240),cmap='Greys_r')
      plt.show()
  print('Saving Model')
  torch.save(Gen,'./generator.pth')
  torch.save(Dis,'./discriminator.pth')

l = []
a = next(iter(data_loader))

mask_0 = torch.zeros((bs,4,240,240),device=device)
mask_0[:,0:1,:,:]=1
mask_1 = torch.zeros((bs,4,240,240),device=device)
mask_1[:,1:2,:,:]=1
mask_2 = torch.zeros((bs,4,240,240),device=device)
mask_2[:,2:3,:,:]=1
mask_3 = torch.zeros((bs,4,240,240),device=device)
mask_3[:,3:4,:,:]=1

t1 = a['T1'].to(device)
t2 = a['T2'].to(device)
t1ce = a['T1CE'].to(device)
flair = a['FLAIR'].to(device)

dummy = torch.zeros((bs,1,240,240),device=device)

t1_recon = Gen(torch.cat([dummy,t2,t1ce,flair,mask_0],dim=1))
t2_recon = Gen(torch.cat([t1,dummy,t1ce,flair,mask_1],dim=1))
t1ce_recon = Gen(torch.cat([t1,t2,dummy,flair,mask_2],dim=1))
flair_recon = Gen(torch.cat([t1,t2,t1ce,dummy,mask_3],dim=1))

plt.figure(figsize=(10, 10))
plt.subplot(2,4,1)
plt.imshow(t1[0].cpu().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('T1')
plt.subplot(2,4,2)
plt.imshow(t2[0].cpu().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('T2')
plt.subplot(2,4,3)
plt.imshow(t1ce[0].cpu().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('T1CE')
plt.subplot(2,4,4)
plt.imshow(flair[0].cpu().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('FLAIR')
plt.subplot(2,4,5)
plt.imshow(t1_recon[0].cpu().detach().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('T1_recon')
plt.subplot(2,4,6)
plt.imshow(t2_recon[0].cpu().detach().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('T2_recon')
plt.subplot(2,4,7)
plt.imshow(t1ce_recon[0].cpu().detach().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('T1CE_recon')
plt.subplot(2,4,8)
plt.imshow(flair_recon[0].cpu().detach().numpy().transpose(1,2,0).reshape(240,240),cmap='Greys_r')
plt.title('FLAIR_recon');

