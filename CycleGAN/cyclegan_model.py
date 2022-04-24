import torch
import torchsummary
from torch import nn
from torch.nn.utils import spectral_norm

import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
from tqdm import tqdm

import os
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import InstanceNorm2d as BN
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

device="cuda"

class CCNR(nn.Module):
    
    def __init__(self,in_ch,out_ch,use_bias):
        super(CCNR, self).__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch//2, kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.conv3x3 = nn.Conv2d(in_ch, out_ch//2, kernel_size=3, stride=1, padding=1, bias=use_bias)
        self.lr = nn.LeakyReLU()
        self.bn = BN(out_ch)
    
    def forward(self,im):
        
        o1 = self.conv1x1(im)
        o2 = self.conv3x3(im)
        o3 = self.lr(torch.cat([o1,o2],dim=1))
        return self.bn(o3)

class GeneratorF(nn.Module):
    def __init__(self,in_ch,use_bias=True,st_ch=4):
        super(GeneratorF, self).__init__()
        self.in_ch = in_ch        
        
        #Encode layers
        self.enc_a0_0 = CCNR(in_ch,st_ch,use_bias)
        self.enc_a0_1 = CCNR(st_ch,st_ch,use_bias)
        self.pool_a0 = nn.Conv2d(st_ch, st_ch*2, kernel_size=2, stride=2, bias=use_bias)
    
        
        self.enc_a1_0 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.enc_a1_1 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.pool_a1 = nn.Conv2d(st_ch*2, st_ch*4, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_a2_0 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.enc_a2_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.pool_a2 = nn.Conv2d(st_ch*4, st_ch*8, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_a3_0 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.enc_a3_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.pool_a3 = nn.Conv2d(st_ch*8, st_ch*16, kernel_size=2, stride=2, bias=use_bias)

	#Decode Layers
        
        self.dec_3_0 = CCNR(st_ch*16,st_ch*16,use_bias)
        self.dec_3_1 = CCNR(st_ch*16,st_ch*16,use_bias)
        # self.dec_3_2 = CCAM(st_ch*16)
        self.convT_3 = nn.ConvTranspose2d(st_ch*16,st_ch*8,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_2_0 = CCNR(st_ch*16,st_ch*8,use_bias)
        self.dec_2_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        # self.dec_2_2 = CCAM(st_ch*8)
        self.convT_2 = nn.ConvTranspose2d(st_ch*8,st_ch*4,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_1_0 = CCNR(st_ch*8,st_ch*4,use_bias)
        self.dec_1_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        # self.dec_1_2 = CCAM(st_ch*4)
        self.convT_1 = nn.ConvTranspose2d(st_ch*4,st_ch*2,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_0_0 = CCNR(st_ch*4,st_ch*2,use_bias)
        self.dec_0_1 = CCNR(st_ch*2,st_ch*2,use_bias)
        # self.dec_0_2 = CCAM(st_ch*2)
        self.convT_0 = nn.ConvTranspose2d(st_ch*2,st_ch*1,kernel_size=2,stride=2,bias=use_bias)
        
        self.final_dec_0 = CCNR(st_ch*1,st_ch*1,use_bias)
        self.final_dec_1 = CCNR(st_ch*1,st_ch*1,use_bias)
        
        self.final_conv = nn.Conv2d(st_ch*1,3,kernel_size=1,stride=1,padding=0,bias=use_bias)

    def forward(self,a):
        down_a0 = self.enc_a0_1(self.enc_a0_0(a))
        pooled_a0 = self.pool_a0(down_a0)
        down_a1 = self.enc_a1_1(self.enc_a1_0(pooled_a0))
        pooled_a1 = self.pool_a1(down_a1)
        down_a2 = self.enc_a2_1(self.enc_a2_0(pooled_a1))
        pooled_a2 = self.pool_a2(down_a2)
        down_a3 = self.enc_a3_1(self.enc_a3_0(pooled_a2))
        pooled_a3 = self.pool_a3(down_a3)
        up_3 = self.dec_3_0(pooled_a3)
        up_3 = self.dec_3_1(up_3)
        up_3 = self.convT_3(up_3)        
        
        
        up_2 = self.dec_2_0(torch.cat([up_3,pooled_a2],dim=1))
        up_2 = self.dec_2_1(up_2)
        up_2 = self.convT_2(up_2)
        
        up_1 = self.dec_1_0(torch.cat([up_2,pooled_a1],dim=1))
        up_1 = self.dec_1_1(up_1)
        up_1 = self.convT_1(up_1)
        
        
        up_0 = self.dec_0_0(torch.cat([up_1,pooled_a0],dim=1))
        up_0 = self.dec_0_1(up_0)
        up_0 = self.convT_0(up_0)
        
        fin_img = self.final_dec_1(self.final_dec_0(up_0))
        
        return nn.ReLU()(self.final_conv(fin_img))
       
class GeneratorG(nn.Module):
    
    def __init__(self,in_ch,use_bias=True,st_ch=4):
        super(GeneratorG, self).__init__()
        self.in_ch = in_ch        
        
        #Encoder A layers
        self.enc_a0_0 = CCNR(in_ch,st_ch,use_bias)
        self.enc_a0_1 = CCNR(st_ch,st_ch,use_bias)
        self.pool_a0 = nn.Conv2d(st_ch, st_ch*2, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_a1_0 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.enc_a1_1 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.pool_a1 = nn.Conv2d(st_ch*2, st_ch*4, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_a2_0 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.enc_a2_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.pool_a2 = nn.Conv2d(st_ch*4, st_ch*8, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_a3_0 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.enc_a3_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.pool_a3 = nn.Conv2d(st_ch*8, st_ch*16, kernel_size=2, stride=2, bias=use_bias)
        
        #Encoder B layers
        self.enc_b0_0 = CCNR(in_ch,st_ch,use_bias)
        self.enc_b0_1 = CCNR(st_ch,st_ch,use_bias)
        self.pool_b0 = nn.Conv2d(st_ch, st_ch*2, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_b1_0 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.enc_b1_1 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.pool_b1 = nn.Conv2d(st_ch*2, st_ch*4, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_b2_0 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.enc_b2_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.pool_b2 = nn.Conv2d(st_ch*4, st_ch*8, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_b3_0 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.enc_b3_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.pool_b3 = nn.Conv2d(st_ch*8, st_ch*16, kernel_size=2, stride=2, bias=use_bias)
        
        
        #Encoder C layers
        self.enc_c0_0 = CCNR(in_ch,st_ch,use_bias)
        self.enc_c0_1 = CCNR(st_ch,st_ch,use_bias)
        self.pool_c0 = nn.Conv2d(st_ch, st_ch*2, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_c1_0 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.enc_c1_1 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.pool_c1 = nn.Conv2d(st_ch*2, st_ch*4, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_c2_0 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.enc_c2_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.pool_c2 = nn.Conv2d(st_ch*4, st_ch*8, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_c3_0 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.enc_c3_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.pool_c3 = nn.Conv2d(st_ch*8, st_ch*16, kernel_size=2, stride=2, bias=use_bias)
        
        
        #Decode Layers
        
        self.dec_3_0 = CCNR(st_ch*48,st_ch*48,use_bias)
        self.dec_3_1 = CCNR(st_ch*48,st_ch*48,use_bias)
        # self.dec_3_2 = CCAM(st_ch*48)
        self.convT_3 = nn.ConvTranspose2d(st_ch*48,st_ch*24,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_2_0 = CCNR(st_ch*48,st_ch*24,use_bias)
        self.dec_2_1 = CCNR(st_ch*24,st_ch*24,use_bias)
        # self.dec_2_2 = CCAM(st_ch*24)
        self.convT_2 = nn.ConvTranspose2d(st_ch*24,st_ch*12,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_1_0 = CCNR(st_ch*24,st_ch*12,use_bias)
        self.dec_1_1 = CCNR(st_ch*12,st_ch*12,use_bias)
        # self.dec_1_2 = CCAM(st_ch*12)
        self.convT_1 = nn.ConvTranspose2d(st_ch*12,st_ch*6,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_0_0 = CCNR(st_ch*12,st_ch*6,use_bias)
        self.dec_0_1 = CCNR(st_ch*6,st_ch*6,use_bias)
        # self.dec_0_2 = CCAM(st_ch*6)
        self.convT_0 = nn.ConvTranspose2d(st_ch*6,st_ch*3,kernel_size=2,stride=2,bias=use_bias)
        
        self.final_dec_0 = CCNR(st_ch*3,st_ch*3,use_bias)
        self.final_dec_1 = CCNR(st_ch*3,st_ch*3,use_bias)
        
        self.final_conv = nn.Conv2d(st_ch*3,1,kernel_size=1,stride=1,padding=0,bias=use_bias)
        
        
    def forward(self,inputs):
      
        # mask = inputs[:,4:,:,:]
        
        # print(mask.shape,inputs[:,0:1,:,:].shape)
        
        a = inputs[:,0:1,:,:]
        b = inputs[:,1:2,:,:]
        c = inputs[:,2:3,:,:]
        # d = inputs[:,3:4,:,:]
        
        # mask = mask[:,:,0,0]
        
        down_a0 = self.enc_a0_1(self.enc_a0_0(a))
        pooled_a0 = self.pool_a0(down_a0)
        down_a1 = self.enc_a1_1(self.enc_a1_0(pooled_a0))
        pooled_a1 = self.pool_a1(down_a1)
        down_a2 = self.enc_a2_1(self.enc_a2_0(pooled_a1))
        pooled_a2 = self.pool_a2(down_a2)
        down_a3 = self.enc_a3_1(self.enc_a3_0(pooled_a2))
        pooled_a3 = self.pool_a3(down_a3)
        
        down_b0 = self.enc_b0_1(self.enc_b0_0(b))
        pooled_b0 = self.pool_b0(down_b0)
        down_b1 = self.enc_b1_1(self.enc_b1_0(pooled_b0))
        pooled_b1 = self.pool_b1(down_b1)
        down_b2 = self.enc_b2_1(self.enc_b2_0(pooled_b1))
        pooled_b2 = self.pool_b2(down_b2)
        down_b3 = self.enc_b3_1(self.enc_b3_0(pooled_b2))
        pooled_b3 = self.pool_b3(down_b3)
        
        down_c0 = self.enc_c0_1(self.enc_c0_0(c))
        pooled_c0 = self.pool_c0(down_c0)
        down_c1 = self.enc_c1_1(self.enc_c1_0(pooled_c0))
        pooled_c1 = self.pool_c1(down_c1)
        down_c2 = self.enc_c2_1(self.enc_c2_0(pooled_c1))
        pooled_c2 = self.pool_c2(down_c2)
        down_c3 = self.enc_c3_1(self.enc_c3_0(pooled_c2))
        pooled_c3 = self.pool_c3(down_c3)
        
       
        up_3 = self.dec_3_0(torch.cat([pooled_a3,pooled_b3,pooled_c3],dim=1))
        up_3 = self.dec_3_1(up_3)
        up_3 = self.convT_3(up_3)        
        
        
        up_2 = self.dec_2_0(torch.cat([up_3,pooled_a2,pooled_b2,pooled_c2],dim=1))
        up_2 = self.dec_2_1(up_2)
        up_2 = self.convT_2(up_2)
        
        up_1 = self.dec_1_0(torch.cat([up_2,pooled_a1,pooled_b1,pooled_c1],dim=1))
        up_1 = self.dec_1_1(up_1)
        up_1 = self.convT_1(up_1)
        
        
        up_0 = self.dec_0_0(torch.cat([up_1,pooled_a0,pooled_b0,pooled_c0],dim=1))
        up_0 = self.dec_0_1(up_0)
        up_0 = self.convT_0(up_0)
        
        fin_img = self.final_dec_1(self.final_dec_0(up_0))
        
        return nn.ReLU()(self.final_conv(fin_img))

class Discriminator(nn.Module):
  
  def __init__(self,in_ch,use_bias=True):
    
      super(Discriminator,self).__init__()
      self.use_bias = use_bias
      self.in_ch = in_ch

      self.path1 = nn.Sequential(*[
          nn.Conv2d(self.in_ch,4,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          nn.Conv2d(4,4,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          nn.Conv2d(4,4,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          spectral_norm(nn.Conv2d(4,16,kernel_size=4,stride=4,bias=False))   
      ])
      
      self.path2 = nn.Sequential(*[
          spectral_norm(nn.Conv2d(self.in_ch,4,kernel_size=3,stride=1,bias=use_bias,padding=2)),
          nn.MaxPool2d(2,2),          
          spectral_norm(nn.Conv2d(4,8,kernel_size=3,stride=1,bias=use_bias,padding=1)),
          nn.MaxPool2d(2,2),
          nn.Conv2d(8,16,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU() 
      ])
      
      self.path3 = nn.Sequential(*[
          spectral_norm(nn.Conv2d(self.in_ch,16,kernel_size=4,stride=4,bias=False,padding=1)),
          nn.Conv2d(16,16,kernel_size=3,stride=1,bias=use_bias,padding=2),
          nn.LeakyReLU(),
          nn.Conv2d(16,16,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          spectral_norm(nn.Conv2d(16,16,kernel_size=3,stride=1,bias=use_bias))
      ])
      
      
      self.merge = nn.Sequential(*[
          nn.Conv2d(16*3,32,kernel_size=4,stride=2,bias=False),
          nn.LeakyReLU(),
          nn.Conv2d(32,64,kernel_size=4,stride=2,bias=False),
          nn.LeakyReLU()
      ])
      
      self.rf = nn.Sequential(*[
          nn.Conv2d(64,128,kernel_size=4,stride=2,bias=False),
          nn.LeakyReLU(),
          nn.Dropout(0.5),
          nn.Conv2d(128,in_ch,kernel_size=5,stride=1,bias=False),
          nn.Sigmoid()
      ])
      
      
  def forward(self,img):
    
    o1 = self.path1(img)
    o2 = self.path2(img)
    o3 = self.path3(img)
    merge_out = self.merge(torch.cat([o1,o2,o3],dim=1))
    rf_out = self.rf(merge_out)
        
    return rf_out[:,:,:,0]
