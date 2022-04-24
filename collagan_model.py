import torch
import torchsummary
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import InstanceNorm2d as BN

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
    
    
class CCAM(nn.Module):
    
    def __init__(self,in_ch):
        super(CCAM,self).__init__()
        
        self.denses = nn.Sequential(
                        *[nn.Linear(in_ch+4,in_ch//4),
                        nn.LeakyReLU(),
                        nn.Linear(in_ch//4,in_ch//4),                         
                        nn.LeakyReLU(),
                        nn.Linear(in_ch//4,in_ch),
                        nn.Sigmoid()]
                        )
        
    def forward(self,im,mask):
        o1 = self.denses(torch.cat([torch.mean(im,dim=[2,3],keepdim=False),mask],dim=1))
        return torch.mul(im,o1.view(o1.shape[0],o1.shape[1],1,1))
    
    
class Generator(nn.Module):
    
    def __init__(self,in_ch,use_bias,st_ch=4):
        super(Generator, self).__init__()
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
        
        
        #Encoder D layers
        self.enc_d0_0 = CCNR(in_ch,st_ch,use_bias)
        self.enc_d0_1 = CCNR(st_ch,st_ch,use_bias)
        self.pool_d0 = nn.Conv2d(st_ch, st_ch*2, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_d1_0 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.enc_d1_1 = CCNR(st_ch*2,st_ch*2,use_bias)
        self.pool_d1 = nn.Conv2d(st_ch*2, st_ch*4, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_d2_0 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.enc_d2_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.pool_d2 = nn.Conv2d(st_ch*4, st_ch*8, kernel_size=2, stride=2, bias=use_bias)
        
        
        self.enc_d3_0 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.enc_d3_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.pool_d3 = nn.Conv2d(st_ch*8, st_ch*16, kernel_size=2, stride=2, bias=use_bias)
        
        #Decode Layers
        
        self.dec_3_0 = CCNR(st_ch*64,st_ch*64,use_bias)
        self.dec_3_1 = CCNR(st_ch*64,st_ch*64,use_bias)
        self.dec_3_2 = CCAM(st_ch*64)
        self.convT_3 = nn.ConvTranspose2d(st_ch*64,st_ch*32,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_2_0 = CCNR(st_ch*64,st_ch*32,use_bias)
        self.dec_2_1 = CCNR(st_ch*32,st_ch*32,use_bias)
        self.dec_2_2 = CCAM(st_ch*32)
        self.convT_2 = nn.ConvTranspose2d(st_ch*32,st_ch*16,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_1_0 = CCNR(st_ch*32,st_ch*16,use_bias)
        self.dec_1_1 = CCNR(st_ch*16,st_ch*16,use_bias)
        self.dec_1_2 = CCAM(st_ch*16)
        self.convT_1 = nn.ConvTranspose2d(st_ch*16,st_ch*8,kernel_size=2,stride=2,bias=use_bias)
        
        
        self.dec_0_0 = CCNR(st_ch*16,st_ch*8,use_bias)
        self.dec_0_1 = CCNR(st_ch*8,st_ch*8,use_bias)
        self.dec_0_2 = CCAM(st_ch*8)
        self.convT_0 = nn.ConvTranspose2d(st_ch*8,st_ch*4,kernel_size=2,stride=2,bias=use_bias)
        
        self.final_dec_0 = CCNR(st_ch*4,st_ch*4,use_bias)
        self.final_dec_1 = CCNR(st_ch*4,st_ch*4,use_bias)
        
        self.final_conv = nn.Conv2d(st_ch*4,1,kernel_size=1,stride=1,bias=use_bias)
        
        
    def forward(self,inputs):
      
        mask = inputs[:,4:,:,:]
        
        a = torch.cat([inputs[:,0:1,:,:],mask],dim=1)
        b = torch.cat([inputs[:,1:2,:,:],mask],dim=1)
        c = torch.cat([inputs[:,2:3,:,:],mask],dim=1)
        d = torch.cat([inputs[:,3:4,:,:],mask],dim=1)
        
        mask = mask[:,:,0,0]
        
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
        
        down_d0 = self.enc_d0_1(self.enc_d0_0(d))
        pooled_d0 = self.pool_d0(down_d0)
        down_d1 = self.enc_d1_1(self.enc_d1_0(pooled_d0))
        pooled_d1 = self.pool_d1(down_d1)
        down_d2 = self.enc_d2_1(self.enc_d2_0(pooled_d1))
        pooled_d2 = self.pool_d2(down_d2)
        down_d3 = self.enc_d3_1(self.enc_d3_0(pooled_d2))
        pooled_d3 = self.pool_d3(down_d3)
        
        
        up_3 = self.dec_3_0(torch.cat([pooled_a3,pooled_b3,pooled_c3,pooled_d3],dim=1))
        up_3 = self.dec_3_2(self.dec_3_1(up_3),mask)
        up_3 = self.convT_3(up_3)        
        
        
        up_2 = self.dec_2_0(torch.cat([up_3,pooled_a2,pooled_b2,pooled_c2,pooled_d2],dim=1))
        up_2 = self.dec_2_2(self.dec_2_1(up_2),mask)
        up_2 = self.convT_2(up_2)
        
        up_1 = self.dec_1_0(torch.cat([up_2,pooled_a1,pooled_b1,pooled_c1,pooled_d1],dim=1))
        up_1 = self.dec_1_2(self.dec_1_1(up_1),mask)
        up_1 = self.convT_1(up_1)
        
        
        up_0 = self.dec_0_0(torch.cat([up_1,pooled_a0,pooled_b0,pooled_c0,pooled_d0],dim=1))
        up_0 = self.dec_0_2(self.dec_0_1(up_0),mask)
        up_0 = self.convT_0(up_0)
        
        fin_img = self.final_dec_1(self.final_dec_0(up_0))
        
        return nn.ReLU()(self.final_conv(fin_img))

class Discriminator(nn.Module):
  
  def __init__(self,use_bias):
    
      super(Discriminator,self).__init__()
      self.use_bias = use_bias
      self.path1 = nn.Sequential(*[
          nn.Conv2d(1,4,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          nn.Conv2d(4,4,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          nn.Conv2d(4,4,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          nn.Conv2d(4,16,kernel_size=4,stride=4,bias=False)   
      ])
      
      self.path2 = nn.Sequential(*[
          nn.Conv2d(1,4,kernel_size=3,stride=1,bias=use_bias,padding=2),
          nn.MaxPool2d(2,2),          
          nn.Conv2d(4,8,kernel_size=3,stride=1,bias=use_bias,padding=1),
          nn.MaxPool2d(2,2),
          nn.Conv2d(8,16,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU()      
      ])
      
      self.path3 = nn.Sequential(*[
          nn.Conv2d(1,16,kernel_size=4,stride=4,bias=False,padding=1),
          nn.Conv2d(16,16,kernel_size=3,stride=1,bias=use_bias,padding=2),
          nn.LeakyReLU(),
          nn.Conv2d(16,16,kernel_size=3,stride=1,bias=use_bias),
          nn.LeakyReLU(),
          nn.Conv2d(16,16,kernel_size=3,stride=1,bias=use_bias)
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
          nn.Conv2d(128,1,kernel_size=5,stride=1,bias=False),
          nn.Sigmoid()
      ])
      
      self.classify = nn.Sequential(*[
          nn.Conv2d(64,128,kernel_size=4,stride=2,bias=False),
          nn.LeakyReLU(),
          nn.Dropout(0.5),
          nn.Conv2d(128,4,kernel_size=5,stride=1,bias=False),
          nn.Softmax()
      ])
      
  def forward(self,img):
    
    o1 = self.path1(img)
    o2 = self.path2(img)
    o3 = self.path3(img)
    merge_out = self.merge(torch.cat([o1,o2,o3],dim=1))
    rf_out = self.rf(merge_out)
    class_out = self.classify(merge_out)
    
    return rf_out,class_out

