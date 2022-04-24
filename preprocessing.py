import matplotlib.pyplot as plt
import nibabel as nib
from PIL import Image
from tqdm import tqdm

import os
import numpy as np

os.listdir('./Dataset/MICCAI_BraTS_2019_Data_Training/LGG/')

types = ['LGG/','HGG/']
paths = []
path = './Dataset/MICCAI_BraTS_2019_Data_Training/'
for i in types:
  p = os.listdir(path+i)
  paths+=[path+i+x+'/'+x for x in p]

im_types = ['_t1.nii.gz','_t1ce.nii.gz','_t2.nii.gz','_flair.nii.gz']
file_map = {'_t1.nii.gz':'T1/','_t1ce.nii.gz':'T1CE/','_t2.nii.gz':'T2/','_flair.nii.gz':'FLAIR/'}
slices = [55,90,100,110]

fig,axes = plt.subplots(1,len(slices),figsize=(10,10))

for j,i in enumerate(slices):
  print(j)
  axes[j].imshow(im.get_fdata()[:,:,i],cmap='Greys_r')

slices = range(slices[0],slices[1])

from random import shuffle
ids = list(range(len(paths)))
shuffle(ids)
valid_ids = ids[:len(ids)//5]
train_ids = ids[len(ids)//5:]

save_path = './Dataset/Preprocessed/'

paths = np.array(paths)

for im_type in im_types:
  for path in tqdm(paths[train_ids]):
    img = nib.load(path+im_type)
    for s in slices:
      Image.fromarray(img.get_fdata()[:,:,s]).save(save_path+'Train/'+file_map[im_type]+path[path.rfind('/')+1:]+'_'+str(s)+'.tiff')

for im_type in im_types:
  for path in tqdm(paths[valid_ids]):
    img = nib.load(path+im_type)
    for s in slices:
      Image.fromarray(img.get_fdata()[:,:,s]).convert('L').save(save_path+'/Valid/'+file_map[im_type]+path[path.rfind('/')+1:]+'_'+str(s)+'.png')

len(os.listdir(save_path+'Valid/T1')),len(os.listdir(save_path+'Valid/T1CE')),len(os.listdir(save_path+'Valid/T2')),len(os.listdir(save_path+'Valid/FLAIR'))

len(os.listdir(save_path+'Train/T1')),len(os.listdir(save_path+'Train/T1CE')),len(os.listdir(save_path+'Train/T2')),len(os.listdir(save_path+'Train/FLAIR'))