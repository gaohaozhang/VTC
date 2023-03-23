import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
import h5py
data_path = 'E:\Gitproject\TransUnet/rose2_haricot/'

# val = 'E:/jiD/23_3_9/data/Synapse/train_npz/'
te_p = 'E:\Gitproject\TransUnet\data/Synapse/val.h5/'
# os.makedirs(tr_p,exist_ok=True)
os.makedirs(te_p,exist_ok=True)
fi = os.listdir(data_path)
l = len(fi)
# print(l)
# print(fi)
# tr =[]
te = []
for i in range(100):
    te.append(fi[i])
# for i in range(100,l):
#     te.append(fi[i])

#
# for i in tr:
#     img_p = data_path+i+'/'+'false.png'
#     gt_p = data_path+i+'/'+ 'gt.png'
#
#     im = cv2.imread(img_p)[:,:,0]
#     gt = cv2.imread(gt_p)[:,:,0]
#     gt2=cv2.imread(gt_p)[:,:,1]
#     im1 = im / 255
#
#     # im1 = im1.transpose(2,0,1)
#     # print(im1.shape,gt.shape)
#     gt1 = gt / 255
#     na = tr_p+i+'.npz'
#     np.savez(na,image=im1, label=gt1)

for i in te:
    img_p = data_path+i+'/'+'false.png'
    gt_p = data_path+i+'/'+ 'gt.png'

    im = cv2.imread(img_p)[:,:,0]
    gt = cv2.imread(gt_p)[:,:,0]
    im1 = im / 255

    # im1 = im1.transpose(2,0,1)
    # print(im1.shape,gt.shape)
    gt1 = gt / 255
    na = te_p+i+'.h5'
    with h5py.File(na, 'w') as f:
        f.create_dataset('image', data=im1)
        f.create_dataset('label', data=gt1)
        f.close()
    # np.savez(na,image=im1, label=gt1)






# x = os.walk(data_path)
# for i in x:
#     print(i)
#print(files)
# for root, dirs, files in os.walk(data_path):
#     print(root)

