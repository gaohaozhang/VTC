import os

import numpy as np
import nibabel as nib
import imageio

import matplotlib.pyplot as plt

#需要修改下面的nii.gz文件的路径
niigz_path = 'E:\Gitproject\TransUnet\TransUNet\predictions\TU_Synapse224\TU_pretrain_ViT-B_16_skip3_epo10_bs24_224/'


#下面定义了保存图片的路径
save_png_path = niigz_path +'predict_png/'
os.makedirs(save_png_path,exist_ok=True)

file = os.listdir(niigz_path)
for f in file:
    na =f.split('_')
    if na[-1] == 'pred.nii.gz':
        #print(f)
        data = nib.load(niigz_path+f)
        data = data.get_fdata()
        data = data.transpose(-1, -2)
        # plt.imshow(data)
        # plt.show()
        h = data.shape[0]
        w = data.shape[1]
        # da = np.empty((h,w,3))
        data1 = np.where(data > 0, 255, data)

        da = np.zeros([h, w, 3])
        da[:, :, 2] = data1
        da = np.array(da, dtype=np.uint8)
        # plt.imshow(da)
        # plt.show()
        nam = na[0].split(".")[0]
        name = save_png_path+ nam +'_pred.png'
        plt.imsave(name, da)

