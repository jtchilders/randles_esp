import glob,os
import pandas as pd
import numpy as np


input_path = '/projects/multiphysics_aesp/aymanzyy/FullCSVs'
output_path = '/projects/datascience/parton/randles_data_sampled'
filelist = glob.glob(input_path + '/*combined_csv*.csv')
nfiles = len(filelist)
print(' found ',nfiles, ' files')

np.random.shuffle(filelist)


def random_subsample(data,n_sub_imgs):
   
   img_size = data.shape[0]
   sub_img_size = int(img_size / n_sub_imgs)
   
   return data[random_sub_img:img_size:n_sub_imgs,:]


for i,filename in enumerate(filelist):
   print('processing file ',i,' of ',nfiles,' files, filename: ',filename)
   data = pd.read_csv(filename)
   fn_no_path = os.path.basename(filename)
   output_fn = os.path.join(output_path,fn_no_path)

   mout = data[data['8'] > 0][['1','2','3']]
   pv_data = data[data['8'] <= 0][['1','2','3','5','6','7']]

   # ensure we have all the data
   assert len(data) == len(mout) + len(pv_data)

   # convert to numpy and data type
   mout = mout.to_numpy().astype(np.float32)
   pv_data = pv_data.to_numpy().astype(np.float32)

   img_size = mout.shape[0]
   n_sub_imgs = 100
   sub_img_size = int(img_size / n_sub_imgs)
   for i in range(n_sub_imgs):
      sub_img = mout[i:img_size:n_sub_imgs,:]
      np.savez_compressed(output_fn.replace('.csv','_mout_%05d.csv.gz' %i),sub_img)
   
   img_size = pv_data.shape[0]
   n_sub_imgs = 1000
   sub_img_size = int(img_size / n_sub_imgs)
   for i in range(n_sub_imgs):
      sub_img = pv_data[i:img_size:n_sub_imgs,:]
      np.savez_compressed(output_fn.replace('.csv','_mout_%05d.csv.gz' %i),sub_img)