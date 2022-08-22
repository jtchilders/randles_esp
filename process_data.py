import glob,os
import pandas as pd
import numpy as np


output_path = '/lus/eagle/projects/datascience/parton/randles_data_sampled'
input_path = '/lus/eagle/projects/multiphysics_aesp/data/FullCSVs'
filelist = glob.glob(input_path + '/00cts?l*combined_csv*.csv')
nfiles = len(filelist)
print(' found ',nfiles, ' files')

np.random.shuffle(filelist)

def get_stride(npoints,output_size):
   return int(npoints / output_size)

def subsample(data,output_size,start_point):
   
   npoints = data.shape[0]

   sample_stride = get_stride(npoints,output_size)
   
   end_point = int(output_size*sample_stride + start_point)

   out = data[start_point:end_point:sample_stride,:]

   # print(hvd.rank(),'in shape:',data.shape,'out shape:',out.shape,'sample_stride:',sample_stride,'start_point:',start_point)
   # sys.stdout.flush()
   
   return out

def create_subimgs(input,output_size,output_fn):
   for i in range(get_stride(len(input),output_size)):
      sub_img = subsample(input,output_size,i)
      np.savez_compressed(output_fn.replace('.csv','_%05d.csv.gz' %i),data=sub_img)

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

   create_subimgs(mout,1e4,output_fn.replace('.csv','_mout.csv'))
   
   create_subimgs(pv_data,1e5,output_fn.replace('.csv','_pv_data.csv'))