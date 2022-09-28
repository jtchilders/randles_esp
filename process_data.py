import glob,os,sys
import pandas as pd
import numpy as np
import mpi4py.MPI as MPI
np.random.seed(124)
mpi = MPI.COMM_WORLD

def get_filelist(unique_filelist,input_path):
   filelist = []
   for unique_filename in unique_filelist:
      filelist = filelist + glob.glob(os.path.join(input_path,unique_filename) + '*.csv')
      # print('rank','%04d' % mpi.Get_rank(),'ufn:',unique_filename,'len:',len(filelist))
   return sorted(filelist)

def get_stride(npoints,output_size):
   return int(npoints / output_size)

def subsample(data,output_size,start_point):
   
   npoints = data.shape[0]

   sample_stride = get_stride(npoints,output_size)
   
   end_point = int(output_size*sample_stride + start_point)

   out = data[start_point:end_point:sample_stride,:]   
   return out

def create_subimgs(input,output_size,output_fn):
   for i in range(get_stride(len(input),output_size)):
      sub_img = subsample(input,output_size,i)
      np.savez_compressed(output_fn.replace('.csv','_%05d.csv.gz' %i),data=sub_img)


def process_filelist(filelist,output_path):
   nfiles = len(filelist)
   nranks = mpi.Get_size()
   rank = mpi.Get_rank()
   if rank == 0:
      os.makedirs(output_path,exist_ok=True)
   mpi.barrier()
   for i in range(rank,nfiles,nranks):
      filename = filelist[i]
      print('rank','%04d' % rank,' processing file ',i,' of ',nfiles,' files, filename: ',filename)
      sys.stdout.flush()
      data = pd.read_csv(filename)
      fn_no_path = os.path.basename(filename)
      output_fn = os.path.join(output_path,fn_no_path)

      mout = data[data['8'] > 0][['1','2','3']]
      pv_data = data[data['8'] <= 0][['1','2','3','5','6','7']]

      # ensure we have all the data
      if(len(data) != len(mout) + len(pv_data)):
         print(rank,' found file with mismatched sizes: ',len(data),' != ',len(mout),' + ',len(pv_data))

      # convert to numpy and data type
      mout = mout.to_numpy().astype(np.float32)
      pv_data = pv_data.to_numpy().astype(np.float32)

      create_subimgs(mout,1e4,output_fn.replace('.csv','_mout.csv'))
      create_subimgs(pv_data,1e5,output_fn.replace('.csv','_pv_data.csv'))

      if i > 10:
         break


def main():
   output_path = '/lus/eagle/projects/multiphysics_aesp/data/FullCSVs_sampled'
   input_path = '/lus/eagle/projects/multiphysics_aesp/data/FullCSVs'

   # if mpi.Get_rank() == 0:
   filelist = glob.glob(input_path + '/*combined_csv*.csv')
   nfiles = len(filelist)
   print(' found ',nfiles, ' files')


   # get uniquely named files
   search_string = 'combined_csv'
   unique_filelist = [ x[:x.find(search_string)] for x in filelist ]
   unique_filelist = sorted(list(set(unique_filelist)))
   print(' found ',len(unique_filelist), ' unique filenames')

   # shuffle list
   np.random.shuffle(unique_filelist)

   # make lists:
   n_train = int(len(unique_filelist) * 0.70)
   n_valid = int(len(unique_filelist) * 0.15)
   n_test  = int(len(unique_filelist) * 0.15)
   print('rank','%04d' % mpi.Get_rank(),'indices:',n_train,n_valid,n_test)
   train_ufl = unique_filelist[:n_train]
   valid_ufl = unique_filelist[n_train:n_train+n_valid]
   test_ufl  = unique_filelist[n_train+n_valid:]


   print('rank','%04d' % mpi.Get_rank(),len(train_ufl),len(valid_ufl),len(test_ufl))
   # print('train_ufl:',train_ufl)
   # print('valid_ufl:',valid_ufl)
   # print('test_ufl:',test_ufl)

   train_fl = get_filelist(train_ufl,input_path)
   valid_fl = get_filelist(valid_ufl,input_path)
   test_fl = get_filelist(test_ufl,input_path)
   # else:
   #    train_fl = None
   #    valid_fl = None
   #    test_fl = None
   # mpi.bcast(train_fl,root=0)
   # mpi.bcast(valid_fl,root=0)
   # mpi.bcast(test_fl,root=0)

   print('rank','%04d' % mpi.Get_rank(),'train/valid/test fl:',len(train_fl),len(valid_fl),len(test_fl))
   sys.stdout.flush()
         


   process_filelist(train_fl,os.path.join(output_path,'train'))
   process_filelist(valid_fl,os.path.join(output_path,'valid'))
   process_filelist(test_fl,os.path.join(output_path,'test'))


if __name__ == '__main__':
   main()