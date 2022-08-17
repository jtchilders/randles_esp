import glob
import pandas as pd
import numpy as np
import pptk

data_path = '/Users/jchilders/randlesESP/data'
filelist = glob.glob(data_path + '/*combined_csv*.csv')
print('Found ',len(filelist), 'files')
print('filelist: ')
for i,filename in enumerate(filelist):
      print(i,'.   ',filename)


def sub_sample(filelist):

   data = pd.read_csv(filelist[0])
   data = data[['1','2','3']].to_numpy()

   img_size = len(data)
   n_sub_imgs = 1000
   sub_img_size = int(img_size / n_sub_imgs)
   for i in range(n_sub_imgs):
      sub_img = data[i*sub_img_size:(i+1)*sub_img_size,:]
      print('   number of points: ',len(sub_img))
      v = pptk.viewer(sub_img)
      v.set(point_size=0.01)
      print('waiting...')
      v.wait()
      #input('hit enter')
      #v.()

def full_image(filelist):

   for filename in filelist:
      print('plotting filename: ',filename)
      data = pd.read_csv(filename)
      print('   number of points: ',len(data))
      data = data[['1','2','3']].to_numpy()

      v = pptk.viewer(data)
      v.set(point_size=0.001)
      print('waiting...')
      v.wait()

sub_sample(filelist)