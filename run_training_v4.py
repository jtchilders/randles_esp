import glob,sys
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import horovod.tensorflow as hvd
import socket
from mpi4py import MPI

def simple_dataset_from_filelist(shell_filelist,batch_size,shard_size=1,shard_rank=0,prefectch_buffer_size=4,shuffle_buffer=10000,num_parallel_calls=8):

   # glob for the input files
   filelist = tf.data.Dataset.from_tensor_slices(np.array(shell_filelist))
   # shuffle and repeat at the input file level
   filelist = filelist.shuffle(shuffle_buffer)
   # filelist = filelist.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=10000,count=config['training']['epochs']))
   # map to read files in parallel
   ds = filelist.map(load_file_and_preprocess,num_parallel_calls=num_parallel_calls)

   # flatten the inputs across file boundaries
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))

   # speficy batch size
   ds = ds.batch(batch_size,drop_remainder=True)

   # shard the data
   ds = ds.shard(shard_size,shard_rank)

   # how many inputs to prefetch to improve pipeline performance
   ds = ds.prefetch(buffer_size=prefectch_buffer_size)

   return ds


def load_file_and_preprocess(path):
   pyf = tf.py_function(wrapped_loader,[path],(tf.float32,tf.float32))
   return pyf

def random_subsample(data,output_size):
   
   npoints = data.shape[0]

   sample_stride = int(npoints / output_size)
   
   start_point = np.random.choice(sample_stride)
   end_point = int(output_size*sample_stride + start_point)

   out = data[start_point:end_point:sample_stride,:]

   # print(hvd.rank(),'in shape:',data.shape,'out shape:',out.shape,'sample_stride:',sample_stride,'start_point:',start_point)
   # sys.stdout.flush()
   
   return out

def get_mout_fn(pv_fn):
   outfn = pv_fn.replace('pv_data','mout')
   outfl = glob.glob(outfn)
   if len(outfl) > 0:
      return outfl[0]
   
   # get filenumber
   # 00cts6lcombined_csv_10s95_mout_00110.csv.gz.npz
   search_string = '_mout_'
   start = outfn.find(search_string) + len(search_string)
   end = outfn.find('.csv.gz.npz',start)
   str_length = end - start
   file_num = int( int(outfn[start:end]) / 2 )

   new_file_num = np.random.randint(file_num)
   outfn = outfn[:start] + (("%%0%dd" % str_length) % new_file_num) + outfn[end:]
   # print('new fn: ',outfn)
   outfl = glob.glob(outfn)
   while len(outfl) == 0:
      new_file_num = np.random.randint(file_num)
      outfn[start:end] = outfn[:start] + (("%%0%dd" % str_length) % new_file_num) + outfn[end:]
      # print('new fn: ',outfn)
      outfl = glob.glob(outfn)
   
   return outfl[0]
   

def wrapped_loader(path):
   path = path.numpy().decode('utf-8')

   try:
      pv_data = np.load(path)
      pv_data = pv_data['data']
   except:
      print('failed to parse pv_out: ',path)
      raise
   
   # 00cts6lcombined_csv_10s95_pv_data_00110.csv.gz.npz
   # need to see if there is a corresponding 
   path = get_mout_fn(path)
   try:
      mout = np.load(path)
      mout = mout['data']
   except:
      print('failed to parse mout: ',path)
      raise
   
   # convert data type and to TF tensor
   mout = tf.convert_to_tensor(mout)
   pv_data = tf.convert_to_tensor(pv_data)
   
   # add a batch dimension
   mout = tf.expand_dims(mout,0)
   pv_data = tf.expand_dims(pv_data,0)
   
   # print(path,mout.shape,pv_data.shape)

   # could do some preprocessing here
   return (mout,pv_data)

class PointNet(tf.keras.Model):
   def __init__(self):
      super(PointNet, self).__init__()
      self.fc1 = tf.keras.layers.Dense(3,activation='relu',name='fc1');
      self.fc2 = tf.keras.layers.Dense(64,activation='relu',name='fc2');
      self.fc3 = tf.keras.layers.Dense(512,activation='relu',name='fc3');
      self.maxpooling = tf.keras.layers.GlobalMaxPool1D()
      self.fc4 = tf.keras.layers.Dense(128,activation='relu',name='fc4');
      self.fc5 = tf.keras.layers.Dense(256,activation='relu',name='fc5');
      self.fc6 = tf.keras.layers.Dense(3,activation='linear',name='fc6');

   def call(self, inputs):
      mout,pvout = inputs              # 3e8 * 2 (2 inputs) * 32 bits
      ## proc mdata
      mout = self.fc1(mout)            # 3e8 * 2 (weight & bias) * 32 bits
      mout = self.fc2(mout)            # 64e8 * 2 * 32 -> gradients -> supporting other things that we don't know
      mout = self.fc3(mout)            # 512e8 * 2 * 32
      mout = tf.expand_dims(mout, 0)
      mout = self.maxpooling(mout)

      ## proc pvdata
      pvout = self.fc1(pvout)          # 3e8 * 2 * 32
      pvout = self.fc2(pvout)          # 64e8 * 2 * 32
      pvout = self.fc4(pvout)          # 512e8 * 2 * 32
      ##
      mout = tf.broadcast_to(mout,[pvout.shape[0],mout.shape[1]])
      out = tf.concat([pvout,mout],axis=-1)
      out = self.fc5(out)              # 256e8 * 2 * 32
      out = self.fc6(out)              # 3e8 * 2 * 32

      return out


@tf.function()
def train_step(mx,pv,y,model,optimizer,criterion,first_batch):

   with tf.GradientTape() as tape:
      logits = model([mx,pv],training=True)
      loss = tf.reduce_mean(criterion(y, logits))
   
   # Horovod: add Horovod Distributed GradientTape.
   tape = hvd.DistributedGradientTape(tape)
   
   grads = tape.gradient(loss, model.trainable_variables)
   optimizer.apply_gradients(zip(grads, model.trainable_variables))
   
   # Horovod: broadcast initial variable states from rank 0 to all other processes.
   # This is necessary to ensure consistent initialization of all workers when
   # training is started with random weights or restored from a checkpoint.
   #
   # Note: broadcast should be done after the first gradient step to ensure optimizer
   # initialization.
   if first_batch:
      hvd.broadcast_variables(model.variables, root_rank=0)
      hvd.broadcast_variables(optimizer.variables(), root_rank=0)

   return loss,logits

@tf.function
def test_step(mx,pv,y,model,criterion):
   logits = model([mx,pv],training=False)
   loss = tf.reduce_mean(criterion(y, logits))
   return loss,logits

def main():

   profile = False
   stop_early = False
   stop_step = 10
   if profile:
      stop_early = True
   hvd.init()
   print('NUM MPI RANKS: %4d  HOST: %20s  LOCAL_RANK:  %4d    GLOBAL_RANK:  %4d' % (hvd.size(),socket.gethostname(),hvd.local_rank(),hvd.rank()))
   sys.stdout.flush()

   if hvd.rank() == 0:
      print('tensorflow version: ',tf.__version__)

   # Horovod: pin GPU to be used to process local rank (one GPU per process)
   gpus = tf.config.experimental.list_physical_devices('GPU')
   for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
   if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
   print('RANK:',hvd.rank(),'GPUS: ',tf.config.experimental.get_visible_devices('GPU'))

   data_path = '/lus/eagle/projects/multiphysics_aesp/data/FullCSVs_sampled/'
   filelist = glob.glob(data_path + '*_pv_data_*.csv.gz.npz')
   if hvd.rank() == 0:
      print(' found ',len(filelist), ' files')
   if len(filelist) == 0: sys.exit(-1)

   nfiles = len(filelist)
   np.random.shuffle(filelist)
   training_ds = simple_dataset_from_filelist(filelist[:int(nfiles*0.8)],1,shard_size=hvd.size(),shard_rank=hvd.rank())
   testing_ds = simple_dataset_from_filelist(filelist[int(nfiles*0.8):],1)

   model = PointNet()
   optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
   criterion = tf.keras.losses.MAE

   train_loss = tf.keras.metrics.Mean('train_loss')
   train_acc  = tf.keras.metrics.MeanSquaredError('train_accuracy')
   train_nmae = tf.keras.metrics.Mean('train_nmae')
   test_loss  = tf.keras.metrics.Mean('test_loss')
   test_acc   = tf.keras.metrics.MeanSquaredError('test_accuracy')
   test_nmae = tf.keras.metrics.Mean('test_nmae')

   if hvd.rank() == 0:
      current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
      logdir = 'logs/gradient_tape/' + current_time
      print('logdir: ',logdir)
      train_log_dir = logdir + '/train'
      test_log_dir = logdir + '/test'
      
      train_summary_writer = tf.summary.create_file_writer(train_log_dir)
      test_summary_writer = tf.summary.create_file_writer(test_log_dir)

   start = datetime.datetime.now()
   first_batch = True
   if profile and hvd.rank() == 0:
      print('profiling')
      tf.profiler.experimental.start(logdir)
   total_steps = 0
   for i_epoch in range(100):
      print("[%02d] rank %02d starting epoch" % (i_epoch,hvd.rank()))
      sys.stdout.flush()
      
      # train loop
      for step,(mx,pv) in enumerate(training_ds):
         if i_epoch == 0:
            total_steps += 1

         mx = tf.squeeze(mx)
         pv = tf.squeeze(pv)

         loss,logits = train_step(mx,pv[:,0:3],pv[:,3:6],model,optimizer,criterion,first_batch)


         npoints = pv.get_shape().as_list()[0]
         a = tf.reduce_sum(tf.abs(pv[:,3:6] - logits),axis=0)
         b = tf.reduce_max(pv[:,3:6],axis=0) - tf.reduce_min(pv[:,3:6],axis=0)
         c = a / b
         nmae = tf.reduce_mean(c/ npoints )

         train_loss(loss)
         train_acc(logits,pv[:,3:6])
         train_nmae(nmae)
         end = datetime.datetime.now()
         dur = end - start

         if(hvd.rank() == 0 and step % 100 == 0):
            print("[%02d][%04d] time = %s train_loss: %f  train_acc: %f" % (i_epoch,step,str(dur),train_loss.result(),train_acc.result()))
            sys.stdout.flush()
            if(first_batch):
               model.summary()
            with train_summary_writer.as_default():
               tf.summary.scalar('loss', train_loss.result(), step=i_epoch*total_steps + step)
               tf.summary.scalar('accuracy', train_acc.result(), step=i_epoch*total_steps + step)
               tf.summary.scalar('NMAE', train_nmae.result(), step=i_epoch*total_steps + step)
         
         first_batch = False
         if stop_early and step >= stop_step:
            if profile and hvd.rank() == 0:
               print('profiler exiting')
               tf.profiler.experimental.stop()
            break
      
      if stop_early:
         break

      # test loop
      
      print("[%02d] rank %02d done training" % (i_epoch,hvd.rank()))
      sys.stdout.flush()
      if hvd.rank() == 0:
         for step,(mx,pv) in enumerate(testing_ds):
            mx = tf.squeeze(mx)
            pv = tf.squeeze(pv)

            loss,logits = test_step(mx,pv[:,0:3],pv[:,3:6],model,criterion)

            npoints = pv.get_shape().as_list()[0]
            a = tf.reduce_sum(tf.abs(pv[:,3:6] - logits),axis=0)
            b = tf.reduce_max(pv[:,3:6],axis=0) - tf.reduce_min(pv[:,3:6],axis=0)
            c = a / b
            nmae = tf.reduce_mean(c/ npoints )
            test_loss(loss)
            test_acc(logits,pv[:,3:6])
            test_nmae(nmae)
            end = datetime.datetime.now()
            dur = end - start
            if(step % 100 == 0):
               log_step = (i_epoch+1)*total_steps
               print("[%02d][%04d] time = %s test_loss: %f  test_acc: %f" % (i_epoch,step,str(dur),test_loss.result(),test_acc.result()))
               sys.stdout.flush()
               with test_summary_writer.as_default():
                  tf.summary.scalar('loss', test_loss.result(), step=log_step)
                  tf.summary.scalar('accuracy', test_acc.result(), step=log_step)
                  tf.summary.scalar('NMAE', test_nmae.result(), step=log_step)
         
         template = 'Epoch {}, Loss: {}, Accuracy: {}, NMAE: {}, Test Loss: {}, Test Accuracy: {}, Test NMAE: {}'
         print (template.format(i_epoch+1,
                              train_loss.result(), 
                              train_acc.result(),
                              train_nmae.result(),
                              test_loss.result(), 
                              test_acc.result(), 
                              test_nmae.result()))
      
      print("[%02d] rank %02d done testing" % (i_epoch,hvd.rank()))
      sys.stdout.flush()



if __name__ == '__main__':
   main()

   print('EXITING PROGRAM   GLOBAL_RANK:  %4d' % (hvd.rank()))
