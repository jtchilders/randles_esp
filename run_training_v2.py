import glob,sys
import pandas as pd
import numpy as np
import tensorflow as tf
import datetime
import horovod.tensorflow as hvd

profile = False
hvd.init()

if hvd.rank() == 0:
   print('tensorflow version: ',tf.__version__)

# Horovod: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
   tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
   tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def simple_dataset_from_filelist(shell_filelist,batch_size,shard_size=1,shard_rank=0,prefectch_buffer_size=1,shuffle_buffer=10000,num_parallel_calls=3):

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

def random_subsample(data,n_sub_imgs):
   
   img_size = data.shape[0]
   sub_img_size = int(img_size / n_sub_imgs)
   
   random_sub_img = np.random.choice(n_sub_imgs)
   
   return data[random_sub_img:img_size:n_sub_imgs,:]


def wrapped_loader(path):
   path = path.numpy().decode('utf-8')

   data = pd.read_csv(path)

   mout = data[data['8'] > 0][['1','2','3']]
   pv_data = data[data['8'] == 0][['1','2','3','5','6','7']]
   
   # ensure we have all the data
   assert len(data) == len(mout) + len(pv_data)
   
   # convert to numpy and data type
   mout = mout.to_numpy().astype(np.float32)
   pv_data = pv_data.to_numpy().astype(np.float32)
   
   # sub sample
   mout = random_subsample(mout,100)
   pv_data = random_subsample(pv_data,1000)
   
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
   logits = model([mx,pv],training=True)
   loss = tf.reduce_mean(criterion(y, logits))
   return loss,logits

data_path = '/projects/multiphysics_aesp/aymanzyy/FullCSVs'
filelist = glob.glob(data_path + '/*combined_csv*.csv')
if hvd.rank() == 0:
   print(' found ',len(filelist), ' files')


nfiles = len(filelist)
np.random.shuffle(filelist)
training_ds = simple_dataset_from_filelist(filelist[:int(nfiles*0.8)],1,shard_size=hvd.size(),shard_rank=hvd.rank())
testing_ds = simple_dataset_from_filelist(filelist[int(nfiles*0.8):],1)

model = PointNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
criterion = tf.keras.losses.MAE

train_loss = tf.keras.metrics.Mean('train_loss')
train_acc  = tf.keras.metrics.MeanSquaredError('train_accuracy')
test_loss  = tf.keras.metrics.Mean('test_loss')
test_acc   = tf.keras.metrics.MeanSquaredError('test_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/gradient_tape/' + current_time
print('logdir: ',logdir)
train_log_dir = logdir + '/train'
test_log_dir = logdir + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

start = datetime.datetime.now()
#model.summary()
first_batch = True
if profile:
   print('profiling')
   tf.profiler.experimental.start(logdir)
total_steps = 0
for i_epoch in range(10):

   # train loop
   for step,(mx,pv) in enumerate(training_ds):
      if i_epoch == 0:
         total_steps += 1

      mx = tf.squeeze(mx)
      pv = tf.squeeze(pv)

      loss,logits = train_step(mx,pv[:,0:3],pv[:,3:6],model,optimizer,criterion,first_batch)
      train_loss(loss)
      train_acc(logits,pv[:,3:6])
      end = datetime.datetime.now()
      dur = end - start
      if(hvd.rank() == 0):

         if(first_batch):
            model.summary()

         print("[%05d] time = %s loss = %f   %f   %f" % (i_epoch*total_steps + step,str(dur),loss,train_loss.result(),train_acc.result()))
         with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=i_epoch*total_steps + step)
            tf.summary.scalar('accuracy', train_acc.result(), step=i_epoch*total_steps + step)
      
      first_batch = False
      if step > 5 and profile:
         print('profiler exiting')
         tf.profiler.experimental.stop()
         sys.exit(0)

   # test loop
   for step,(mx,pv) in enumerate(testing_ds):
      mx = tf.squeeze(mx)
      pv = tf.squeeze(pv)

      loss,logits = test_step(mx,pv[:,0:3],pv[:,3:6],model,criterion)
      test_loss(loss)
      test_acc(logits,pv[:,3:6])
      end = datetime.datetime.now()
      dur = end - start
      if(hvd.rank() == 0):
         print("[%05d] time = %s loss = %f" % (step,str(dur),loss))
   with test_summary_writer.as_default():
      tf.summary.scalar('loss', test_loss.result(), step=i_epoch*total_steps)
      tf.summary.scalar('accuracy', test_acc.result(), step=i_epoch*total_steps)
   
   template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
   print (template.format(i_epoch+1,
                         train_loss.result(), 
                         train_acc.result()*100,
                         test_loss.result(), 
                         test_acc.result()*100))
