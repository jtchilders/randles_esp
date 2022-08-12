import glob
import pandas as pd
import numpy as np
import tensorflow as tf
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_path = '/projects/multiphysics_aesp/aymanzyy/FullCSVs'
filelist = glob.glob(data_path + '/*combined_csv*.csv')
print(len(filelist))



def simple_dataset_from_filelist(shell_filelist,batch_size,shard_size=1,shard_rank=0,prefectch_buffer_size=10000,shuffle_buffer=10000,num_parallel_calls=None):

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


def wrapped_loader(path):
    path = path.numpy().decode('utf-8')

    data = pd.read_csv(path)

    mout = data[data['8'] > 0][['1','2','3']]
    pv_data = data[data['8'] == 0][['1','2','3','5','6','7']]

    # ensure we have all the data
    assert len(data) == len(mout) + len(pv_data)

    # convert data type and to TF tensor
    mout = tf.convert_to_tensor(mout.to_numpy().astype(np.float32))
    pv_data = tf.convert_to_tensor(pv_data.to_numpy().astype(np.float32))

    # add a batch dimension
    mout = tf.expand_dims(mout,0)
    pv_data = tf.expand_dims(pv_data,0)

    print(path,mout.shape,pv_data.shape)

    # could do some preprocessing here
    return (mout,pv_data)

class PointNet(tf.keras.Model):
    def __init__(self):
        super(PointNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(3,activation='relu');
        self.fc2 = tf.keras.layers.Dense(64,activation='relu');
        # self.fc3 = keras.layers.Dense(64,activation='relu');
        self.fc_feature = tf.keras.layers.Dense(512,activation='relu');
        self.maxpooling = tf.keras.layers.GlobalMaxPool1D()
        self.fc4 = tf.keras.layers.Dense(128,activation='relu');
        self.fc5 = tf.keras.layers.Dense(256,activation='relu');
        self.drop1 = tf.keras.layers.Dropout(0.1);
        self.drop3 = tf.keras.layers.Dropout(0.3);
        self.logist = tf.keras.layers.Dense(3,activation='linear');

    def call(self, inputs):
        mout,pvout = inputs         # 3e8 * 2 (2 inputs) * 32 bits
        ## proc mdata
        mout = self.fc1(mout)        # 3e8 * 2 (weight & bias) * 32 bits
        mout = self.fc2(mout)         # 64e8 * 2 * 32 -> gradients -> supporting other things that we don't know
        # mout = self.fc3(mout)
        mout = self.fc_feature(mout)  # 512e8 * 2 * 32
        mout = tf.expand_dims(mout, 0)
        mout = self.maxpooling(mout)

        ## proc pvdata
        pvout = self.fc1(pvout)      # 3e8 * 2 * 32
        pvout = self.fc2(pvout)       # 64e8 * 2 * 32
        # pvout = self.fc3(pvout)
        pvout = self.fc4(pvout)       # 512e8 * 2 * 32
        ##
        mout = tf.broadcast_to(mout,[pvout.shape[0],mout.shape[1]])
        out = tf.concat([pvout,mout],axis=-1)
        # out = pvout+mout
        # out = self.drop3(out)
        out = self.fc5(out)           # 256e8 * 2 * 32
        # out = self.drop1(out)
        out = self.logist(out)        # 3e8 * 2 * 32

        return out


def train_step(mx,pv,y,model,optimizer,criterion,choice_num=10000):
    print(mx.shape,pv.shape,y.shape)
    idx = np.arange(pv.shape[0])
    idx = np.random.choice(idx,choice_num,replace=False)
    pv = pv.numpy()[idx,:]
    y  =  y.numpy()[idx,:]

    print(idx.shape,pv.shape)
    with tf.GradientTape() as tape:
        logits = model([mx,pv],training=True)
        loss = tf.reduce_mean(criterion(y, logits))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


training_ds = simple_dataset_from_filelist(filelist,1)

model = PointNet()
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
criterion = tf.keras.losses.MAE

#model.summary()

for step,(mx,pv) in enumerate(training_ds):

    mx = tf.squeeze(mx)
    pv = tf.squeeze(pv)

    loss = train_step(mx,pv[:,0:3],pv[:,3:6],model,optimizer,criterion)

    print("[%05d] loss = %f" % (step,loss))
