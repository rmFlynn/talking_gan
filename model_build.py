import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
# load all the layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D
# load model
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model, Model
# load back end
from tensorflow.keras import backend
# load optimizers
from tensorflow.keras.optimizers import Adam
# load other tensor stuff
import tensorflow as tf
# load other stuff stuff
import os
import numpy as np

def noise_enc():
    ne = None
    return ne

###Rory code

def context_enc():
    # one aproch
    # https://towardsdatascience.com/an-approach-towards-convolutional-recurrent-neural-networks-f54cbeecd4a6

    # vary setings
    shape_in = int(320), int(8), int(1)
    shape_out = int(8134), int(120)
    dropoutrate = 0.3

    x_start = Input(shape=(shape_in))
    x = x_start

    for _i, _cnt in enumerate((2, 2)):
        x = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(x)
        x = BatchNormalization(axis=-1)(x)
        #x = Activation('relu')(x)
        x = LeakyReLU()(x)
        #x = MaxPooling2D(pool_size=(2,2), dim_ordering="th" )(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropoutrate)(x)

    x = Permute((2, 1, 3))(x)
    x = Reshape((1, 16000))(x)

    # The Gru/recurrent portion
    # Get some knowledge
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    for r in (10,10):
        x = Bidirectional(
                GRU(r,
                    activation='tanh',
                    dropout=dropoutrate,
                    recurrent_dropout=dropoutrate,
                    return_sequences=True),
                merge_mode='concat')(x)
        for f in ((2,2)):
            x = TimeDistributed(Dense(f))(x)

    x = Dropout(dropoutrate)(x)
    x = TimeDistributed(Dense(880))(x)
    # arbitrary reshape may be a problem
    x = Reshape((22,40,1))(x)
    out = Activation('sigmoid', name='strong_out')(x)
    ce = out
    return ce, x_start

#UNET Functions

def down_block(x, filters, kernal_size=(3, 3), padding='same', strides=1):
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(x)
    c = LeakyReLU()(c)
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(c)
    c = LeakyReLU()(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernal_size=(3, 3), padding='same', strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(concat)
    c = LeakyReLU()(c)
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(c)
    c = LeakyReLU()(c)
    return c

def make_vid(data, name):
    data = np.concatenate([data, data, data], axis=3)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = 30
    shape = (data.shape[2], data.shape[1])
    out = cv2.VideoWriter(name + ".avi", fourcc, fps, shape)
    for f in data:
        out.write((f).astype('uint8'))
    out.release()


data = np.load('./video1.npy')
data = data / 255
data.shape

f = [8, 16, 32, 64, 128, 256]

inputs = Input((72, 128, 1))

# Down bloc encoding
c1, d = down_block(inputs, f[0])
c2, d = down_block(d, f[1])
c3, d = down_block(d, f[2])
d = ZeroPadding2D(((0, 1), (0, 0)))(d)
c4, d = down_block(d, f[3])
d = ZeroPadding2D(((0, 1), (0, 0)))(d)
c5, d = down_block(d, f[4])
d = Flatten()(d)

n = Dense(1536)(d)
n = Dense(1536)(n)
n = LeakyReLU()(n)
n = Reshape((3, 4, 128))(n)

# Up bloc decoding
u = up_block(n, c5, f[4])
u = Cropping2D(((0, 1), (0, 0)))(u)
u = up_block(u, c4, f[3])
u = Cropping2D(((0, 1), (0, 0)))(u)
u = up_block(u, c3, f[2])
u = up_block(u, c2, f[1])
u = up_block(u, c1, f[0])

u = Conv2D(1, (1, 1), padding='same')(u)


bn = bottleneck(p5, ae, ne, f[5])

#Up bloc decoding
u1 = up_block1(bn, c5, f[4])
u2 = up_block(u1, c4, f[3])
u3 = up_block(u2, c3, f[2])
u4 = up_block(u3, c2, f[1])
u5 = up_block(u4, c1, f[0])

#autoencoder egress layer. Flatten and any perceptron layers would succeed this layer
outputs = Conv2D(3, (1, 1), padding='same', activation = 'tanh')(u5)
#outputs = Conv2D(3, (1, 1), padding='same', activation = 'tanh')(bn)

#Keras model output
model = Model([inputs, inputs2], outputs, name='gener')
#model.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])

    return model, [inputs, inputs2]

def build_discriminator():
    # one aproch
    # https://towardsdatascience.com/an-approach-towards-convolutional-recurrent-neural-networks-f54cbeecd4a6

    # vary settings
    shape_in_x = int(320), int(8), int(1)
    shape_in_y = int(64), int(128), int(3)
    #shape_out = int(8134), int(120)
    dropoutrate = 0.3

    x_start = Input(shape=(shape_in_x))
    x = x_start

    for _i, _cnt in enumerate((2, 2)):
        x = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(x)
        x = BatchNormalization(axis=-1)(x)
        #x = Activation('relu')(x)
        x = LeakyReLU()(x)
        #x = MaxPooling2D(pool_size=(2,2), dim_ordering="th" )(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropoutrate)(x)

    x = Flatten()(x) # shape out = none 16000
    #out = Activation('sigmoid', name='strong_out')(x)
    #audio_context = Model(inputs=x_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()

    y_start = Input(shape=(shape_in_y))
    y = y_start

    for _i, _cnt in enumerate((2, 2)):
        y = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(y)
        y = BatchNormalization(axis=-1)(y)
        #y = Activation('relu')(y)
        y = LeakyReLU()(y)
        #y = MayPooling2D(pool_size=(2,2), dim_ordering="th" )(y)
        y = MaxPooling2D(pool_size=2)(y)
        y = Dropout(dropoutrate)(y)

    y = Flatten()(y)
    #out = Activation('sigmoid', name='strong_out')(y)
    #audio_context = Model(inputs=y_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()# (None, 51200)

    z = Concatenate()([y, x])
    #z = Reshape((-1, 1))(z) # Why dose this not work?
    z = Reshape((67200, 1))(z)
    #out = Activation('sigmoid', name='strong_out')(z)
    #audio_context = Model(inputs=[x_start,y_start], outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary() # (None, 67200)

    # The Gru/recurrent portion
    # Get some knowledge
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    for r in (10,10):
        z = Bidirectional(
                GRU(r,
                    activation='tanh',
                    dropout=dropoutrate,
                    recurrent_dropout=dropoutrate,
                    return_sequences=True),
                merge_mode='concat')(z)
        for f in ((2,2)):
            z = TimeDistributed(Dense(f))(z)

    #z = Dropout(dropoutrate)(z)
    #z = TimeDistributed(Dense(880))(z)
    # arbitrary reshape may be a problem
    z = Flatten()(z)
    z = Dropout(dropoutrate)(z)
    z = Dense(1,activation = 'sigmoid')(z)

    #out = Activation('sigmoid', name='strong_out')(z)
    descrim = Model(inputs=[y_start, x_start], outputs=z)
    descrim.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #descrim.summary()
    #ce = audio_context
    return descrim, x_start

def build_frame_discriminator():
    # one aproch
    # https://towardsdatascience.com/an-approach-towards-convolutional-recurrent-neural-networks-f54cbeecd4a6

    # vary settings
    shape_in_x = int(64), int(128), int(3)
    #shape_out = int(8134), int(120)
    dropoutrate = 0.3

    x_start = Input(shape=(shape_in_x))
    x1_start = Input(shape=(shape_in_x))
    x = x_start

    for _i, _cnt in enumerate((2, 2)):
        x = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(x)
        x = BatchNormalization(axis=-1)(x)
        #x = Activation('relu')(x)
        x = LeakyReLU()(x)
        #x = MaxPooling2D(pool_size=(2,2), dim_ordering="th" )(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(dropoutrate)(x)

    x = Flatten()(x) # shape out = none 16000
    #out = Activation('sigmoid', name='strong_out')(x)
    #audio_context = Model(inputs=x_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()
    # I don't think we need y

    #y_start = Input(shape=(shape_in_y))
    #y = y_start

    #for _i, _cnt in enumerate((2, 2)):
    #    y = Conv2D(filters = 100, kernel_size=(2, 2), padding='same',)(y)
    #    y = BatchNormalization(axis=-1)(y)
    #    #y = Activation('relu')(y)
    #    y = LeakyReLU()(y)
    #    #y = MayPooling2D(pool_size=(2,2), dim_ordering="th" )(y)
    #    y = MaxPooling2D(pool_size=2)(y)
    #    y = Dropout(dropoutrate)(y)

    #y = Flatten()(y)
    #out = Activation('sigmoid', name='strong_out')(y)
    #audio_context = Model(inputs=y_start, outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary()# (None, 51200)

    # z is duplicate picture input
    z = Concatenate()([x, x])
    #z = Reshape((-1, 1))(z) # Why dose this not work?
    #z = Reshape((67200, 1))(z)
    #out = Activation('sigmoid', name='strong_out')(z)
    #audio_context = Model(inputs=[x_start,y_start], outputs=out)
    #audio_context.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #audio_context.summary() # (None, 67200)

    # The Gru/recurrent portion
    # Get some knowledge
    # http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    #for r in (10,10):
    #    z = Bidirectional(
    #            GRU(r,
    #                activation='tanh',
    #                dropout=dropoutrate,
    #                recurrent_dropout=dropoutrate,
    #                return_sequences=True),
    #            merge_mode='concat')(z)
    #    for f in ((2,2)):
    #        z = TimeDistributed(Dense(f))(z)

    #z = Dropout(dropoutrate)(z)
    #z = TimeDistributed(Dense(880))(z)
    # arbitrary reshape may be a problem
    #z = Flatten()(z)
    z = Dropout(dropoutrate)(z)
    z = Dense(1000,activation = 'sigmoid')(z)
    z = Dense(100,activation = 'sigmoid')(z)
    z = Dense(1,activation = 'sigmoid')(z)

    #out = Activation('sigmoid', name='strong_out')(z)
    descrim = Model(inputs=[x1_start, x_start], outputs=z)
    descrim.compile(optimizer='Adam', loss='binary_crossentropy',metrics = ['accuracy'])
    #descrim.summary()
    #ce = audio_context
    return descrim, x_start

#  discriminator, aud_input = build_discriminator()
#  #discriminator.summary()
#  generator, inputs = build_generator()
#  gd_joint = generator(inputs)
#
#  # For the combined model we will only train the generator
#  discriminator.trainable = False
#
#  # The discriminator takes generated images as input and determines validity
#  validity = discriminator([gd_joint, aud_input])
#
#  # The combined model  (stacked generator and discriminator)
#  # Trains the generator to fool the discriminator
#  inputs.append(aud_input)
#  combination = Model(inputs, validity)
#  combination.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
#
#
#  # Because of time needed, we use a Numpy preprocessed file.
#  training_vid_path = "/home/dl-group/data/Video/video1.npy"
#  training_audio_path = "/home/dl-group/data/Audio/audio1.npy"
#
#  print("Loading training data.")
#  training_vid = np.load(training_vid_path)
#  training_aud = np.load(training_audio_path)
#  # take the first video and make a matrix used as part of generator context
#  first_vid = np.zeros(training_vid.shape, dtype='float16')
#  for i in range(first_vid.shape[0]):
#      first_vid[i,:,:,:] = training_vid[0,:,:,:]
#
#  training_vid.shape
#  training_aud.shape
#
#  #
#  # Training block
#  #
#
#
#  cnt = 1
#  num_epoch = 200
#  num_epoch = 1
#  batch_size = 10
#  batch_size = 1
#  seed_size = 42
#  work_path = './'
#  save_freq = 10
#  save_freq = 1
#  metrics = []
#
#  for epoch in range(num_epoch):
#      idi = np.random.randint(0, training_vid.shape[0]-batch_size)
#      idx = list(range(idi, idi + batch_size))
#      x_real_vid = training_vid[idx]
#      x_real_aud = training_aud[idx]
#      image_context = first_vid[idx]
#      # Generate some images
#      # seed = np.random.normal(0,1,(batch_size,seed_size))
#      x_fake_vid = generator.predict([image_context, x_real_aud])
#      y_real = np.ones((batch_size))
#      y_fake = np.zeros((batch_size))
#      # Train discriminator on real and fake
#      discriminator_metric_real = discriminator.train_on_batch([x_real_vid, x_real_aud], y_real)
#      discriminator_metric_generated = discriminator.train_on_batch([x_fake_vid, x_real_aud], y_fake)
#      discriminator_metric = 0.5 * np.add(discriminator_metric_real,discriminator_metric_generated)
#      # Train generator on Calculate losses
#      # y_pred = discriminator.predict([x_real_aud, x_fake_vid])
#      generator_metric = combination.train_on_batch([image_context, x_real_aud, x_real_aud], y_real)
#      metrics.append([discriminator_metric, generator_metric])
#      # Time for an update?
#      # if epoch % save_freq == 0:
#      #     save_images(cnt, fixed_seed)
#      #     cnt += 1
#      #  if epoch % save_freq == 0:   print(f"Epoch {epoch}, Discriminator accuarcy: {discriminator_metric[1]}, Generator accuracy: {generator_metric[1]}")
#      if epoch > 1 and epoch % save_freq == 0:
#          generator.save(os.path.join(work_path,"generator-e{}.h5".format(epoch)))
#          discriminator.save(os.path.join(work_path,"discriminator-e{}.h5".format(epoch)))
#
#  import pandas as pd
#  pd.DataFrame(np.asmatrix([[m[0][0] for m in metrics], [m[0][1] for m in metrics], [m[1] for m in metrics]]).T)
