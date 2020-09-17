
import warnings
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
# load all the layers
# from tensorflow.keras.layers import Activation
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Reshape
# from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import normalize
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.losses import MeanSquaredLogarithmicError
# load model
from tensorflow.keras import Model
# from tensorflow.keras.models import Sequential, load_model, Model
# load back end
# from tensorflow.keras import backend
# load optimizers
# from tensorflow.keras.optimizers import Adam
# load other tensor stuff
# import tensorflow as tf
# load other stuff stuff
import numpy as np
import cv2

#        ################
#        ##  Functions ##
#        ################

def audio_block(x, filters, kernal_size=(2, 2), padding='same', strides=1):
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(x)
    c = LeakyReLU()(c)
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(c)
    c = LeakyReLU()(c)
    return c

def image_block(x, filters, kernal_size=(3, 3), padding='same', strides=1):
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(x)
    c = LeakyReLU(alpha=0.1)(c)
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(c)
    c = LeakyReLU(alpha=0.1)(c)
    c = MaxPool2D((2, 2), (2, 2))(c)
    return c

def down_block(x, filters, kernal_size=(3, 3), padding='same', strides=1):
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(x)
    c = LeakyReLU(alpha=0.1)(c)
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(c)
    c = LeakyReLU(alpha=0.1)(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernal_size=(3, 3), padding='same', strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(concat)
    c = LeakyReLU(alpha=0.1)(c)
    c = Conv2D(filters, kernal_size, padding=padding, strides=strides)(c)
    c = LeakyReLU(alpha=0.1)(c)
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


class TalkingGan():
    # Data shape entering the convolusion
    input_image = Input((72, 128, 1))
    input_ident = Input((72, 128, 2))
    input_audio = Input((320, 8, 1))
    
    # Filters per layers
    f = [8, 16, 32, 64, 128, 256]
    
    # #######################
    # ##    Context Def    ##
    # #######################
    
    a = audio_block(input_audio, f[0])
    a = audio_block(a, f[1])
    a = audio_block(a, f[2])
    a = audio_block(a, f[3])
    a = Flatten()(a)
    # a = Dense(1000)(a)
    # a = Dense(1000)(a)
    # model = Model(input_audio, a)
    # model.summary()
    
    # #######################
    # ##   Generator Def   ##
    # #######################
    
    # Down bloc encoding
    c1, d = down_block(input_image, f[0])
    c2, d = down_block(d, f[1])
    c3, d = down_block(d, f[2])
    d = ZeroPadding2D(((0, 1), (0, 0)))(d)
    c4, d = down_block(d, f[3])
    d = ZeroPadding2D(((0, 1), (0, 0)))(d)
    c5, d = down_block(d, f[4])
    d = Flatten()(d)
    
    n = Concatenate()([d, a])
    n = Dense(3000)(n)
    n = Reshape((-1, 1))(n)
    n = GRU(8,
            activation='tanh',
            dropout=0.2,
            recurrent_dropout=0.3,
            return_sequences=True)(n)
    n = Flatten()(n)
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
    u = Conv2D(1, (1, 1), padding='same', activation='tanh')(u)
    
    generator = Model([input_image, input_audio], u)
    generator.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
    
    # #######################
    # ##    Context Def    ##
    # #######################
    fd = image_block(input_ident, f[0])
    fd = image_block(fd, f[1])
    fd = image_block(fd, f[2])
    fd = image_block(fd, f[3])
    fd = Flatten()(fd)
    fd = Dense(1000)(fd)
    fd = Dense(100)(fd)
    fd = Dense(10, activation='sigmoid')(fd)
    fd = Dense(1, activation='sigmoid')(fd)
    ident_discriminator = Model(input_ident, fd)
    ident_discriminator.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

    def train(self, cnt=1, num_epoch=1000, batch_size=2**8, seed_size=42, work_path='./', save_freq=100, metrics=[]):
        pass

def load_video(path_video, path_audio):
    video = np.load(path_video)
    audio = np.load(path_audio)
    audio_min = np.min(audio)
    audio = audio - audio_min
    audio_max = np.max(audio)
    audio = audio / audio_max
    video = video / 255
    image = video[0]
    audio = audio.astype('float16')
    return video, image, audio

def get_image_context(video, image):
    s = video.shape
    s = (s[0], s[1], s[2], 1)
    image_input = np.zeros(s, dtype='float16')
    for i in range(s[0]):
        image_input[i, :, :, :] = image
    return image_input

def get_video_context(video, image_input):
    s = video.shape
    s = (s[0], s[1], s[2], 2)
    image_contex = np.zeros(s, dtype='float16')
    image_input = image_input.reshape(s[0], s[1], s[2])
    video = video.reshape(s[0], s[1], s[2])
    image_contex[:, :, :, 0] = image_input
    image_contex[:, :, :, 1] = video
    return image_contex

def get_batch_video(video, audio, batch_size):
    batch_num = video.shape[0] // batch_size
    batch_video = np.array_split(video, batch_num)
    batch_audio = np.array_split(audio, batch_num)
    index = list(range(batch_num))
    np.random.shuffle(index)
    return index, batch_video, batch_audio

def get_ident_training_set(fake_video, real_video, image_input):
    ident_y = np.concatenate((
        np.zeros(fake_video.shape[0]),
        np.ones(real_video.shape[0])
    ))
    ident_video = np.concatenate(
        (real_video, fake_video),
        axis=0
    )
    ident_image = np.concatenate(
        (image_input, image_input),
        axis=0
    )
    ident_video = np.concatenate(
        (ident_video, ident_image),
        axis=-1
    )
    return ident_video, ident_y


cnt = 1
num_epoch = 12
batch_size = 2**7
seed_size = 42
work_path = './'
save_freq = 100
metrics = []

video, image, audio = load_video(path_video='./video1.npy', path_audio='./data/Audio/audio1.npy')
index, batch_video, batch_audio = get_batch_video(video, audio, batch_size)
tgan = TalkingGan()
for j in num_epoch:
    for i in index:
        real_video = batch_video[i]
        real_audio = batch_audio[i]
        image_input = get_image_context(real_video, image)
        # Train the generator
        tgan.generator.train_on_batch([image_input, real_audio], real_video)
        # make a training set
        fake_video = tgan.generator.predict([image_input, real_audio])
        ident_video, ident_y = get_ident_training_set(fake_video, real_video, image_input)
        tgan.ident_discriminator.train_on_batch(ident_video, ident_y)

pred = tgan.generator.predict([image, audio])
make_vid(video * 255, "vid1")
make_vid(pred * (255), "vid1_unet")

