import warnings
import tensorflow as tf
# Don't fear the future
warnings.simplefilter(action='ignore', category=FutureWarning)
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import Model
import numpy as np
import cv2
strategy = tf.distribute.MirroredStrategy()

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

class TalkingGan():

    def make_vid(self, data, name):
        data = np.concatenate([data, data, data], axis=3)
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = 30
        shape = (data.shape[2], data.shape[1])
        out = cv2.VideoWriter(name + ".avi", fourcc, fps, shape)
        for f in data:
            out.write((f).astype('uint8'))
        out.release()

    def __init__(self):
        # Data shape entering the convolusion
        self.input_image = Input((72, 128, 1))
        self.input_ident = Input((72, 128, 1))
        self.input_audio = Input((320, 8, 1))
        self.filter_per_layers = [8, 16, 32, 64, 128, 256]


    def define_generator(self):
        f = self.filter_per_layers

        # #######################
        # ##    Context Def    ##
        # #######################

        a = audio_block(self.input_audio, f[0])
        a = audio_block(a, f[1])
        a = audio_block(a, f[2])
        a = audio_block(a, f[3])
        a = Flatten()(a)
        # a = Dense(1000)(a)
        # model = Model(input_audio, a)
        # model.summary()

        # #######################
        # ##   Generator Def   ##
        # #######################

        # Down bloc encoding
        c1, d = down_block(self.input_image, f[0])
        c2, d = down_block(d, f[1])
        c3, d = down_block(d, f[2])
        d = ZeroPadding2D(((0, 1), (0, 0)))(d)
        c4, d = down_block(d, f[3])
        d = ZeroPadding2D(((0, 1), (0, 0)))(d)
        c5, d = down_block(d, f[4])
        d = Flatten()(d)

        n = Concatenate()([d, a])
        n = Dense(1000)(n)
        n = LeakyReLU()(n)
        n = Reshape((1000, 1))(n)
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

        self.generator = Model([self.input_image, self.input_audio], u)
        self.generator.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])

    def define_ident_discriminator(self):
        f = self.filter_per_layers

        fr = Concatenate()([self.input_image, self.input_ident])
        fr = image_block(fr, f[0])
        fr = image_block(fr, f[1])
        fr = image_block(fr, f[2])
        fr = image_block(fr, f[3])
        fr = Flatten()(fr)

        fd = Dense(1000)(fr)
        fd = Dense(100)(fd)
        fd = Dense(10, activation='sigmoid')(fd)
        fd = Dense(1, activation='sigmoid')(fd)

        self.ident_discriminator = Model([self.input_image, self.input_ident], fd)
        self.ident_discriminator.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

    def define_seq_discriminator(self):
        f = self.filter_per_layers

        fr = image_block(self.input_image, f[0])
        fr = image_block(fr, f[1])
        fr = image_block(fr, f[2])
        fr = image_block(fr, f[3])
        fr = Flatten()(fr)

        a = audio_block(self.input_audio, f[0])
        a = audio_block(a, f[1])
        a = audio_block(a, f[2])
        a = audio_block(a, f[3])
        a = Flatten()(a)

        s = Concatenate()([fr, a])
        s = Dense(1000)(s)
        s = LeakyReLU()(s)
        s = Reshape((1000, 1))(s)
        s = GRU(8,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.3,
                return_sequences=True)(s)
        s = Flatten()(s)
        s = Dense(1000)(s)
        s = Dense(100)(s)
        s = Dense(10, activation='sigmoid')(s)
        s = Dense(1, activation='sigmoid')(s)

        self.seq_discriminator = Model([self.input_image, self.input_audio], s)
        self.seq_discriminator.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

    def stack_models(self):
        self.seq_discriminator.trainable = False
        self.seq_training_stack = Model(
            [self.input_image, self.input_audio],
            self.seq_discriminator([self.generator([self.input_image, self.input_audio]), self.input_audio]))
        self.seq_training_stack.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

        self.ident_discriminator.trainable = False
        self.ident_training_stack = Model(
            [self.input_image, self.input_ident, self.input_audio],
            self.seq_discriminator([self.generator([self.input_image, self.input_audio]), self.input_audio]))
        self.ident_training_stack.compile(optimizer=Adam(), loss=BinaryCrossentropy(), metrics=['accuracy'])

# strategy = tf.distribute.MirroredStrategy()
# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy
# strategy = tf.distribute.get_strategy()

# tgan.define_generator()

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

def get_class_training_set(fake_video, real_video):
    r = np.random.rand(fake_video.shape[0]) > 0.5
    clast_y = np.zeros(fake_video.shape[0])
    clast_y[r] = np.ones(real_video.shape[0])[r]
    clast_video = fake_video
    clast_video[r] = real_video[r]
    return clast_video, clast_y
    # clast_video = np.concatenate((
    #     fake_video,
    #     real_video
    # ))
    # clast_image = np.concatenate(
    #     (image_input, image_input),
    #     axis=0
    # )
    # clast_audio= np.concatenate(
    #     (audio, audio),
    #     axis=0
    # )
    # clast_video = np.concatenate(
    #     (clast_video, clast_image),
    #     axis=-1
    # )


def train(self, cnt=1, num_epoch=1000, batch_size=2**8, seed_size=42, work_path='./', save_freq=100):
    pass

cnt = 1
num_epoch = 12
batch_size = 2**7
seed_size = 42
work_path = './'
save_freq = 100

video, image, audio = load_video(path_video='./video1.npy', path_audio='./data/Audio/audio1.npy')
index, batch_video, batch_audio = get_batch_video(video, audio, batch_size)

tgan = TalkingGan()
with strategy.scope():
    tgan.define_generator()
    tgan.define_ident_discriminator()
    tgan.define_seq_discriminator()
    tgan.stack_models()

video.shape
images = get_image_context(video, image)
tgan.generator.fit([images, audio], video, batch_size=2**7)

# tgan.generator.fit([np.repeat(image, audio.shape[0]).shape, audio]video)
for j in range(num_epoch):
    for i in index:
        real_video = batch_video[i]
        real_audio = batch_audio[i]
        image_input = get_image_context(real_video, image)
        # Train the generator
        # tgan.generator.train_on_batch([image_input, real_audio], real_video)
        y_gen = np.ones((real_audio.shape[0], 1))
        # make a training set
        fake_video = tgan.generator.predict([image_input, real_audio])
        clast_video, clast_y = get_class_training_set(fake_video, real_video)
        # tgan.generator.train_on_batch([image_input, real_audio], real_video)
        loss = [None] * 5
        acur = [None] * 5
        loss[0], acur[0] = tgan.seq_training_stack.train_on_batch([image_input, real_audio], y_gen)
        loss[1], acur[1] = tgan.ident_training_stack.train_on_batch([real_video, image_input, real_audio], y_gen)
        loss[2], acur[2] = tgan.ident_discriminator.train_on_batch([clast_video, image_input], clast_y)
        loss[3], acur[3] = tgan.seq_discriminator.train_on_batch([clast_video, real_audio], clast_y)
        # loss[4], acur[4] = tgan.generator.train_on_batch([image_input, real_audio], real_video)
        print(loss)
        print(acur)

image.shape
pred = tgan.generator.predict([image_input,real_audio])
tgan.make_vid(video * 255, "vid1")
tgan.make_vid(pred * (255), "vid1_unet")

