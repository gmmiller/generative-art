"""
This code adds the ability to RGB images using the gan architecture
"""

# Keras Machine Learning tools
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

# Tensorflow, Pyplot, Numpy
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Pillow for image manipulation
from PIL import Image
# glob for file paths/names
from glob import glob
import os

DIM_SIZE = 32
ALPHA = 0.2

class GAN():
    def __init__(self):
        # Define image specs
        self.img_rows = DIM_SIZE
        self.img_cols = DIM_SIZE
        self.channels = 3 #RGB
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # Stochastic optimizer
        adam = Adam(0.0002, 0.5)
        sgd = SGD(lr=ALPHA)

        # Build/compile DISCRIMINATOR
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

        # Build/compile GENERATOR
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy',
            optimizer=adam)

        # The generator takes noise as input and generated imgs
        # input white noise image vector of size 100x1
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model (stacked generator and discriminator)
        # Input noise => generated images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy',
            optimizer=adam)

    def build_generator(self):

        noise_shape = (100,)

        #Keras "Sequential" model for deep learning
        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()
        model.add(Flatten(input_shape=img_shape))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=ALPHA))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=ALPHA))

        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def get_image(self, image_path, width, height, mode):

        image = Image.open(image_path)
        if image.size != (width, height):
            image = image.resize([width, height])

        return np.array(image.convert(mode))

    def get_batch(self, image_files, width, height, mode):
        data_batch = np.array(
            [self.get_image(sample_file, width, height, mode) for sample_file in image_files])

        return data_batch

    def train(self, epochs, batch_size=128, save_interval=50):

        data_dir = './data'
        X_train = self.get_batch(glob(os.path.join(data_dir, '*.jpg')), DIM_SIZE, DIM_SIZE, 'RGB')


        #Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5


        half_batch = int(batch_size / 2)

        #Create lists for logging the losses
        d_loss_logs_r = []
        d_loss_logs_f = []
        g_loss_logs = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            #Append the logs with the loss values in each training step
            d_loss_logs_r.append([epoch, d_loss[0]])
            d_loss_logs_f.append([epoch, d_loss[1]])
            g_loss_logs.append([epoch, g_loss])

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)


            #Convert the log lists to numpy arrays
            d_loss_logs_r_a = np.array(d_loss_logs_r)
            d_loss_logs_f_a = np.array(d_loss_logs_f)
            g_loss_logs_a = np.array(g_loss_logs)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (1/2.5) * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("output/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, save_interval=100)